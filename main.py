from fastapi import FastAPI, HTTPException, File, UploadFile
from pathlib import Path
from contextlib import asynccontextmanager
from google import genai
import uuid
import json
from PIL import Image
import io
from services.database import engine,ai_key
from services.s3 import download_to_bytes, list_files
from typing import List
from services.tables import (
    initialize_tables,
    gemini_analysis_results,
    check_and_upload_dims
)

from services.ingestion import ingest_buffer_pipeline, ingest_file_pipeline
from services.ml import load_surcharge_model, predict_surcharge, SurchargePredictionRequest
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs when the server starts
    logger.info("System Startup: Checking Dimension Tables...")
    initialize_tables()
    check_and_upload_dims()
    
    # Load ML models
    logger.info("Loading surcharge prediction model...")
    load_surcharge_model()
    
    # Configure Gemini API (support old and new google-genai interfaces)
    try:
        # older interface used configure()
        genai.configure(api_key=ai_key)
    except AttributeError:
        # new google-genai may expose a Client class or rely on env var
        try:
            Client = getattr(genai, "Client", None)
            if Client:
                genai.client = Client(api_key=ai_key)
            else:
                import os
                os.environ.setdefault("GENAI_API_KEY", ai_key)
        except Exception:
            import os
            os.environ.setdefault("GENAI_API_KEY", ai_key)
    yield
    # This runs when the server stops
    logger.info("System Shutdown")


# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"status": "API is active"}

@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
    ### to ingest a file and check/upload dimension tables
    try:
        res = ingest_file_pipeline(file) 
        check_and_upload_dims()
        return {"status": "success", "details": res}
    
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/analyze-screenshots", tags=["Analysis"])
async def analyze_screenshots(screenshot1: UploadFile = File(...), screenshot2: UploadFile = File(...)):
    """Endpoint to upload 2 screenshots, analyze them with Gemini, and store structured results in PostgreSQL"""
    try:
        # Read and convert images
        image1_bytes = await screenshot1.read()
        image2_bytes = await screenshot2.read()
        
        image1 = Image.open(io.BytesIO(image1_bytes))
        image2 = Image.open(io.BytesIO(image2_bytes))
        
        # Initialize Gemini client and model
        client = genai.Client(api_key=ai_key)
        
        # Create structured prompt for analysis
        prompt = """Analyze these two screenshots and provide analysis in a structured JSON format with exactly these 4 fields:
        {
            "screenshot1_analysis": "Detailed analysis of the first screenshot",
            "screenshot2_analysis": "Detailed analysis of the second screenshot",
            "comparison_analysis": "Detailed comparison between the two screenshots highlighting differences and similarities",
            "future_perspectives": "Based on the data shown, what are the future perspectives or trends you can identify?"
        }
        Return ONLY valid JSON, no additional text."""
        
        # Generate response with images
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image1, image2]
        )
        
        # Parse the JSON response
        response_text = response.text.strip()
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        analysis_data = json.loads(response_text)
        
        # Store in database with structured fields
        insert_stmt = gemini_analysis_results.insert().values(
            id=uuid.uuid4(),
            screenshot1_filename=screenshot1.filename,
            screenshot2_filename=screenshot2.filename,

            screenshot1_analysis=analysis_data.get("screenshot1_analysis"),
            screenshot2_analysis=analysis_data.get("screenshot2_analysis"),
            comparison_analysis=analysis_data.get("comparison_analysis"),
            future_perspectives=analysis_data.get("future_perspectives")
        )
        with engine.begin() as conn:
            conn.execute(insert_stmt)
        
        return {
            "status": "success", 
            "analysis": analysis_data,
            "message": "Structured analysis completed and stored in database"
        }
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Failed to parse Gemini response as JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(500, str(e))



@app.get("/s3-files")
async def get_s3_files():
    """
    Return list of files available in S3 under the expected prefix.
    If no files found, returns a message suitable for the UI: 'sorry, nothing to upload'.
    """
    try:
        files = list_files()
        if not files:
            return {"status": "empty", "message": "sorry, nothing to upload", "files": []}
        return {"status": "ok", "files": files}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/ingest-from-s3")
def ingest_from_s3(selected_keys: List[str]):
    """
    Accepts a JSON array of S3 keys (strings). Downloads each selected file from S3
    into memory and runs the ingestion pipeline.
    Example body:
    ["taxi-data/2023-01-file.csv", "taxi-data/2023-02-file.parquet"]
    """
    if not selected_keys:
        raise HTTPException(status_code=400, detail="No files selected")

    results = []
    for key in selected_keys:
        try:
            buffer = download_to_bytes(key)
            filename = Path(key).name
            res = ingest_buffer_pipeline(buffer, filename)
            results.append(res)
        except ValueError as e:
            # duplicate or domain validation
            results.append({"file": key, "status": "skipped", "reason": str(e)})
        except Exception as e:
            results.append({"file": key, "status": "error", "reason": str(e)})

    # After ingest(s) run dimension check/upload if needed
    try:
        check_and_upload_dims()
    except Exception:
        pass

    return {"status": "done", "results": results}

@app.post("/predict/surcharge", tags=["ML Inference"])
def predict_surcharge_endpoint(request: SurchargePredictionRequest):
    """
    Predict if a trip occurred during rush hour based on features.
    
    Returns:
    - is_rush_hour: 1 if predicted to be rush hour, 0 otherwise
    - probability_rush_hour: Confidence probability for rush hour
    - feature_values: Extracted features used for prediction
    """
    print("DEBUG: Received prediction request")
    logger.info(f"Received prediction request: {request}")
    try:
        result = predict_surcharge(request)
        return {
            "status": "success",
            "prediction": "It has extra surcharge" if result.get("has_extra_surcharge")==1 else "No extra surcharge",
            "probability": round(result.get("probability_extra_surcharge"), 4),
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")