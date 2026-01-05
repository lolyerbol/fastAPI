from fastapi import FastAPI, HTTPException, File, UploadFile
from pathlib import Path
from contextlib import asynccontextmanager
import google.genai as genai
import uuid
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
    """Endpoint to upload 2 screenshots, analyze them with Gemini, and store the result in PostgreSQL"""
    try:
        # Read and convert images
        image1_bytes = await screenshot1.read()
        image2_bytes = await screenshot2.read()
        
        image1 = Image.open(io.BytesIO(image1_bytes))
        image2 = Image.open(io.BytesIO(image2_bytes))
        
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create prompt for analysis
        prompt = "Analyze these two screenshots and provide a detailed comparison or analysis of their content."
        
        # Generate response (defensive extraction)
        response = model.generate_content([prompt, image1, image2])
        analysis_result = getattr(response, 'text', None) or getattr(response, 'content', None) or str(response)

        # Store in database using a transaction
        insert_stmt = gemini_analysis_results.insert().values(
            id=uuid.uuid4(),
            screenshot1_filename=screenshot1.filename,
            screenshot2_filename=screenshot2.filename,
            analysis_result=analysis_result
        )
        with engine.begin() as conn:
            conn.execute(insert_stmt)
        
        return {
            "status": "success", 
            "analysis": analysis_result,
            "message": "Analysis completed and stored in database"
        }
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
    into memory and runs the ingestion pipeline WITHOUT re-uploading to S3.
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

    