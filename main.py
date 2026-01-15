from fastapi import FastAPI, HTTPException, File, UploadFile, APIRouter, Depends
from pathlib import Path
from contextlib import asynccontextmanager
from worker import analyze_screenshots_task, upload_file
from services.s3 import upload_file_to_s3, aws_bucket_name
from services.ml import load_surcharge_model, predict_surcharge, SurchargePredictionRequest
import logging
from celery.result import AsyncResult
from io import BytesIO
import uuid
import boto3
from services.health import router as health_router

router = APIRouter()



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs when the server starts
    logger.info("System Startup: Checking Dimension Tables...")
#    initialize_tables()
#    check_and_upload_dims()
    
    # Load ML models
#    logger.info("Loading surcharge prediction model...")
    load_surcharge_model()
    
    yield
    # This runs when the server stops
    logger.info("System Shutdown")


# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

app.include_router(health_router)


@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    task_result = AsyncResult(task_id)
    if task_result.state == "PENDING":
        return {"task_id": task_id, "status": "pending"}
    elif task_result.state == "SUCCESS":
        return {"task_id": task_id, "status": "success", "result": task_result.result}
    elif task_result.state == "FAILURE":
        return {"task_id": task_id, "status": "failure", "error": str(task_result.result)}
    else:
        return {"task_id": task_id, "status": task_result.state}

@app.get("/")
async def root():
    return {"status": "API is active"}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    ### to ingest a file and check/upload dimension tables
    try:
        buffer = BytesIO(file.file.read())
        upload_file_to_s3(buffer,file.filename)
        res = upload_file.delay(f"taxi-data/{file.filename}")
        return {
            "status": "task_dispatched",
            "task_id": res.id
        }
    
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(500, str(e))


@app.post("/analyze-screenshots", tags=["Analysis"])
async def analyze_screenshots(screenshot1: UploadFile = File(...), screenshot2: UploadFile = File(...)):
# 1. Генерируем уникальные ключи для S3
    task_id = str(uuid.uuid4())
    
    s3_key1 = f"temp/{task_id}/{screenshot1.filename}"
    s3_key2 = f"temp/{task_id}/{screenshot2.filename}"
    s3_client = boto3.client("s3")
    # 2. Быстрая загрузка в S3 (не блокируем API надолго)
    s3_client.upload_fileobj(screenshot1.file, aws_bucket_name, s3_key1)
    s3_client.upload_fileobj(screenshot2.file, aws_bucket_name, s3_key2)

    # 3. Отправляем задачу в RabbitMQ
    task = analyze_screenshots_task.delay(s3_key1, s3_key2,screenshot1.filename,screenshot2.filename)

    return {
        "message": "Analysis started",
        "task_id": task.id,
        "check_status_url": f"/status/{task.id}"
    }




    

@app.post("/predict/extra", tags=["ML Inference"])
def predict_surcharge_endpoint(request: SurchargePredictionRequest = Depends()):
    """
    Predict if a trip occurred during rush hour based on features.
         
    Returns:
    - is_rush_hour: 1 if predicted to be rush hour, 0 otherwise
    - probability_rush_hour: Confidence probability for rush hour
    - feature_values: Extracted features used for prediction
    """
    
    logger.info(f"Received prediction request: {request}")
    try:
#        load_surcharge_model()
        result = predict_surcharge(request)
        return {
            "status": "success",
            "prediction": "It has extra surcharge" if result.get("has_extra_surcharge")==1 else "No extra surcharge",
            "probability": round(result.get("probability_extra_surcharge"), 4),
        }
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")