import os
import socket
import boto3
import psycopg2
import tensorflow as tf
import pika

from fastapi import APIRouter
from pathlib import Path
from botocore.exceptions import BotoCoreError, ClientError

router = APIRouter()


@router.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "dependencies": {
            "s3": check_s3(),
            "postgresql": check_postgres(),
            "rabbitmq": check_rabbitmq(),
            "ml_model": check_ml_model(),
            "ai_service": check_ai_service(),
        },
    }

def check_s3():
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION"),
        )

        bucket = os.getenv("aws_bucket_name")
        s3.head_bucket(Bucket=bucket)

        return {"status": "healthy"}

    except (ClientError, BotoCoreError) as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }
    
def check_postgres():
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME")
    aws_bucket_name = os.getenv("aws_bucket_name")

    if not all([DB_USER, DB_PASSWORD, DB_NAME]):
        raise RuntimeError("Database credentials are not loaded from .env")

    dsn = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
    if not dsn:
        return {"status": "unhealthy", "error": "POSTGRES_DSN not set"}

    try:
        conn = psycopg2.connect(dsn, connect_timeout=3)
        conn.close()
        return {"status": "healthy"}

    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def check_rabbitmq():
    url = os.getenv("broker_url")
    if not url:
        return {"status": "unhealthy", "error": "RABBITMQ_URL not set"}

    try:
        params = pika.URLParameters(url)
        connection = pika.BlockingConnection(params)
        connection.close()
        return {"status": "healthy"}

    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def check_ml_model():
    model_path = os.getenv("TF_MODEL_PATH")
    if not model_path:
        return {
            "status": "unhealthy",
            "error": "TF_MODEL_PATH not set",
        }

    path = Path(model_path)

    if not path.exists():
        return {
            "status": "needs_training",
            "message": "ML model not found. Training required.",
        }

    try:
        tf.keras.models.load_model(path)
        return {"status": "healthy"}

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


def check_ai_service():
    api_key = os.getenv("ai_key")

    if not api_key:
        return {
            "status": "unhealthy",
            "error": "AI_API_KEY not set",
        }

    return {"status": "healthy"}
