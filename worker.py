from services.database import aws_bucket_name
from services.s3 import download_to_bytes
from celery_config import celery_app
from botocore.exceptions import ClientError
from services.ingestion import ingest_buffer_pipeline
from services.ai import read_images
from services.tables import check_and_upload_dims
import boto3

BUCKET_NAME = aws_bucket_name
s3_client = boto3.client("s3")

@celery_app.task(name="upload_file")
def upload_file(s3_key: str, version_id: str = None) -> dict:
    """
    Celery task to download a file from S3, process it through the ETL pipeline,
    and return the result.
    """
    try:
        # Download file from S3
        file_buffer, version_id = download_to_bytes(s3_key, version_id)
        # Process the file through the ETL pipeline
        result = ingest_buffer_pipeline(file_buffer, s3_key, version_id)
        check_and_upload_dims()

        return {
            "status": "success",
            "s3_key": s3_key,
            "version_id": version_id,
            "result": result
        }
    except ClientError as e:
        return {
            "status": "error",
            "message": str(e)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    

@celery_app.task(name="analyze_screenshots_task", bind=True, max_retries=3)
def analyze_screenshots_task(self, s3_key1: str, s3_key2: str, filename1: str, filename2: str) -> dict:
    try:
        print("Downloading screenshots from S3...")    
        obj1 = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key1)
        obj2 = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key2)
        print("Screenshots downloaded. Reading images...")
        image1_bytes = obj1['Body'].read()
        image2_bytes = obj2['Body'].read()

        res = read_images(image1_bytes,filename1, image2_bytes, filename2)
        
        return {"status": "success", "analysis": res}
    except Exception as exc:
        raise self.retry(exc=exc, countdown=300)    