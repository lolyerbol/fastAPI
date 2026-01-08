import boto3

from pathlib import Path
from io import BytesIO
from typing import List, Optional
from services.database import aws_bucket_name

def upload_file_to_s3(buffer: BytesIO, file_name: str) -> dict:
    s3_client = boto3.client("s3")
    bucket_name = aws_bucket_name
    key = f"taxi-data/{file_name}"
    buffer.seek(0)
    response = s3_client.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=buffer
    )

    return {"s3_key": key, "version_id": response.get("VersionId")}


DOWNLOAD_DIR = Path("/tmp/downloads")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

def download_from_s3(s3_key: str, version_id: str) -> Path:
    bucket_name = "nyct-bucket"
    local_path = DOWNLOAD_DIR / Path(s3_key).name
    s3_client = boto3.client("s3")
    s3_client.download_file(
        Bucket=bucket_name,
        Key=s3_key,
        Filename=str(local_path),
        ExtraArgs={"VersionId": version_id}
    )

    return local_path


def list_files(prefix: str = "taxi-data/") -> List[dict]:
    s3_client = boto3.client("s3")
    bucket_name = "nyct-bucket"
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    results = []
    for page in page_iterator:
        for obj in page.get("Contents", []):
            results.append({
                "Key": obj["Key"],
                "LastModified": obj["LastModified"].isoformat(),
                "Size": obj["Size"]
            })
    return results

def download_to_bytes(s3_key: str, version_id: Optional[str] = None) -> BytesIO:
    s3_client = boto3.client("s3")
    bucket_name = "nyct-bucket"
    params = {"Bucket": bucket_name, "Key": s3_key}
    if version_id:
        params["VersionId"] = version_id
    resp = s3_client.get_object(**params)
    b = BytesIO(resp["Body"].read())
    b.seek(0)
    return b