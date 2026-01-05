import uuid
from pathlib import Path
from fastapi import UploadFile
from sqlalchemy import text
from services.database import engine
from .etl import clean_df
import pandas as pd
from services.s3 import upload_file_to_s3
import hashlib
import re
from datetime import date
import csv
from io import StringIO, BytesIO


UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)


def extract_period(filename: str) -> date:
    match = re.search(r"(\d{4})-(\d{2})", filename)
    if not match:
        raise ValueError("Cannot extract period from filename")

    year, month = map(int, match.groups())
    return date(year, month, 1)


def calculate_checksum_bytesio(buffer: BytesIO) -> str:
    buffer.seek(0)
    md5 = hashlib.md5()
    md5.update(buffer.read())
    buffer.seek(0)
    return md5.hexdigest()

def psql_insert_copy(table, conn, keys, data_iter):
    """
    High-speed insertion using PostgreSQL COPY command via a memory buffer.
    """
    # Use the raw connection to access the cursor for COPY
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        # Handle schema and table names correctly with quotes
        columns = ', '.join('"{}"'.format(k) for k in keys)
        table_name = f'"{table.schema}"."{table.name}"' if table.schema else f'"{table.name}"'
        
        sql = f'COPY {table_name} ({columns}) FROM STDIN WITH CSV'
        cur.copy_expert(sql=sql, file=s_buf)

def ingest_file_pipeline(file: UploadFile):

    # --- подготовка buffer ---
    buffer = BytesIO(file.file.read())  # читаем весь UploadFile в память
    file.file.seek(0)  # на всякий случай

    file_name = file.filename
    checksum = calculate_checksum_bytesio(buffer)

    # --- извлекаем period из имени файла ---
    period = extract_period(file_name)
    file_id = uuid.uuid4()

    # --- работа с БД ---
    with engine.begin() as conn:
        # metadata table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS file_ingestion_metadata (
                id uuid primary key,
                file_name text not null,
                checksum text not null,
                period date not null,
                loaded_at timestamp default now(),
                s3_key text,
                s3_version_id text,
                status text not null default 'success',
                UNIQUE(file_name, checksum)
            )
        """))

        # проверка дубликата
        exists = conn.execute(
            text("""
                SELECT 1 FROM file_ingestion_metadata
                WHERE file_name = :fn AND checksum = :cs
            """), {"fn": file_name, "cs": checksum}
        ).fetchone()

        if exists:
            raise ValueError("File already ingested")

        # --- S3 upload ---
        s3_result = upload_file_to_s3(buffer, file_name)  # возвращает s3_key + version_id
        print(f"Uploaded to S3: {s3_result}")
        # --- чистка и подготовка DF ---
        df = pd.read_parquet(buffer) if file_name.endswith(".parquet") else pd.read_csv(buffer)
        df = clean_df(df)
        print(f"DataFrame cleaned, {len(df)} rows ready for ingestion")
        
        df["source_file_id"] = file_id

        # staging table
        conn.execute(text("""
            CREATE TEMP TABLE nyc_taxi_trips_staging
            (LIKE nyc_taxi_trips INCLUDING ALL)
            ON COMMIT PRESERVE ROWS
        """))

        df.to_sql(
            "nyc_taxi_trips_staging",
            conn,
            if_exists="append",
            index=False,
            method=psql_insert_copy,
            chunksize=100000
        )

        # merge staging → main
        conn.execute(text("""
            INSERT INTO nyc_taxi_trips
            SELECT * FROM nyc_taxi_trips_staging
        """))

        # commit metadata
        conn.execute(
            text("""
                INSERT INTO file_ingestion_metadata
                (id, file_name, checksum, period, s3_key, s3_version_id, status)
                VALUES (:id, :fn, :cs, :period, :s3_key, :vid, :status)
            """),
            {
                "id": file_id,
                "fn": file_name,
                "cs": checksum,
                "row_count": len(df),
                "period": period,
                "s3_key": s3_result["s3_key"],
                "vid": s3_result["version_id"],
                "status": "success",
            }
        )

    return {
        "status": "success",
        "file": file_name,
        "period": str(period),
        "file_id": str(file_id),
        "s3_key": s3_result["s3_key"],
        "version_id": s3_result["version_id"]
    }

def ingest_buffer_pipeline(buffer: BytesIO, file_name: str):
    """
    Ingest a file provided as an in-memory BytesIO (e.g., downloaded from S3).
    This performs the same ingestion steps as `ingest_file_pipeline` except the S3 upload.
    """
    buffer.seek(0)
    checksum = calculate_checksum_bytesio(buffer)

    period = extract_period(file_name)
    file_id = uuid.uuid4()

    with engine.begin() as conn:
        # metadata table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS file_ingestion_metadata (
                id uuid primary key,
                file_name text not null,
                checksum text not null,
                period date not null,
                loaded_at timestamp default now(),
                s3_key text,
                s3_version_id text,
                status text not null default 'success',
                UNIQUE(file_name, checksum)
            )
        """))

        # duplicate check
        exists = conn.execute(
            text("""
                SELECT 1 FROM file_ingestion_metadata
                WHERE file_name = :fn AND checksum = :cs
            """), {"fn": file_name, "cs": checksum}
        ).fetchone()

        if exists:
            raise ValueError(f"File already ingested: {file_name}")

        # read DF from buffer
        df = pd.read_parquet(buffer) if file_name.endswith(".parquet") else pd.read_csv(buffer)
        df = clean_df(df)
        df["source_file_id"] = file_id

        # staging
        conn.execute(text("""
            CREATE TEMP TABLE nyc_taxi_trips_staging
            (LIKE nyc_taxi_trips INCLUDING ALL)
            ON COMMIT PRESERVE ROWS
        """))

        df.to_sql(
            "nyc_taxi_trips_staging",
            conn,
            if_exists="append",
            index=False,
            method=psql_insert_copy,
            chunksize=100000
        )

        # merge
        conn.execute(text("""
            INSERT INTO nyc_taxi_trips
            SELECT * FROM nyc_taxi_trips_staging
        """))

        # commit metadata (no s3 upload here)
        conn.execute(
            text("""
                INSERT INTO file_ingestion_metadata
                (id, file_name, checksum, period, s3_key, s3_version_id, status)
                VALUES (:id, :fn, :cs, :period, :s3_key, :vid, :status)
            """),
            {
                "id": file_id,
                "fn": file_name,
                "cs": checksum,
                "period": period,
                "s3_key": None,
                "vid": None,
                "status": "success",
            }
        )

    return {
        "status": "success",
        "file": file_name,
        "period": str(period),
        "file_id": str(file_id),
        "s3_key": None,
        "version_id": None
    }
