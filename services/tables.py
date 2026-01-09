
import pandas as pd
import uuid
from sqlalchemy import (
    MetaData, Table, Column,
    String, Integer, Date, DateTime,
    Numeric, Text, inspect, text
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from services.database import engine
metadata = MetaData()

file_ingestion_metadata = Table(
    "file_ingestion_metadata",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("file_name", String, nullable=False),
    Column("checksum", String, nullable=False),
    Column("period", Date, nullable=False),
    Column("row_count", Integer),
    Column("status", String, nullable=False),
    Column("loaded_at", DateTime, server_default=func.now()),
)

nyc_taxi_trips = Table(
    "nyc_taxi_trips",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("tpep_pickup_datetime", DateTime),
    Column("tpep_dropoff_datetime", DateTime),
    Column("passenger_count", Integer),
    Column("trip_distance", Numeric),
    Column("store_and_fwd_flag", Text),
    Column("pickup_location_id", Integer),
    Column("dropoff_location_id", Integer),
    
    Column("payment_type", Integer),
    Column("fare_amount", Numeric),
    Column("extra", Numeric),
    Column("mta_tax", Numeric),
    Column("tip_amount", Numeric),
    Column("tolls_amount", Numeric),
    Column("improvement_surcharge", Numeric),
    Column("total_amount", Numeric),
    Column("vendor_id", Integer),
    Column("ratecode_id", Integer),
    Column("trip_duration_minutes", Numeric),
    Column("source_file_id", UUID(as_uuid=True), nullable=False),
)

gemini_analysis_results = Table(
    "gemini_analysis_results",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column("screenshot1_filename", String, nullable=False),
    Column("screenshot2_filename", String, nullable=False),

    Column("screenshot1_analysis", Text, nullable=True),
    Column("screenshot2_analysis", Text, nullable=True),
    Column("comparison_analysis", Text, nullable=True),
    Column("future_perspectives", Text, nullable=True),
    Column("created_at", DateTime, server_default=func.now()),
)


def initialize_tables():
    inspector = inspect(engine)
    # Create any missing tables/columns according to metadata
    metadata.create_all(engine)

    # Handle legacy column names if present: rename PULocationID/DOLocationID -> pickup_location_id/dropoff_location_id
    with engine.begin() as conn:
        if inspector.has_table('nyc_taxi_trips'):
            cols = [c['name'] for c in inspector.get_columns('nyc_taxi_trips')]
            if 'PULocationID' in cols and 'pickup_location_id' not in cols:
                conn.execute(text('ALTER TABLE nyc_taxi_trips RENAME COLUMN "PULocationID" TO pickup_location_id;'))
            if 'DOLocationID' in cols and 'dropoff_location_id' not in cols:
                conn.execute(text('ALTER TABLE nyc_taxi_trips RENAME COLUMN "DOLocationID" TO dropoff_location_id;'))
    # Ensure file_ingestion_metadata has s3_key and s3_version_id columns
    with engine.begin() as conn:
        if inspector.has_table('file_ingestion_metadata'):
            meta_cols = [c['name'] for c in inspector.get_columns('file_ingestion_metadata')]
            if 's3_key' not in meta_cols:
                conn.execute(text("ALTER TABLE file_ingestion_metadata ADD COLUMN s3_key text;"))
            if 's3_version_id' not in meta_cols:
                conn.execute(text("ALTER TABLE file_ingestion_metadata ADD COLUMN s3_version_id text;"))



# __________________________Dimensional Tables_______________________________
def check_and_upload_dims():
    # 1. Define the DataFrames inside the function for memory efficiency
    city_zones = pd.read_csv('./data/taxi_zone_lookup.csv')
    city_zones.rename(columns={'LocationID': 'location_id', 'Borough': 'borough', 'Zone': 'zone', 'service_zone': 'service_zone'}, inplace=True)
    payment_type_df = pd.DataFrame({
        "payment_type": [0,1,2,3,4,5,6],
        "payment_type_name": ["Flex Fare trip","Credit card","Cash","No charge","Dispute","Unknown","Voided trip"]
    })

    vendor_df = pd.DataFrame({
        "vendor_id": [1,2,6,7],
        "vendor_name": ["Creative Mobile Technologies, LLC", "Curb Mobility, LLC","Myle Technologies Inc","Helix"]
    })

    rate_code_df = pd.DataFrame({
        "rate_code_id": [1,2,3,4,5,6,99], 
        "rate_code_name": ["Standard rate","JFK","Newark","Nassau or Westchester","Negotiated fare","Group ride","Null/unknown"]
    })

    # Map table names to their respective DataFrames
    dim_tables = {
        "dim_payment_type": payment_type_df,
        "dim_vendor": vendor_df,
        "dim_rate_code": rate_code_df,
        "dim_city_zone":city_zones
    }

    inspector = inspect(engine)
    
    with engine.connect() as conn:
        for table_name, df in dim_tables.items():
            # Check if the table exists in PostgreSQL
            if not inspector.has_table(table_name):
                print(f"Table {table_name} not found. Uploading...")
                df.to_sql(table_name, engine, if_exists='fail', index=False)
                
                # Optional: Add Primary Key so you can link to fact tables later
                pk_col = df.columns[0]
                conn.execute(text(f"ALTER TABLE {table_name} ADD PRIMARY KEY ({pk_col});"))
                conn.commit()
            else:
                # Requirement: Print specific message if it exists
                print(f"{table_name} already exists")

def prepare_ingestion_artifacts(file_name, buffer=None, file=None):
    """
    Prepares buffer, file_name, checksum, period, file_id for ingestion pipelines.
    If file is provided (UploadFile), reads buffer from it. If buffer is provided, uses it.
    Returns: buffer, file_name, checksum, period, file_id
    """
    import uuid
    from io import BytesIO
    from services.ingestion import extract_period, calculate_checksum_bytesio
    if file is not None:
        buffer = BytesIO(file.file.read())
        file.file.seek(0)
        file_name = file.filename
    elif buffer is not None:
        buffer.seek(0)
    else:
        raise ValueError("Either file or buffer must be provided")
    checksum = calculate_checksum_bytesio(buffer)
    period = extract_period(file_name)
    
    file_id = uuid.uuid4()
    
    with engine.begin() as conn:
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
        exists = conn.execute(
            text("""
SELECT 1 FROM file_ingestion_metadata
WHERE file_name = :fn AND checksum = :cs
"""), {"fn": file_name, "cs": checksum}
        ).fetchone()
        if exists:
            raise ValueError(f"File already ingested: {file_name}")
    return  checksum, period, file_id