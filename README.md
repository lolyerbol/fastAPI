
# FastAPI + Celery Data Processing & ML Inference Service

This service provides asynchronous data ingestion, AI-powered screenshot analysis, and real-time machine learning inference using FastAPI, Celery, AWS S3, PostgreSQL, and external AI services.

---

## Architecture Overview

**Main components**
- FastAPI – REST API layer
- Celery – background task processing
- RabbitMQ – message broker
- AWS S3 – temporary object storage
- PostgreSQL – persistent storage
- ML model – surcharge prediction
- Gemini API – screenshot analysis

**Execution model**
- API endpoints are non-blocking
- Long-running jobs are executed by Celery workers
- Each job is tracked using a task ID

---

## Application Lifecycle

### Startup
- System initialization
- ML surcharge prediction model is loaded into memory

### Shutdown
- Graceful shutdown with logging

---

## API Endpoints

### Health Check

**GET /**

Confirms that the API is running.

```json
{
  "status": "API is active"
}
```

---

### Task Status

**GET /task-status/{task_id}**

Returns the current state of a Celery task.

**Possible states**

* `PENDING`
* `SUCCESS`
* `FAILURE`

```json
{
  "task_id": "uuid",
  "status": "success",
  "result": {}
}
```

---

## File Ingestion and ETL

### Endpoint

**POST /ingest**

Uploads a file and processes it through an ETL pipeline with validation against existing PostgreSQL tables.

### Processing Flow

1. File is uploaded through the API
2. File is saved to AWS S3
3. A Celery task is dispatched
4. Worker downloads the file from S3
5. System checks whether the table already exists in PostgreSQL

   * If the table exists → an error is raised
   * If the table does not exist → processing continues
6. ETL pipeline is executed
7. Data is loaded into a temporary table
8. Data is safely merged into the target table
9. Dimension tables are validated and uploaded if required

### Response

```json
{
  "status": "task_dispatched",
  "task_id": "celery-task-id"
}
```

---

## Screenshot Analysis with AI

### Endpoint

**POST /analyze-screenshots**

Uploads two screenshots and processes them asynchronously using an AI model.

### Processing Flow

1. Two screenshots are uploaded via the API
2. Files are stored in AWS S3 using a unique task identifier
3. Celery worker downloads the screenshots
4. Images are read and preprocessed
5. Images are sent to the Gemini AI API
6. Analysis results are generated
7. Results are saved to PostgreSQL
8. Task retries automatically on failure (up to 3 attempts)

### Response

```json
{
  "message": "Analysis started",
  "task_id": "celery-task-id",
  "check_status_url": "/task-status/{task_id}"
}
```

---

## ML Inference – Extra Surcharge Prediction

### Endpoint

**POST /predict/extra**

Runs synchronous machine learning inference for a single data entry.

### Processing Flow

1. Request payload is received
2. Preloaded ML model performs prediction
3. Prediction and probability are returned immediately
### Input
```python
{
  tpep_pickup_datetime: datetime,
  trip_distance: float,
  trip_duration_minutes: float,
  fare_amount: float,
  pickup_location_id: int

}
```

### Response

```json
{
  "status": "success",
  "prediction": "It has extra surcharge",
  "probability": 0.8732
}
```

---

## Celery Worker Tasks

### File Ingestion Task

**Task name:** `upload_file`

**Responsibilities**

* Download file from AWS S3
* Execute ETL pipeline
* Validate and upload dimension tables
* Safely insert data into PostgreSQL
### Response
```json
{
  "status": "success",
  "file": file_name,
  "period": str(period),
  "file_id": str(file_id),
  "s3_key": file_name,
  "version_id": versionID
}

```
---

### Screenshot Analysis Task

**Task name:** `analyze_screenshots_task`

**Responsibilities**

* Download screenshots from AWS S3
* Read and preprocess images
* Send images to Gemini AI API
* Persist analysis results to PostgreSQL
* Automatically retry on failure

---

## Error Handling and Reliability

* Temporary tables ensure safe database writes
* Automatic retries for AI-related tasks
* Centralized logging
* Clear task-level error reporting
* API remains responsive under heavy load

---
### Response
```json
{
  "status": "success", 
  "analysis": analysis_data,
  "message": "Structured analysis completed and stored in database"
}
```

## Technology Stack

* Python 3.11+
* FastAPI
* Celery
* RabbitMQ
* AWS S3 (boto3)
* PostgreSQL
* Gemini AI API
* scikit-learn or equivalent ML framework

---

## Notes

* ML model is loaded once at application startup
* Designed for scalability and fault tolerance
* Suitable for production asynchronous workloads

---

## Potential Enhancements

* Authentication and authorization
* API rate limiting
* Observability (Prometheus, OpenTelemetry)
* ML model versioning
* Data lineage and audit logging


