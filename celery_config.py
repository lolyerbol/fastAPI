from celery import Celery
import os
broker_url = os.getenv("broker_url")
celery_app = Celery(
    "processor",
    broker=broker_url,
    backend="rpc://",
    include=["worker"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)