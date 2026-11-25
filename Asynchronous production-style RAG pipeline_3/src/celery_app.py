"""Celery application (Redis broker + result backend)."""
import os

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("rag", broker=REDIS_URL, backend=REDIS_URL, include=["src.tasks"])

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    task_acks_late=True,               # ack after the task finishes -> redelivered if the worker dies
    task_reject_on_worker_lost=True,   # requeue when the worker process is killed (OOM, SIGKILL)
    worker_prefetch_multiplier=1,      # long tasks: don't hoard messages
    task_time_limit=60 * 30,           # hard kill after 30 min (huge OCR jobs)
    task_soft_time_limit=60 * 25,
    result_expires=60 * 60 * 24,
    broker_transport_options={"visibility_timeout": 60 * 60},  # > longest task, avoids duplicate delivery
    beat_schedule={
        "requeue-stuck-documents": {"task": "src.tasks.requeue_stuck_documents", "schedule": 600.0},
    },
)
