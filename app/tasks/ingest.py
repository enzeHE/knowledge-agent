from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "knowledge_agent",
    broker=settings.redis_url,
    backend=settings.redis_url,
)


@celery_app.task(bind=True)
def ingest_document_task(self, file_path: str, doc_id: int):
    import subprocess
    import sys
    import os

    script = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "ingest_single.py")
    result = subprocess.run(
        [sys.executable, script, file_path, str(doc_id)],
        capture_output=True,
        text=True,
        cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
    )
    if result.returncode != 0:
        raise Exception(f"Ingest failed with code {result.returncode}: {result.stderr}")
    # returncode=0表示成功,即使stderr有警告也不影响
    return {"doc_id": doc_id, "output": result.stdout, "warnings": result.stderr if result.stderr else None}
