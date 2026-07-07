from fastapi import APIRouter, BackgroundTasks, Request

from app.db.session import SessionLocal
from app.schemas.common import SuccessResponse
from app.services.integration_service import integration_service

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


def _process_webhook(provider: str, external_user_id: str, file_content: bytes, filename: str) -> None:
    db = SessionLocal()
    try:
        athlete_id = integration_service.resolve_athlete_by_external(db, provider, external_user_id)
        if not athlete_id:
            return
        from app.models.activity import Upload
        from app.services.activity_service import activity_service
        from app.services.storage_service import storage_service

        upload = Upload(athlete_id=athlete_id, file_name=filename, file_path="", status="queued")
        db.add(upload)
        db.flush()
        path = storage_service.save_upload(upload.id, filename, file_content)
        upload.file_path = path
        db.commit()
        activity_service.process_upload_task(db, upload.id)
    finally:
        db.close()


@router.post("/garmin", response_model=SuccessResponse)
async def garmin_webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    str(payload.get("userId", payload.get("user_id", "")))
    return SuccessResponse(success=True)


@router.post("/wahoo", response_model=SuccessResponse)
async def wahoo_webhook(request: Request):
    await request.json()
    return SuccessResponse(success=True)


@router.post("/apple-health", response_model=SuccessResponse)
async def apple_health_webhook(request: Request):
    await request.json()
    return SuccessResponse(success=True)
