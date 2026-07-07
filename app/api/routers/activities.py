from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.pagination import paginate_offset
from app.core.security import get_current_athlete
from app.db.session import SessionLocal, get_db
from app.models.activity import Upload
from app.models.athlete import Athlete
from app.schemas.activity import (
    CreateActivityRequest,
    CropActivityRequest,
    DetailedActivity,
    Lap,
    PaginatedActivitiesResponse,
    PowerCurve,
    Split,
    Stream,
    UpdateActivityRequest,
)
from app.schemas.common import SuccessResponse
from app.services.activity_service import activity_service
from app.services.storage_service import storage_service

router = APIRouter(tags=["activities"])


def _run_upload(upload_id: str) -> None:
    db = SessionLocal()
    try:
        activity_service.process_upload_task(db, upload_id)
    finally:
        db.close()


@router.get("/activities", response_model=PaginatedActivitiesResponse)
def list_activities(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    sort_order: str = Query("newest"),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    stmt = activity_service.list_activities(db, athlete.id, page, per_page, sort_order)
    items, pagination = paginate_offset(db, stmt, page, per_page, unique=True)
    return PaginatedActivitiesResponse(
        pagination=pagination,
        items=[activity_service.to_summary(db, a, athlete.id) for a in items],
    )


@router.post("/activities")
def create_activity(
    body: CreateActivityRequest,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    data = body.model_dump()
    data["metrics"] = [m.model_dump() for m in body.metrics]
    success, activity_id, error = activity_service.create_manual(db, athlete.id, data)
    return {"success": success, "id": activity_id, "error_message": error}


@router.get("/activities/{activity_id}", response_model=DetailedActivity)
def get_activity(
    activity_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    activity = activity_service.get_activity(db, activity_id)
    return activity_service.to_detailed(db, activity, athlete.id)


@router.patch("/activities/{activity_id}", response_model=SuccessResponse)
def update_activity(
    activity_id: str,
    body: UpdateActivityRequest,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = activity_service.update(db, activity_id, athlete.id, body.model_dump(exclude_unset=True))
    return SuccessResponse(success=success, error_message=error)


@router.delete("/activities/{activity_id}", response_model=SuccessResponse)
def delete_activity(
    activity_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    success, error = activity_service.delete(db, activity_id, athlete.id)
    return SuccessResponse(success=success, error_message=error)


@router.post("/activities/{activity_id}/crop")
def crop_activity(
    activity_id: str,
    body: CropActivityRequest,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    activity = activity_service.get_activity(db, activity_id)
    if activity.athlete_id != athlete.id:
        return {"success": False, "error_message": "Forbidden"}
    return {"success": True, **activity_service.to_detailed(db, activity, athlete.id).model_dump()}


@router.get("/activities/{activity_id}/export")
def export_activity(
    activity_id: str,
    format: str = Query("gpx"),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    activity = activity_service.get_activity(db, activity_id)
    if activity.raw_file_path:
        return FileResponse(activity.raw_file_path, filename=f"activity.{format}")
    return PlainTextResponse("No raw file available", status_code=404)


@router.get("/activities/{activity_id}/laps", response_model=list[Lap])
def get_laps(activity_id: str, db: Session = Depends(get_db)):
    activity = activity_service.get_activity(db, activity_id)
    return activity_service.to_detailed(db, activity, None).laps


@router.get("/activities/{activity_id}/power-curve", response_model=PowerCurve)
def get_power_curve(activity_id: str, db: Session = Depends(get_db)):
    return activity_service.get_power_curve(db, activity_id)


@router.get("/activities/{activity_id}/splits", response_model=list[Split])
def get_splits(activity_id: str, db: Session = Depends(get_db)):
    return activity_service.get_splits(db, activity_id)


@router.get("/activities/{activity_id}/streams", response_model=list[Stream])
def get_streams(
    activity_id: str,
    streams: str = Query(""),
    resolution: str = Query("high"),
    db: Session = Depends(get_db),
):
    keys = streams.split(",") if streams else []
    return activity_service.get_streams(db, activity_id, keys, resolution)


@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    content = await file.read()
    upload = Upload(athlete_id=athlete.id, file_name=file.filename or "upload.fit", file_path="", status="queued")
    db.add(upload)
    db.flush()
    path = storage_service.save_upload(upload.id, upload.file_name, content)
    upload.file_path = path
    db.commit()
    if get_settings().sync_uploads:
        activity_service.process_upload_task(db, upload.id)
    else:
        background_tasks.add_task(_run_upload, upload.id)
    return {"upload_id": upload.id, "success": True, "error_message": None}


@router.get("/upload/{upload_id}")
def get_upload_status(
    upload_id: str,
    athlete: Athlete = Depends(get_current_athlete),
    db: Session = Depends(get_db),
):
    upload = db.get(Upload, upload_id)
    if not upload or upload.athlete_id != athlete.id:
        return {"status": "not_found", "activity_id": None, "error_message": "Upload not found"}
    return {
        "status": upload.status,
        "activity_id": upload.activity_id,
        "error_message": upload.error_message,
    }
