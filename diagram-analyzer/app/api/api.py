from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import uuid

from model.model_service import analyze

router = APIRouter(prefix="/api", tags=["api"])

DATA_DIR = Path("/app/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Health
# =========================================================

@router.get("/health")
def health():
    return {"status": "ok"}


# =========================================================
# CSV → PNG
# =========================================================

@router.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No file uploaded")

    file_id = uuid.uuid4()
    saved_path = DATA_DIR / f"{file_id}.csv"

    with saved_path.open("wb") as f:
        f.write(await file.read())

    result = analyze(file_path=str(saved_path), input_type="csv")

    return {"result": result}


# =========================================================
# Diagram → table
# =========================================================

@router.post("/upload-diagram")
async def upload_diagram(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No file uploaded")

    ext = Path(file.filename).suffix.lower()

    file_id = uuid.uuid4()
    saved_path = DATA_DIR / f"{file_id}{ext}"

    with saved_path.open("wb") as f:
        f.write(await file.read())

    result = analyze(file_path=str(saved_path), input_type="diagram")

    return {"result": result}
