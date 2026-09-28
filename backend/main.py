from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil

from src.prediction.predict import predict_animal


app = FastAPI(
    title="Animal Classification API",
    description="Cattle and Buffalo Image Classification API",
    version="1.0"
)


# Frontend paths
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(exist_ok=True)


# Serve CSS and JavaScript files
app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static"
)


# Open frontend
@app.get("/app")
def frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


# API home
@app.get("/")
def home():
    return {
        "message": "Animal Classification API is running"
    }


ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Check file extension
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, PNG, and WEBP images are allowed."
        )

    # Safe filename
    safe_filename = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_filename

    # Save image
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Prediction
    label, confidence = predict_animal(str(file_path))

    return {
        "prediction": label,
        "confidence": round(float(confidence) * 100, 2)
    }