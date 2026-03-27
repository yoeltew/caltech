import io
import time

from fastapi import APIRouter, File, UploadFile, Depends
from sqlalchemy.orm import Session
from fastai.vision.all import PILImage

from app.database import get_db
from app.models import Prediction
from app.config import settings

router = APIRouter()

# Global variable for the model
learn = None

def get_learner():
    global learn
    if learn is None:
        from fastai.vision.all import load_learner
        from pathlib import Path
        model_path = Path(settings.models_dir) / "caltech_model.pkl"
        if model_path.exists():
            print(f"Lazy loading model from {model_path}...")
            try:
                learn = load_learner(model_path, cpu=True)
            except Exception as e:
                print(f"Lazy load failed: {e}")
        else:
            print(f"Model path not found: {model_path}")
    return learn


@router.post("/predict")
def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    start = time.perf_counter()

    # Ensure model is loaded
    model = get_learner()
    if model is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail="Model not loaded")

    image_bytes = file.file.read()
    img = PILImage.create(io.BytesIO(image_bytes))
    pred, idx, probs = model.predict(img)

    processing_time_ms = int((time.perf_counter() - start) * 1000)

    probabilities = {
        model.dls.vocab[i]: round(float(p), 4)
        for i, p in enumerate(probs)
    }

    # Log to database
    db_prediction = Prediction(
        prediction=str(pred),
        confidence=float(probs[idx]),
        probabilities=probabilities,
        processing_time_ms=processing_time_ms,
    )
    db.add(db_prediction)
    db.commit()

    return {
        "prediction": str(pred),
        "confidence": round(float(probs[idx]), 4),
        "probabilities": probabilities,
        "processing_time_ms": processing_time_ms,
    }
