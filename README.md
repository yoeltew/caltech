# Pet Classifier — Deployment Starter

Deploy your trained pet breed classifier as a web app.

## Quick Start

1. Copy your `pet_classifier.pkl` (from Lesson 2, Part 11) into the `models/` folder
2. Run: `docker-compose up --build`
3. Open `http://localhost:8501` for the Streamlit UI
4. Or call the API directly:

```bash
curl -X POST -F "file=@cat.jpg" http://localhost:8000/predict
```

## What's Inside

| File | Purpose |
|------|---------|
| `app/predict.py` | The ML inference code (load image → predict → return) |
| `frontend/app.py` | Streamlit UI for uploading images and viewing predictions |
| `app/main.py` | FastAPI app, loads model at startup |
| `app/models.py` | SQLAlchemy model for logging predictions to PostgreSQL |

## Note on PyTorch

The API container installs **CPU-only PyTorch** to keep the image small (~1.5 GB vs ~5 GB with CUDA). For a ResNet34 classifier, CPU inference at ~50-200ms per image is more than fast enough.

If you need GPU inference later (larger models, higher throughput), swap the `--extra-index-url` in `requirements-api.txt` to the CUDA version.
# caltech
