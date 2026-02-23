import os
import io
import torch
import mlflow.pytorch
import dagshub
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from dotenv import load_dotenv
from ultralytics import YOLO

# ======================================
# LOAD ENV
# ======================================

load_dotenv()

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# ======================================
# CONNECT TO DAGSHUB
# ======================================

dagshub.init(
    repo_owner="sarthaksaitwal3",
    repo_name="final-road-detect-defect",
    mlflow=True,
)

MODEL_NAME = "Road_Defect_YOLOv8"
MODEL_VERSION = "1"

print("Loading model from registry...")

pytorch_model = mlflow.pytorch.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
pytorch_model.to(device)

yolo_model = YOLO()
yolo_model.model = pytorch_model

print("Model Loaded Successfully")

# ======================================
# FASTAPI APP
# ======================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files properly
app.mount("/static", StaticFiles(directory="static"), name="static")

from fastapi.responses import FileResponse

@app.get("/")
def read_index():
    return FileResponse("static/index.html")


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    lat: float = None,
    lon: float = None
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = yolo_model.predict(image, device=device, conf=0.4)

    detections = []

    for box in results[0].boxes:
        detections.append({
            "class": int(box.cls[0]),
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist()
        })

    print("GPS:", lat, lon)

    return {
        "detections": detections,
        "latitude": lat,
        "longitude": lon
    }