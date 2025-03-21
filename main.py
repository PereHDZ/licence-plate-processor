from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import numpy as np
from io import BytesIO
from PIL import Image
import os
from fastapi.middleware.cors import CORSMiddleware
import urllib.request
from ultralytics import YOLO

print("Running FastAPI from:", os.path.abspath(__file__))

MODEL_PATH = "best_e30_f0.pt"
FILE_ID = "1WOs2QNzmO_U9cgFHpESVFeb6RqIyrTLe"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

model = YOLO(MODEL_PATH)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLO model
model = YOLO("best_e30_f0.pt")

def group_and_sort_characters(detections, y_threshold=20):
    """
    Groups detected characters into rows based on y-coordinates and sorts within each row.
    
    Parameters:
        detections (list): List of tuples (x_center, y_center, text).
        y_threshold (int): Maximum y difference to consider characters in the same row.

    Returns:
        str: Ordered license plate text.
    """
    # Step 1: Sort by y_center first
    detections.sort(key=lambda d: d[1])

    # Step 2: Group characters into rows based on y proximity
    rows = []
    current_row = []

    for i, (x, y, text) in enumerate(detections):
        if i == 0:
            current_row.append((x, y, text))
            continue
        
        prev_x, prev_y, _ = detections[i - 1]

        # If the y difference is small, consider it part of the same row
        if abs(y - prev_y) < y_threshold:
            current_row.append((x, y, text))
        else:
            # Start a new row
            rows.append(current_row)
            current_row = [(x, y, text)]

    # Add the last row
    if current_row:
        rows.append(current_row)

    # Step 3: Sort characters within each row by x_center
    for row in rows:
        row.sort(key=lambda d: d[0])  # Sort left to right

    # Step 4: Flatten the sorted rows and extract the text
    ordered_characters = [char for row in rows for _, _, char in row]
    return ordered_characters

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    image = np.array(image)

    results = model(image)
    
    detections = []
    
    for result in results:
        for box in result.boxes:
            text = result.names[int(box.cls[0])]  # Character detected
            x_center = box.xywh[0][0].item()  # X-center of the bounding box
            y_center = box.xywh[0][1].item()  # Y-center of the bounding box
            
            # Store character with its position
            detections.append((x_center, y_center, text))

    # Sort characters correctly
    ordered_text = group_and_sort_characters(detections)

    return {"license_plate": ordered_text}
