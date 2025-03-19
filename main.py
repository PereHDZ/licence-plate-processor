# from fastapi import FastAPI, UploadFile, File
# from typing import Annotated

# app = FastAPI()

# @app.post("/upload/")
# async def create_file(file: Annotated[bytes, File()]):
#     return {"file_size": len(file)}




from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import numpy as np
from io import BytesIO
from PIL import Image
import os

print("Running FastAPI from:", os.path.abspath(__file__))

app = FastAPI()

# Load the YOLO model
model = YOLO("best_e30_f0.pt")  

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    image = np.array(image)

    results = model(image)
    
    # Extract text from detected license plates
    plate_texts = []
    for result in results:
        for box in result.boxes:
            text = result.names[int(box.cls[0])]  
            plate_texts.append(text)
    
    return {"license_plate": plate_texts}

































# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from io import BytesIO
# from PIL import Image
# from ultralytics import YOLO

# app = FastAPI()

# # Enable CORS (Fixes frontend communication issues)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allow all origins
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load YOLO model
# try:
#     model = YOLO("C:/Users/tisor/Projects/licence-reader/licence-plate-processor/best_e30_f0.pt")
#     print("✅ Model loaded successfully!")
# except Exception as e:
#     print(f"❌ Error loading model: {e}")

# # Correct the FastAPI file handling
# @app.post("/predict/", summary="Upload an image")
# async def predict_license_plate(file: UploadFile = File(..., description="Upload an image file")):

#     try:
#         print("📷 Received image for processing")

#         # Read the uploaded image
#         image_bytes = await file.read()
#         image = Image.open(BytesIO(image_bytes)).convert("RGB")

#         # Perform YOLO inference
#         results = model(image)[0]  # Get first result
#         predictions = results.boxes.data.cpu().numpy()  # Extract bounding boxes

#         # Check if any license plate was detected
#         if len(predictions) > 0:
#             plate_text = "Detected Plate" 
#             print(f"✅ License Plate Detected: {plate_text}")
#             return {"license_plate": plate_text}
#         else:
#             print("🚫 No license plate detected")
#             return {"error": "No license plate detected"}

#     except Exception as e:
#         print(f"❌ Error during prediction: {e}")
#         return {"error": str(e)}


# # uvicorn main:app --reload --log-level debug
