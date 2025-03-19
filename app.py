import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# URL of the FastAPI backend
API_URL = "http://127.0.0.1:8001/predict/"

# Streamlit frontend
st.title("License Plate Reader")

# Allow the user to upload an image
uploaded_file = st.file_uploader("Upload an Image of a License Plate", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert image to bytes
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="JPEG")  # Convert to JPEG format
    img_byte_arr = img_byte_arr.getvalue()

    # Send request to FastAPI backend
    files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
    response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        prediction = response.json()
        if "license_plate" in prediction:
            st.write(f"Detected License Plate: {prediction['license_plate']}")
        else:
            st.write("No license plate detected.")
    else:
        st.write(f"Error connecting to the prediction API: {response.status_code}")


    #   streamlit run app.py