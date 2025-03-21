import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import warnings
import logging

# Suppress Streamlit and other warnings from being displayed on the app
logging.getLogger('streamlit').setLevel(logging.ERROR)  # Set Streamlit logger to show only errors
warnings.filterwarnings("ignore")  # Suppress other warnings

# URL of the FastAPI backend
API_URL = "https://licence-plate-processor.onrender.com/predict/"

# Streamlit frontend
st.title("License Plate Reader")

# Allow the user to upload an image
uploaded_file = st.file_uploader("Upload an Image of a License Plate", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    # Use `use_container_width` instead of the deprecated `use_column_width`
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Convert image to bytes
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="JPEG")  
    img_byte_arr = img_byte_arr.getvalue()

    # Send request to FastAPI backend
    files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
    response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        prediction = response.json()
        if "license_plate" in prediction:
            region = prediction['license_plate'][:-7]  # All elements except the last 7 (so we leave space for next sections)
            vehicle_type = prediction['license_plate'][-7]  
            year = prediction['license_plate'][-6:-4]  
            number = prediction['license_plate'][-4:]

            def convert_array(arr):
                if len(arr) == 1:
                    return str(arr[0])
                else:
                    return ' '.join(map(str, arr))

            st.write(f"Region: {convert_array(region)}")      
            st.write(f"Vehicle type: {vehicle_type}")
            st.write(f"License number: {''.join(map(str, year))}-{''.join(map(str,number))}")
        else:
            st.write("No license plate detected.")
    else:
        st.write(f"Error connecting to the prediction API: {response.status_code}")

    #   streamlit run app.py
