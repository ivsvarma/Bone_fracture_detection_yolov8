import streamlit as st
import torch
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# Load the YOLO model
model_path = "./trained_model.pt"
model = YOLO(model_path)


def infer_image(image):
    results = model(image)
    fracture_detected = False  


    for result in results:
        if result.boxes is not None:
            for box in result.boxes.xyxy:
                st.write(f'Detected box: {box.numpy()}')
                fracture_detected = True

        if result.probs is not None:
            st.write(f'Class probabilities: {result.probs.numpy()}')

        
        annotated_img = result.plot()

        
        st.image(annotated_img, caption='Detection Result', use_column_width=True)

        # Save the result to a local directory
        output_path = Path('output/detection_result.jpg')
        cv2.imwrite(str(output_path), annotated_img)
        st.write(f"Saved annotated image to: {output_path}")


    if fracture_detected:
        st.error("Fracture detected! Please consult a healthcare professional immediately.")
        st.subheader("Suggested Remedies:")
        st.write("""
        - *Immobilization:* Use a cast or splint to prevent further movement.
        - *Pain Management:* Apply cold packs and consider over-the-counter pain relievers.
        - *Seek Medical Attention:* Visit an orthopedic specialist for X-rays and treatment.
        - *Rehabilitation:* Follow recommended physiotherapy for recovery.
        """)
    else:
        st.success("No fractures detected. The bone appears healthy.")


st.title("Bone Fracture Detection using YOLOv8")


uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

  
    if st.button("Predict"):
        infer_image(image)  