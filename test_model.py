from ultralytics import YOLO

model_path = './trained_model.pt'  # Path to your model
model = YOLO(model_path)  # Load the model

# Perform inference on the image
results = model('./img_to/tested/image copy 2.png')

# Access the first result (if only one image processed)
results[0].show()  # Display the annotated image
