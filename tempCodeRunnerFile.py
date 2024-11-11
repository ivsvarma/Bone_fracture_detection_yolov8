import torch
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the YOLO model
model_path = "./trained_model.pt"
model = YOLO(model_path)

def infer_webcam_with_matplotlib():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create a matplotlib figure and axis
    plt.ion()  # Interactive mode ON
    fig, ax = plt.subplots()

    # Display the first frame to initialize the plot
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from webcam.")
        cap.release()
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_display = ax.imshow(frame_rgb)
    ax.axis('off')  # Hide axes

    while True:
        # Read a new frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Run YOLO inference on the frame
        results = model(frame)

        # Draw bounding boxes
        for result in results:
            if result.boxes is not None:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box.numpy())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Convert BGR frame to RGB for matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_display.set_data(frame_rgb)  # Update the displayed image

        # Redraw the plot
        fig.canvas.flush_events()

        # Break the loop if 'q' is pressed
        if plt.waitforbuttonpress(timeout=0.01):  # Wait for key press (non-blocking)
            break

    # Release the webcam and close the plot
    cap.release()
    plt.close()

# Run the webcam inference function
infer_webcam_with_matplotlib()
