
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2


def train_model():
    
    print("Current working directory:", os.getcwd())
    print("Train images path:", "C:/Users/DELL/Desktop/MINOR PROJECT/data/train/images")
    print("Validation images path:", "C:/Users/DELL/Desktop/MINOR PROJECT/data/valid/images")


    train_images = './data/train/images'
    train_labels = './data/train/labels'

    test_images = './data/test/images'
    test_labels = './data/test/labels'

    val_images = './data/valid/images'
    val_labels = './data/valid/labels'

    assert os.path.exists(train_images), "Training images path does not exist!"
    assert os.path.exists(test_images), "Test images path does not exist!"
    assert os.path.exists(val_images), "Validation images path does not exist!"

    
    model = YOLO('yolov8s.pt')

    model.train(
        data='./data/data.yaml',  
        epochs=35,                
        imgsz=640,                
        seed=42,                  
        amp=False                 
    )

  
    model_path = 'trained_model.pt'
    model.save(model_path)
    print(f"Model saved to {model_path}")

    sample_image_path = './data/test/images/image1_1000_png.rf.a53c5e186c03961bf88075c6e3e94cf6.jpg'
    assert os.path.exists(sample_image_path), "Test image does not exist!"

    results = model(sample_image_path)

    
    results[0].plot()

    
    output_image_path = './detection_result.jpg'
    cv2.imwrite(output_image_path, results[0].img)
    print(f"Detection result saved to {output_image_path}")

    

    
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(results[0].img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    
    metrics = model.val(data='./data/data.yaml', split='val')
    print("Validation metrics:", metrics)

if __name__ == '__main__':
    train_model()