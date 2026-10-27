from ultralytics import YOLO

def main():
    # 1. Pre-trained YOLOv8 small model load karein (aap yolov8n.pt bhi use kar sakte hain)
    model = YOLO('yolov8s.pt')

    # 2. Dataset train karna shuru karein
    results = model.train(
        data='data.yaml',    # Aapke folder mein data.yaml maujud hai
        epochs=20,           # Apni zaroorat ke mutabiq epochs barha ya ghata sakte hain
        imgsz=640,           # Image size
        batch=16,            # Batch size (agar GPU memory kam ho toh 8 ya 4 kar dein)
        device=0,            # Agar GPU use karna hai toh 0, warna 'cpu' likh dein
        workers=4            # Data loading workers
    )

    print("Training successfully completed!")

if __name__ == '__main__':
    main()