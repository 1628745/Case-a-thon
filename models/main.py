from ultralytics import YOLO
import torch


def main():
    torch.cuda.is_available()
    torch.cuda.device_count()

    # Load a pretrained YOLOv12n model
    model = YOLO("yolo12n.pt")  

    # Train the model
    model.train(
        data="object_detection_dataset/data.yaml",  # path to your YAML file
        epochs=50,
        imgsz=640,
        batch=64,
        freeze=5,
        device="0",
        save=True
    )
    model.train(
        data="object_detection_dataset/data.yaml",  # path to your YAML file
        epochs=100,
        imgsz=640,
        batch=64,
        device="0",
        weights="runs/detect/train8/weights/last.pt"  # path to last saved weights
    )
def eval():
    model = YOLO("runs/detect/train11/weights/best.pt")
    metrics = model.val(save=True)  

if __name__ == "__main__":
    main()
    #eval()
