code :

from ultralytics import YOLO

def main():
   
    model = YOLO("yolov8x.pt")   

    
    results = model(
        "Input/images.jpg",  
        conf=0.5,
        imgsz=800,           
        save=True
    )

   
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            print(f"{class_name} : {conf:.2f}")

if __name__ == "__main__":
    main()
