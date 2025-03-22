import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

img_path = './assests/dogscats.jpg'
img = cv2.imread(img_path)

results = model(img_path)
for result in results:
    for box in result.boxes:
        print(box)


# Loop through the detections
for result in results:
    for box in result.boxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
        class_id = int(box.cls)  # Get class ID
        conf = box.conf[0].item()  # Get confidence score

        if (conf > 0.5):
            color = (0,0,245)  
            thickness = 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Put label text
            label = f"{model.names[class_id]}: {conf:.2f}"  # Class name and confidence
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, color, 2)

# Show the image with detections
cv2.imshow("Detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
