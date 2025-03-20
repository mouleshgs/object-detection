import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Read the image
img_path = './assests/my-image.jpg'
img = cv2.imread(img_path)

# Perform inference
results = model(img_path)

# Loop through the detections
for result in results:
    for box in result.boxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
        class_id = int(box.cls)  # Get class ID
        conf = box.conf[0].item()  # Get confidence score

        # Draw rectangle around detected object
        color = (0, 255, 0)  # Green color for the bounding box
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
