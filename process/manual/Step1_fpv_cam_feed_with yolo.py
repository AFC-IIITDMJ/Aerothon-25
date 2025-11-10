import cv2
import numpy as np

camera_idx = 0

# Load YOLOv4 configuration and weights
config_path = "process/shapes/yolov4-tiny-custom.cfg"
weights_path = "process/shapes/yolov4-tiny-custom_best.weights"
names_path = "process/shapes/obj.names"

# Load the class names
with open(names_path, "r") as f:
    classes = f.read().strip().split("\n")

# Load the YOLO model
net = cv2.dnn.readNet(weights_path, config_path)

# Load the webcam
cap = cv2.VideoCapture(camera_idx)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Prepare the frame for YOLO input
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Set the input for the YOLO network
    net.setInput(blob)

    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Run the forward pass to get the detections
    outs = net.forward(output_layers)

    # Process the detections
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Threshold for detecting object
                # Get bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to filter out overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the boxes and labels on the frame
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    center_x = width // 2

    cv2.line(frame, (center_x, 0), (center_x, height), (0, 255, 0), 2)

    horizon_y = height // 2
    cv2.line(frame, (0, horizon_y), (width, horizon_y), (0, 255, 0), 2)

    scale = 1.5
    frame = cv2.resize(frame, (int(width*scale), int(height*scale)))

    # Show the resulting frame
    cv2.imshow("YOLOv4 Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
