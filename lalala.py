import cv2
import time
import numpy as np
import os
import serial


# Define the prices of the products
product_prices = {
    'banana': 3,
    'apple': 2,
    'orange': 4,
    'carrot': 1.5,
    'cell phone': 5,
}

# Get the current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set the paths of the YOLO model files and class names file
weights_path = os.path.join(script_dir, "yolov3.weights")
config_path = os.path.join(script_dir, "yolov3.cfg")
names_path = os.path.join(script_dir, "coco.names")

# Load YOLO
net = cv2.dnn.readNet(weights_path, config_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class names
classes = []
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

ser = serial.Serial('/dev/cu.usbserial-1120', 57600)

# Open the camera
cap = cv2.VideoCapture(0)

# Check if the camera is successfully opened
if not cap.isOpened():
    print("Failed to open the camera.")
    exit()

# Define button dimensions
button_width = 150
button_height = 40
button_padding = 10

close_button_x = 0
close_button_y = 0
start_button_x = 0
start_button_y = 0

# Initialize the flag for button clicks
close_button_clicked = False
start_button_clicked = False

# Dictionary to store product information
product_info = {
    'banana': {
        'price': 3,
        'description': 'Fresh banana from the local market.'
    },
    'apple': {
        'price': 2,
        'description': 'Crisp and juicy apple from the orchard.'
    },
    'orange': {
        'price': 4,
        'description': 'Sweet and tangy orange packed with Vitamin C.'
    },
    'carrot': {
        'price': 1.5,
        'description': 'Healthy and crunchy carrot for snacking.'
    },
    'cell phone': {
        'price': 5,
        'description': 'Latest model cell phone with advanced features.'
    },
}

def display_product_info(label):
    product = label.lower()
    if product in product_info:
        price = product_info[product]['price']
        description = product_info[product]['description']

        # Create a new window for product information
        info_window = np.zeros((200, 600, 3), np.uint8)
        cv2.putText(info_window, f"Product: {product}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_window, f"Price: {price} Lari", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_window, f"Description: {description}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the product information window
        cv2.imshow('Product Information', info_window)

def detect_objects(frame, weight):
    global close_button_clicked, start_button_clicked

    # Resize the frame for processing
    resized = cv2.resize(frame, (416, 416))

    # Normalize the image
    blob = cv2.dnn.blobFromImage(resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Set the input for the network
    net.setInput(blob)

    # Run forward pass through the network
    outs = net.forward(output_layers)

    # Process the outputs
    class_ids = []
    confidences = []
    boxes = []
    labels = []  # List to store the detected labels

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                # Calculate top-left coordinates of bounding box
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, width, height])
                labels.append(classes[class_id])  # Add the label to the list

    # Apply non-maximum suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the frame
    for i in range(len(boxes)):
        if i in indices:
            x, y, width, height = boxes[i]
            label = labels[i]  # Get the label from the list
            confidence = confidences[i]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Draw label with confidence
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Check if the detected object is one of the specified products
            if label.lower() in product_prices:
                price = product_prices[label.lower()]
                # Convert grams to kilograms
                weight_in_kg = float(weight) / 1000
                # Calculate total price
                total_price = round(price * weight_in_kg, 2)
                # Draw total price
                cv2.putText(frame, f"Total Price: {total_price:.2f} Lari", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                display_product_info(label)
                break
    else:
        # No object with price data detected, display 0 Lari
        cv2.putText(frame, "Total Price: 0 Lari", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the weight information on the frame
    cv2.putText(frame, f"Weight: {weight} g", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw buttons on the frame
    cv2.rectangle(frame, (close_button_x, close_button_y), (close_button_x + button_width, close_button_y + button_height), (255, 0, 0), -1)
    cv2.putText(frame, "Close", (close_button_x + 10, close_button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.rectangle(frame, (start_button_x, start_button_y), (start_button_x + button_width, start_button_y + button_height), (0, 255, 0), -1)
    cv2.putText(frame, "Start", (start_button_x + 10, start_button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame


# Main loop
while True:
    # Read the frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame from the camera.")
        break

    # Flip the frame horizontally for natural viewing
    frame = cv2.flip(frame, 1)

    # Get the weight from the serial port
    ser_data = ser.readline().decode().strip()
    if ser_data:
        weight = ser_data.split(',')[0]
        frame = detect_objects(frame, weight)

    # Display the frame
    cv2.imshow('Smart Cart', frame)

    # Check for button clicks
    if close_button_clicked:
        break

    if start_button_clicked:
        # Reset the flag
        start_button_clicked = False

    # Wait for ESC key press to exit
    if cv2.waitKey(1) == 27:
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
ser.close()
