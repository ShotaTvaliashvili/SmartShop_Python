

# import cv2
# import time
# import numpy as np
# import os

# # Define the prices of the products
# product_prices = {
#     'banana': 3,
#     'apple': 2,
#     'orange': 4,
#     'carrot': 1.5,
#     'cell phone': 5,
# }

# # Get the current directory
# script_dir = os.path.dirname(os.path.abspath(__file__))

# # Set the paths of the YOLO model files and class names file
# weights_path = os.path.join(script_dir, "yolov3.weights")
# config_path = os.path.join(script_dir, "yolov3.cfg")
# names_path = os.path.join(script_dir, "coco.names")

# # Load YOLO
# net = cv2.dnn.readNet(weights_path, config_path)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# # Load class names
# classes = []
# with open(names_path, "r") as f:
#     classes = [line.strip() for line in f.readlines()]

# # Get output layer names
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# # Open the camera
# cap = cv2.VideoCapture(0)

# # Check if the camera is successfully opened
# if not cap.isOpened():
#     print("Failed to open the camera.")
#     exit()

# # Define button dimensions
# button_width = 150
# button_height = 40
# button_padding = 10

# # Initialize the flag for button click
# close_button_clicked = False

# def detect_objects(frame, weight):
#     global close_button_clicked

#     # Resize the frame for processing
#     resized = cv2.resize(frame, (416, 416))

#     # Normalize the image
#     blob = cv2.dnn.blobFromImage(resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

#     # Set the input for the network
#     net.setInput(blob)

#     # Run forward pass through the network
#     outs = net.forward(output_layers)

#     # Process the outputs
#     class_ids = []
#     confidences = []
#     boxes = []
#     labels = []  # List to store the detected labels

#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]

#             if confidence > 0.5:
#                 # Object detected
#                 center_x = int(detection[0] * frame.shape[1])
#                 center_y = int(detection[1] * frame.shape[0])
#                 width = int(detection[2] * frame.shape[1])
#                 height = int(detection[3] * frame.shape[0])

#                 # Calculate top-left coordinates of bounding box
#                 x = int(center_x - width / 2)
#                 y = int(center_y - height / 2)

#                 class_ids.append(class_id)
#                 confidences.append(float(confidence))
#                 boxes.append([x, y, width, height])
#                 labels.append(classes[class_id])  # Add the label to the list

#     # Apply non-maximum suppression to remove overlapping boxes
#     indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#     # Draw bounding boxes and labels on the frame
#     for i in range(len(boxes)):
#         if i in indices:
#             x, y, width, height = boxes[i]
#             label = labels[i]  # Get the label from the list
#             confidence = confidences[i]

#             # Draw bounding box
#             cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

#             # Draw label with confidence
#             cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#             # Check if the detected object is one of the specified products
#             if label.lower() in product_prices:
#                 price = product_prices[label.lower()]
#                 # Convert grams to kilograms
#                 weight_in_kg = float(weight) / 1000
#                 # Calculate total price
#                 total_price = round(price * weight_in_kg, 2)
#                 # Draw total price
#                 cv2.putText(frame, f"Total Price: {total_price:.2f} Lari", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                 break
#     else:
#         # No object with price data detected, display 0 Lari
#         cv2.putText(frame, "Total Price: 0 Lari", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     # Display the weight information on the frame
#     cv2.putText(frame, 'Weight: ' + weight + ' gr', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     # Draw "Close" button
#     close_button_x = button_padding
#     close_button_y = frame.shape[0] - button_height - button_padding
#     cv2.rectangle(frame, (close_button_x, close_button_y), (close_button_x + button_width, close_button_y + button_height), (0, 0, 255), cv2.FILLED)
#     text_width, text_height = cv2.getTextSize("Close", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
#     text_x = close_button_x + (button_width - text_width) // 2
#     text_y = close_button_y + (button_height + text_height) // 2
#     cv2.putText(frame, "Close", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     # Check if the "Close" button is clicked
#     if close_button_clicked:
#         # Release the camera and close all windows
#         cap.release()
#         cv2.destroyAllWindows()
#         return

#     # Display the frame with bounding boxes, labels, and weight information
#     cv2.imshow('Camera Feed', frame)

# def on_mouse_click(event, x, y, flags, param):
#     global close_button_clicked

#     # Check if the left mouse button is clicked
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # Check if the click is within the "Close" button bounds
#         if x >= button_padding and x <= button_padding + button_width and y >= cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - button_padding - button_height and y <= cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - button_padding:
#             close_button_clicked = True

# # Register the mouse click event
# cv2.namedWindow('Camera Feed')
# cv2.setMouseCallback('Camera Feed', on_mouse_click)

# while True:
#     # Read a frame from the camera
#     ret, frame = cap.read()

#     # Check if the frame is successfully read
#     if not ret:
#         print("Failed to read a frame.")
#         break

#     # Convert the frame to grayscale for weight detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Perform weight detection logic here...
#     # ...

#     # Call the detect_objects function
#     detect_objects(frame, '1000')

#     # Check if the "Close" button is clicked
#     if close_button_clicked:
#         break

#     # Wait for a key press and check if it is 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the camera and close all windows
# cap.release()
# cv2.destroyAllWindows()



# import cv2
# import time
# import numpy as np
# import os

# # Define the prices of the products
# product_prices = {
#     'banana': 3,
#     'apple': 2,
#     'orange': 4,
#     'carrot': 1.5,
#     'cell phone': 5,
# }

# # Get the current directory
# script_dir = os.path.dirname(os.path.abspath(__file__))

# # Set the paths of the YOLO model files and class names file
# weights_path = os.path.join(script_dir, "yolov3.weights")
# config_path = os.path.join(script_dir, "yolov3.cfg")
# names_path = os.path.join(script_dir, "coco.names")

# # Load YOLO
# net = cv2.dnn.readNet(weights_path, config_path)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# # Load class names
# classes = []
# with open(names_path, "r") as f:
#     classes = [line.strip() for line in f.readlines()]

# # Get output layer names
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# # Open the camera
# cap = cv2.VideoCapture(0)

# # Check if the camera is successfully opened
# if not cap.isOpened():
#     print("Failed to open the camera.")
#     exit()

# # Define button dimensions
# button_width = 150
# button_height = 40
# button_padding = 10

# # Initialize the flag for button clicks
# close_button_clicked = False
# start_button_clicked = False

# def detect_objects(frame, weight):
#     global close_button_clicked, start_button_clicked

#     # Resize the frame for processing
#     resized = cv2.resize(frame, (416, 416))

#     # Normalize the image
#     blob = cv2.dnn.blobFromImage(resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

#     # Set the input for the network
#     net.setInput(blob)

#     # Run forward pass through the network
#     outs = net.forward(output_layers)

#     # Process the outputs
#     class_ids = []
#     confidences = []
#     boxes = []
#     labels = []  # List to store the detected labels

#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]

#             if confidence > 0.5:
#                 # Object detected
#                 center_x = int(detection[0] * frame.shape[1])
#                 center_y = int(detection[1] * frame.shape[0])
#                 width = int(detection[2] * frame.shape[1])
#                 height = int(detection[3] * frame.shape[0])

#                 # Calculate top-left coordinates of bounding box
#                 x = int(center_x - width / 2)
#                 y = int(center_y - height / 2)

#                 class_ids.append(class_id)
#                 confidences.append(float(confidence))
#                 boxes.append([x, y, width, height])
#                 labels.append(classes[class_id])  # Add the label to the list

#     # Apply non-maximum suppression to remove overlapping boxes
#     indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#     # Draw bounding boxes and labels on the frame
#     for i in range(len(boxes)):
#         if i in indices:
#             x, y, width, height = boxes[i]
#             label = labels[i]  # Get the label from the list
#             confidence = confidences[i]

#             # Draw bounding box
#             cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

#             # Draw label with confidence
#             cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#             # Check if the detected object is one of the specified products
#             if label.lower() in product_prices:
#                 price = product_prices[label.lower()]
#                 # Convert grams to kilograms
#                 weight_in_kg = float(weight) / 1000
#                 # Calculate total price
#                 total_price = round(price * weight_in_kg, 2)
#                 # Draw total price
#                 cv2.putText(frame, f"Total Price: {total_price:.2f} Lari", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                 break
#     else:
#         # No object with price data detected, display 0 Lari
#         cv2.putText(frame, "Total Price: 0 Lari", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     # Display the weight information on the frame
#     cv2.putText(frame, 'Weight: ' + weight + ' gr', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     # Draw "Close" button
#     close_button_x = button_padding
#     close_button_y = frame.shape[0] - button_height - button_padding
#     cv2.rectangle(frame, (close_button_x, close_button_y), (close_button_x + button_width, close_button_y + button_height), (0, 0, 255), cv2.FILLED)
#     text_width, text_height = cv2.getTextSize("Close", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
#     text_x = close_button_x + (button_width - text_width) // 2
#     text_y = close_button_y + (button_height + text_height) // 2
#     cv2.putText(frame, "Close", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     # Draw "Start Buying" button
#     start_button_x = close_button_x + button_width + button_padding
#     start_button_y = close_button_y
#     cv2.rectangle(frame, (start_button_x, start_button_y), (start_button_x + button_width, start_button_y + button_height), (0, 255, 0), cv2.FILLED)
#     text_width, text_height = cv2.getTextSize("Start Buying", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
#     text_x = start_button_x + (button_width - text_width) // 2
#     text_y = start_button_y + (button_height + text_height) // 2
#     cv2.putText(frame, "Start Buying", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


#     # Check if the "Close" button is clicked
#     if close_button_clicked:
#         # Release the camera and close all windows
#         cap.release()
#         cv2.destroyAllWindows()
#         return

#     # Check if the "Start Buying" button is clicked
#     if start_button_clicked:
#         return
#         # Perform the buying logic here...
#         # ...

#     # Display the frame with bounding boxes, labels, and weight information
#     cv2.imshow('Camera Feed', frame)

# def on_mouse_click(event, x, y, flags, param):
#     global close_button_clicked, start_button_clicked

#     # Check if the left mouse button is clicked
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # Check if the click is within the "Close" button bounds
#         if x >= button_padding and x <= button_padding + button_width and y >= cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - button_padding - button_height and y <= cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - button_padding:
#             close_button_clicked = True
#         # Check if the click is within the "Start Buying" button bounds
#         elif x >= button_padding + button_width + button_padding and x <= button_padding + button_width + button_padding + button_width and y >= cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - button_padding - button_height and y <= cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - button_padding:
#             start_button_clicked = True

# # Register the mouse click event
# cv2.namedWindow('Camera Feed')
# cv2.setMouseCallback('Camera Feed', on_mouse_click)

# while True:
#     # Read a frame from the camera
#     ret, frame = cap.read()

#     # Check if the frame is successfully read
#     if not ret:
#         print("Failed to read a frame.")
#         break

#     # Convert the frame to grayscale for weight detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Perform weight detection logic here...
#     # ...

#     # Call the detect_objects function
#     detect_objects(frame, '1000')

#     # Check if the "Close" button is clicked
#     if close_button_clicked:
#         break

#     # Wait for a key press and check if it is 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the camera and close all windows
# cap.release()
# cv2.destroyAllWindows()




import cv2
import time
import numpy as np
import os
import serial
from tkinter import *
from pydub import AudioSegment
from pydub.playback import play
import threading


# Define the prices of the products
product_prices = {
    'banana': 3,
    'apple': 2,
    'orange': 4,
    'carrot': 1.5,
    'cell phone': 5,
}

audio_file = "./Gadaxdilia.m4a"  # Replace with the path to your audio file
audio = AudioSegment.from_file(audio_file)

# Create a threading event
audio_event = threading.Event()

# Define a thread function to play the audio
def play_audio_thread():
    while True:
        audio_event.wait()  # Wait for the event to be set
        play(audio)  # Play the audio file
        audio_event.clear()  # Reset the event

# Start the audio thread
audio_thread = threading.Thread(target=play_audio_thread)
audio_thread.start()


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

arduino = serial.Serial('/dev/cu.usbserial-1110', 57600)

# Open the camera
cap = cv2.VideoCapture(0)

# Check if the camera is successfully opened
if not cap.isOpened():
    print("Failed to open the camera.")
    exit()



def detect_objects(frame, weight):

    resized = cv2.resize(frame, (416, 416))
    # Resize the frame for processing

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
                break
    else:
        # No object with price data detected, display 0 Lari
        cv2.putText(frame, "Total Price: 0 Lari", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the weight information on the frame
    cv2.putText(frame, 'Weight: ' + weight + ' gr', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    
    # Display the frame with bounding boxes, labels, and weight information
    cv2.imshow('Camera Feed', frame)


# Register the mouse click event
cv2.namedWindow('Camera Feed')

audio_thread = threading.Thread(target=play_audio_thread)
audio_thread.start()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame is successfully read
    if not ret:
        print("Failed to read a frame.")
        break

    # Convert the frame to grayscale for weight detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    line = arduino.readline().decode('utf-8', errors='ignore').strip()
    print(line)
        # Check if the line contains weight data
    if line.startswith("Load_cell"):
        # Extract the weight value from the line
        weight = line.split(":")[1].strip()
        weight = float(weight)

        # Check if weight is less than 1
        if weight < 1:
            weight = 0
        weight = str(weight)
        detect_objects(frame, '1000')

        # Perform object detection on the frame and display the weight and price if applicable
    elif  line.startswith("Message"):
        cardPermission = line.split(':')[1].strip();
        print(cardPermission)
        audio_event.set()   


    # Wait for a key press and check if it is 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

# Please note that the added functionality will open a new window titled "Product Information" when the "Start Buying" button is clicked. However, the content of this window is currently the same as the main camera feed. You would need to modify the code in the `detect_objects` function to display the specific product information you desire in the "Product Information" window.


