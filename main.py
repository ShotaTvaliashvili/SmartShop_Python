# import cv2
# import time
# import numpy as np
# import os

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


# def take_photo():
#     # Open the camera
#     cap = cv2.VideoCapture(0)

#     # Allow the camera to warm up
#     time.sleep(3)

#     # Capture a frame
#     ret, frame = cap.read()

#     # Release the camera
#     cap.release()

#     # Return the frame
#     return frame


# def detect_objects(image):
#     # Resize the image for processing
#     resized = cv2.resize(image, (416, 416))

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
#                 center_x = int(detection[0] * image.shape[1])
#                 center_y = int(detection[1] * image.shape[0])
#                 width = int(detection[2] * image.shape[1])
#                 height = int(detection[3] * image.shape[0])

#                 # Calculate top-left coordinates of bounding box
#                 x = int(center_x - width / 2)
#                 y = int(center_y - height / 2)

#                 class_ids.append(class_id)
#                 confidences.append(float(confidence))
#                 boxes.append([x, y, width, height])
#                 labels.append(classes[class_id])  # Add the label to the list

#     # Apply non-maximum suppression to remove overlapping boxes
#     indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#     # Draw bounding boxes and labels on the image
#     for i in range(len(boxes)):
#         if i in indices:
#             x, y, width, height = boxes[i]
#             label = labels[i]  # Get the label from the list
#             confidence = confidences[i]

#             # Draw bounding box
#             cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

#             # Draw label with confidence
#             cv2.putText(image, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
#                         2)

#             # Check if the object is a person
#             if label == "person":
#                 # Create a new window for person detection
#                 cv2.namedWindow('Person Detection')
#                 cv2.moveWindow('Person Detection', 700, 100)

#                 # Display the object and price in the new window
#                 cv2.putText(image, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
#                             2)
#                 cv2.putText(image, "Price: 100 Lari", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                 cv2.imshow('Person Detection', image)

#     # Print the detected labels in the console
#     print("Detected objects:", ", ".join(labels))


# def main():
#     # Create a window
#     cv2.namedWindow('Camera Window')

#     # Flag to track if the start button was clicked
#     start_clicked = False

#     def button_callback(event, x, y, flags, param):
#         nonlocal start_clicked

#         if event == cv2.EVENT_LBUTTONDOWN:
#             # Set the start button click flag
#             start_clicked = True

#     # Create a blank frame for the start button
#     start_frame = 255 * np.ones((100, 200, 3), np.uint8)
#     cv2.putText(start_frame, 'Press "S" to capture photo', (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

#     # Display the start button frame in the main window
#     cv2.imshow('Camera Window', start_frame)
#     cv2.setMouseCallback('Camera Window', button_callback)

#     while True:
#         # Check for key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         # Check if the start button was clicked
#         if start_clicked:
#             # Release the start button frame
#             cv2.destroyAllWindows()

#             # Display the camera feed in the main window
#             cap = cv2.VideoCapture(0)
#             while True:
#                 ret, frame = cap.read()
#                 cv2.imshow('Camera Window', frame)

#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#                 # Capture a photo after 6 seconds
#                 if cv2.waitKey(1) & 0xFF == ord('s'):
#                     # Take a photo
#                     photo = take_photo()

#                     # Close the camera feed window
#                     cv2.destroyAllWindows()

#                     # Detect objects in the photo
#                     detect_objects(photo)

#                     # Display the captured photo with object detection
#                     cv2.imshow('Captured Photo', photo)
#                     cv2.waitKey(0)

#                     # Save the photo
#                     cv2.imwrite('photo.jpg', photo)

#                     # Break out of the loop
#                     break

#             # Release the camera
#             cap.release()

#             # Break out of the outer loop
#             break

#     # Close all windows
#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     main()



import cv2
import time
import numpy as np
import os

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
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

def take_photo():
    # Open the camera
    cap = cv2.VideoCapture(0)
    
    # Allow the camera to warm up
    time.sleep(3)
    
    # Capture a frame
    ret, frame = cap.read()
    
    # Release the camera
    cap.release()
    
    # Return the frame
    return frame

def detect_objects(image):
    # Resize the image for processing
    resized = cv2.resize(image, (416, 416))

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
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                width = int(detection[2] * image.shape[1])
                height = int(detection[3] * image.shape[0])

                # Calculate top-left coordinates of bounding box
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, width, height])
                labels.append(classes[class_id])  # Add the label to the list

    # Apply non-maximum suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the image
    for i in range(len(boxes)):
        if i in indices:
            x, y, width, height = boxes[i]
            label = labels[i]  # Get the label from the list
            confidence = confidences[i]

            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Draw label with confidence
            cv2.putText(image, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)

    # Print the detected labels in the console
    print("Detected objects:", ", ".join(labels))


def main():
    # Create a window
    cv2.namedWindow('Camera Window')
    
    # Flag to track if the start button was clicked
    start_clicked = False
    
    def button_callback(event, x, y, flags, param):
        nonlocal start_clicked
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Set the start button click flag
            start_clicked = True
    
    # Create a blank frame for the start button
    start_frame = 255 * np.ones((100, 200, 3), np.uint8)
    cv2.putText(start_frame, 'Press "S" to capture photo', (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Display the start button frame in the main window
    cv2.imshow('Camera Window', start_frame)
    cv2.setMouseCallback('Camera Window', button_callback)
    
    while True:
        # Check for key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Check if the start button was clicked
        if start_clicked:
            # Release the start button frame
            cv2.destroyAllWindows()
            
            # Get the camera index
            usb_camera_index = -1
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    usb_camera_index = i
                    cap.release()
                    break
            
            if usb_camera_index == -1:
                print("USB camera not found.")
                break
            
            # Display the camera feed in the main window
            cap = cv2.VideoCapture(usb_camera_index)
            while True:
                ret, frame = cap.read()
                cv2.imshow('Camera Window', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Capture a photo after 6 seconds
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    # Take a photo
                    photo = take_photo()
                    
                    # Close the camera feed window
                    cv2.destroyAllWindows()
                    
                    # Detect objects in the photo
                    detect_objects(photo)
                    
                    # Display the captured photo with object detection
                    cv2.imshow('Captured Photo', photo)
                    cv2.waitKey(0)
                    
                    # Save the photo
                    cv2.imwrite('photo.jpg', photo)
                    
                    # Break out of the loop
                    break
                    
            # Release the camera
            cap.release()
            
            # Break out of the outer loop
            break
    
    # Close all windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

