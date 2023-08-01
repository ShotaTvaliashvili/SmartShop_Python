import cv2
import numpy as np
import os
import time
import serial
from tkinter import *
from pydub import AudioSegment
from pydub.playback import play
import threading

product_prices = {
        'banana': 3,
        'apple': 2,
        'orange': 4,
        'carrot': 1.5,
        'cell phone': 5,
}

EnableToPay = 0

ProductLabel = ''
ProductLabelWeight = ''
ProductLabelPrice = ''

paidProduct = ''
paidProductWeight = '0'
paidProductPrice = '0'

class ObjectDetector:
    def __init__(self, weights_path, config_path, names_path):
        self.weights_path = weights_path
        self.config_path = config_path
        self.names_path = names_path
        self.net = None
        self.classes = []
        self.layer_names = None
        self.output_layers = None

        self.load_yolo()
        self.load_class_names()

    def load_yolo(self):
        self.net = cv2.dnn.readNet(self.weights_path, self.config_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def load_class_names(self):
        with open(self.names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def detect_objects(self, frame, weight):
        global paidProduct, paidProductPrice, paidProductWeight
        resized = cv2.resize(frame, (416, 416))
        blob = cv2.dnn.blobFromImage(resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []
        labels = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])

                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, width, height])
                    labels.append(self.classes[class_id])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            global EnableToPay, ProductLabel, ProductLabelPrice, ProductLabelWeight
            if i in indices:
                x, y, width, height = boxes[i]
                label = labels[i]
                confidence = confidences[i]

                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if label.lower() in product_prices:
                    price = product_prices[label.lower()]
                    weight_in_kg = float(weight) / 1000
                    total_price = round(price * weight_in_kg, 2)
                    EnableToPay = total_price

                    ProductLabel = label
                    ProductLabelPrice = total_price
                    ProductLabelWeight = weight

                    cv2.rectangle(frame, (0, 0), (360, 200), (100, 100, 100,), cv2.FILLED)
                    cv2.putText(frame, f"Product: {label}", (20, 30 ), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(frame, f"1KG Price: {price} GEL", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(frame, f"Weight: {weight} gr", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(frame, f"Total Price: {total_price:.2f} GEL", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(frame, f"Start payment", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 3)

                    break
        else:
            cv2.rectangle(frame, (0, 0), (400, 100), (100, 100, 100), cv2.FILLED)
            cv2.putText(frame, f"Weight: {weight} gr", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, "Total Price: 0 GEL", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            EnableToPay = 0
        if paidProductWeight != "0" and paidProductPrice != "0":
            cv2.putText(frame, f"Product: {paidProduct}", (370, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Weight: {paidProductWeight} gr", (370, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Total Price: {paidProductPrice} GEL", (370, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Camera Feed', frame)



def play_audio(audio):
    play(audio)


class WeightScanner:

    def __init__(self, arduino_port, audio_event):
        self.arduino = serial.Serial(arduino_port, 115200)
        self.audio_event = audio_event

    def read_weight(self):
        global EnableToPay
        line = self.arduino.readline().decode('utf-8', errors='ignore').strip()
        print(line)
        if line.startswith("Load_cell"):
            weight = line.split(":")[1].strip()
            weight = float(weight)

            if weight < 1:
                weight = 0
            weight = str(weight)
            return weight
        elif line.startswith("Message"):
            message = line.split(":")[1].strip()
            print("Received message:", message)
            self.play_audio_message(message)
        return None
    def play_audio_message(self, message):
        audio_mapping = {
            "Access denied": "./FailSound.m4a",
            "Authorized access": "./SuccessSound.m4a",
            # Add more audio mappings here for different messages
        }
        if EnableToPay == 0:
            return
        print(message)
        if message == "Authorized access":
            global ProductLabel, ProductLabelPrice, ProductLabelWeight, paidProduct, paidProductPrice, paidProductWeight

            paidProduct = ProductLabel
            paidProductPrice = ProductLabelPrice
            paidProductWeight = ProductLabelWeight

        if message in audio_mapping:
            audio_file = audio_mapping[message]
            audio = AudioSegment.from_file(audio_file)
            play(audio)
            self.audio_event.set()
            self.audio_event.clear()


def main():

    weights_path = "yolov3.weights"
    config_path = "yolov3.cfg"
    names_path = "coco.names"
    arduino_port = '/dev/cu.usbserial-1110'
    audio_file = "./Gadaxdilia.m4a"


    audio_event = threading.Event()

    # def play_audio_thread():
    #     while True:
    #         audio_event.wait()
    #         play_audio(audio)
    #         audio_event.clear()

    # audio_thread = threading.Thread(target=play_audio_thread)
    # audio_thread.start()

    detector = ObjectDetector(weights_path, config_path, names_path)
    scanner = WeightScanner(arduino_port, audio_event)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open the camera.")
        exit()

    cv2.namedWindow('Camera Feed')

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read a frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        weight = scanner.read_weight()
        print(weight)
        if weight is not None:
            detector.detect_objects(frame, weight)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

