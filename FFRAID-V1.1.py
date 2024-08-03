from __future__ import print_function

import multiprocessing
import queue
import threading
import time
import cv2
import os
import supervision as sv
from ultralytics import YOLO
import typer
import sys
import logging
import multiprocessing
import ollama
import sys
from random import randint


# Set up logging to suppress YOLOv10 output
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Load the model
model = YOLO("yolov10n.pt")
app = typer.Typer()

# Initiate trackers list
trackers = []

category_dict = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}
frame_switch = False
terminate_flag = False
def process_webcam():
    global frame_switch, trackers, terminate_flag
    cap = cv2.VideoCapture(0)  # 0 is typically the default webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initiate confidence threshold
    confidence_threshold = 0.5

    while True:
        # Start Webcam
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        for box, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
            if frame_switch is True:
                cv2.imwrite("screen.jpg", frame)
                frame_switch = False

            # If high confidence object add to object detection
            if confidence >= confidence_threshold:
                if class_id in category_dict:
                    class_name = category_dict[class_id]
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (143, 48, 0, 0), 3)
                    cv2.putText(frame, f"{class_name}: {confidence:.2f}", (int(box[0]), int(box[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (143, 48, 0, 0), 2)

            # If not high confidence object opt for tracker
            else:
                if len(trackers) < 4 and 0.2 <= confidence < 0.5:

                    # Check if the detected object is close to existing trackers
                    close_to_existing = False
                    for tracker in trackers:
                        success, tracked_box = tracker.update(frame)
                        if success:
                            x, y, w, h = map(int, tracked_box)
                            existing_area = w * h
                            new_area = (box[2] - box[0]) * (box[3] - box[1])
                            overlap_threshold = 0.7  # Adjust as needed
                            overlap_area = max(0, min(box[2], x + w) - max(box[0], x)) * max(0, min(box[3], y + h) - max(
                                box[1], y))
                            overlap_ratio = overlap_area / (existing_area + new_area - overlap_area)
                            if overlap_ratio > overlap_threshold:
                                close_to_existing = True
                                break

                    # If a new item initiate tracker
                    if close_to_existing is False:
                        x1, y1, x2, y2 = map(int, box)
                        width = x2 - x1
                        height = y2 - y1

                        # Initialize tracker
                        tracker = cv2.TrackerKCF_create()
                        tracker.init(frame, (x1, y1, width, height))
                        trackers.insert(0, tracker)

        # Update trackers only for detected objects
        for tracker in trackers:
            success, tracked_box = tracker.update(frame)
            if success:
                x, y, w, h = map(int, tracked_box)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Unknown", (x + 20, y),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Multi-Object Tracking", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            terminate_flag = True
            break

    cap.release()
    cv2.destroyAllWindows()

def chat_loop():
    conversation = []  # Initialize an empty conversation list
    conversation.append({"role": "user", "content": "Hi, Im hooking you up to a visual Aid that will Analyze the room for you. When I say purple potato you will receive a pre-processed description from what another language model sees so you dont have to do all the thinking tell me what you think about what you see and give me insight if you see fit. So, purple potato is like a magic word that tells the other language model to do its thing and describe what it sees. Im a guy with blonde hair in case im in the pictures because they're taken from a live feed of the room right now."})
    conversation.append({"role": "assistant", "content": "ok I will, i will explain to you what the other model gives me and offer my technical assistance with anything you need help with, I will also let you know if i see you because you're a guy with blonde hair"})
    while True:
        user_input = input("You: ")  # Get user input

        # Check if user input is "q" (case-insensitive)
        if user_input.lower() == "q":
            print("Goodbye! FFRAID signing off.")
            break  # Exit the loop

        if "purple potato" in user_input.lower():
            # Execute your custom function here
            print("Detected 'purple potato'! Executing custom function...")
            user_input += data_model()

        #append user message to conversation
        conversation.append({"role": "user", "content": user_input})
        # Chat with the model
        response = ollama.chat(model="FFRAID", messages=conversation)
        print(f"Model: {response['message']['content']}")  # Print the model's response

        # Add the model's response to the conversation
        conversation.append({"role": "assistant", "content": response["message"]["content"]})


def data_model():
    global frame_switch
    frame_switch = True
    time.sleep(3)
    res = ollama.chat(
        model="ImageProcessing",
        messages=[
            {
                'role': 'user',
                'content': 'Describe this image:',
                'images': ['./screen.jpg']
            }
        ]
    )
    print(res["message"]["content"])
    return res["message"]["content"]

def tracker_update():
    global trackers
    # Periodically pop trackers to remove old trackers
    while terminate_flag is not True:
        time.sleep(10)
        if len(trackers) > 0:
            trackers.pop()



if __name__ == "__main__":
    # Create threads for the object detection loop, chat loop, and list updating
    object_detection_thread = threading.Thread(target=process_webcam)
    chat_thread = threading.Thread(target=chat_loop)
    tracker_thread = threading.Thread(target=tracker_update)

    # Start threads
    object_detection_thread.start()
    chat_thread.start()
    tracker_thread.start()


    # Wait for threads to finish
    object_detection_thread.join()
    chat_thread.join()
    tracker_thread.join()

