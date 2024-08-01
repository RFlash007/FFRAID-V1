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
#from llama import chat_loop


# Set up logging to suppress YOLOv10 output
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Load the model
model = YOLO("yolov10n.pt")
app = typer.Typer()
#create tracker
tracker = cv2.TrackerKCF_create()

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
def process_webcam():
    global frame_switch
    cap = cv2.VideoCapture(0)  # 0 is typically the default webcam
    # Set webcam resolution to 1080p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    confidence_threshold = 0.5  # Set your desired confidence threshold here

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        for box, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
            if confidence >= confidence_threshold:
                if class_id in category_dict:
                    class_name = category_dict[class_id]
                else:
                    class_name = "Unknown"  # Assign the custom label for unclassified objects
                #save frame
                if frame_switch is True:
                    cv2.imwrite("screen.jpg", frame)
                    frame_switch = False

                x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                center_x = x + w // 2
                center_y = y + h // 2
                radius = 150
                circle_color = (143, 48, 0, 0)  # Light blue color (RGB values)
                cv2.circle(frame, (x+200, y+160), radius, circle_color, 5)  # Draw the circle
                cv2.putText(frame, f"{class_name}: {confidence:.2f}", (x + 20, y),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.7, circle_color, 2)



        cv2.imshow("Webcam", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
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


if __name__ == "__main__":
    # Create threads for the object detection loop and chat loop
    object_detection_thread = threading.Thread(target=process_webcam)
    chat_thread = threading.Thread(target=chat_loop)

    # Start both threads
    object_detection_thread.start()
    chat_thread.start()


    # Wait for both threads to finish
    object_detection_thread.join()
    chat_thread.join()