from ultralytics import YOLO
import cv2
import datetime
import requests
import time
import base64
from recognition import classify_person
import os

DANGER_STATE = { #for a danger alarm system
    "active": False,
    "expiration_time": 0
}

ROBERT_SERVER = "http://172.20.10.2:5001"

model = YOLO("yolov8n.pt")

def run_yolo(frame):
    results = model(frame, verbose=False)
    result = results[0] #for just one frame
    detections = []
    boxes = result.boxes
    for box in boxes:
        x_c, y_c, w, h = box.xywh[0].tolist()      
        conf = float(box.conf[0])      
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id] 
        detection = {
                "class_name": cls_name,
                "class_id": cls_id,
                "confidence": conf,
                "bbox": {
                    "x_center": x_c,
                    "y_center": y_c,
                    "width": w,
                    "height": h
                }
            }
        detections.append(detection)
            
    return detections

if __name__ == "__main__":
    cap = cv2.VideoCapture(1) #starting camera
    json_data = []
    if not cap.isOpened():
        print("No camera found.")
    else:
        print("Press ENTER to quit.")
        frame_id=0
        last_sent_time = 0
        start_log = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            current_time = datetime.datetime.now().isoformat()
            dets = run_yolo(frame)
            person_dets = [d for d in dets if d["class_name"] == "person"] 
            if person_dets:
                if start_log==False:
                    start_log = True
                person_bbox = person_dets[0]["bbox"] #get just first person
                person_info = classify_person(frame, person_bbox)
            else:
                person_info = {"type": "unknown", "name": None, "distance": None}

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8') #GitHub CoPilot (Gemini 3 Pro model) was used to generate this

            frame_data = {
                "timestamp": current_time,
                "frame_id": frame_id,
                "detections": dets,
                "person_info": person_info,
                "image": jpg_as_text
            }

            if time.time() - last_sent_time > 1.5:
                try:
                    response = requests.post(f"{ROBERT_SERVER}/frame_result", json=frame_data, timeout=(0.05, 0.2)) #sending frame_data over to RPi
                    data = response.json()
                    if data.get("threat_flag", False):#turning on alarm system
                        DANGER_STATE["active"] = True
                        DANGER_STATE["expiration_time"] = time.time() + 30
                    last_sent_time = time.time()
                except Exception as e:
                    print(f"Error sending: {e}")
            if start_log:
                json_data.append(frame_data)

            for d in dets:
                bbox = d["bbox"]
                x_c, y_c, w, h = bbox["x_center"], bbox["y_center"], bbox["width"], bbox["height"]
                x1 = int(x_c - w / 2)
                y1 = int(y_c - h / 2)
                x2 = int(x_c + w / 2)
                y2 = int(y_c + h / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{d['class_name']} {d['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
            if DANGER_STATE["active"]:
                if time.time() > DANGER_STATE["expiration_time"]: #turn off after 30s
                    DANGER_STATE["active"] = False
                    print("Threat mode expired.")
                elif person_dets:
                    cv2.putText(frame, "THREAT DETECTED!", (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 4)
                    os.system("afplay alarm.mp3 &") 
                    print("ALARM!!!")

            cv2.imshow("YOLO Detector", frame)    
            if (cv2.waitKey(1) & 0xFF == 13):
                break
        cap.release()
        cv2.destroyAllWindows()
