import cv2
import numpy as np
import face_recognition
import os
import json

path = 'data'
img_path = 'Images'
cache = {}
thumbnail_width = 400

encodings_path = os.path.join(path, "encodings.npy")
names_path = os.path.join(path, "names.json")

if not os.path.exists(encodings_path) or not os.path.exists(names_path):
    print ("Run encode_faces.py first.")
    exit()

my_encode_list = np.load(encodings_path)
with open (names_path, "r") as file:
    names = json.load(file)

list_thresh = 0.5


def crop_yolo_bbox(img, bbox): #converting yolo box to opencv box
    h, w, _ = img.shape

    x_c = bbox["x_center"]
    y_c = bbox["y_center"]
    bw = bbox["width"]
    bh = bbox["height"]

    x1 = int(x_c - bw / 2)
    y1 = int(y_c - bh / 2)
    x2 = int(x_c + bw / 2)
    y2 = int(y_c + bh / 2)

    # dealing with edge case
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2]

def classify_person(img, human_bbox): #determine if person is known or unknown
    cropped = crop_yolo_bbox(img, human_bbox)
    if cropped is None:
        return {"type": "unknown", "name": None}
    img_shrink = cv2.resize(cropped, (0, 0), None, 0.20, 0.20)
    img_rgb = cv2.cvtColor(img_shrink, cv2.COLOR_BGR2RGB)
    faces_loc = face_recognition.face_locations(img_rgb)
    if not faces_loc:
        return {"type": "unknown", "name": None}
    encode_imgs = face_recognition.face_encodings(img_rgb, faces_loc, num_jitters=2, model="small")
    if not encode_imgs:
        return {"type": "unknown", "name": None}

    encoded_face = encode_imgs[0]
    list_dist = face_recognition.face_distance(my_encode_list, encoded_face)
    if len(list_dist) == 0:
        return {"type": "unknown", "name": None}
    best_index = int(np.argmin(list_dist))
    best_dist = float(list_dist[best_index])
    if best_dist < list_thresh:
        name = names[best_index]
        return {
            "type": "friend", "name": name}
    else:
        return {
            "type": "unknown", "name": None}