import os
import json
import numpy as np
import cv2
import face_recognition

img_path = "Images"
data_path = "data"

import_images = []
names = []

my_list = os.listdir(img_path)
for name in my_list:
    if os.path.splitext(name)[1] != ".jpg":
        continue
    curr = cv2.imread(f'{img_path}/{name}')
    import_images.append(curr)
    names.append(name[:name.rfind('.')])

def compute_enconding(images):
    encode_list = []
    for im in images:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        detection = face_recognition.face_locations(im)
        if detection:
            encode_img = face_recognition.face_encodings(im)[0]
            if len(encode_img)>0:
                encode_list.append(encode_img)
            else:
                print ("Encoding failed.")
        else:
            print("No face detected.")
    
    return encode_list

my_encode_list = compute_enconding(import_images)
print ("Encoding complete")

#save as .npy and json
encode_array = np.vstack(my_encode_list)
np.save(os.path.join(data_path, "encodings.npy"), encode_array)
with open(os.path.join(data_path, "names.json"), "w") as file:
    json.dump(names, file)
