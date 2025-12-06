## Team Member Names
- Rafael Wang (USC ID: 6189106477)
- Robert Fan (USC ID: 7072371273)

## Program Execution
Client and server should be on the same local network.
### Client Side
Encode the images from the Images folder to generate the files in Data folder using: 
```bash
python3 encode_faces.py
```  
Then, run the Yolo Webcome using:
```bash
python3 yolo_detector.py
```

### Server Side
Retrieve the json format message from the client side and disect the info:
```
python server.py
```
The address of dashboard can be viewed in the terminal after starting the server

## External Libraries Used
- ultralytics
- opencv-python
- requests
- numpy
- face_recognition
- flask
- datetime

### LLM Usage
Prompts can be found in prompts.txt
