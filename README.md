# Drowsiness Detection System

Drowsiness Detection System built with OpenCV and Machine Learning concepts in order to detect drowsiness in real-time video streams.

## 📸Sample Output Images:
  let's look at some of the sample images where the system detects the drowsiness `img-1 represents eyes open` and `img-2 represents eyes closed`.

<img src="https://github.com/Dinesh-Manoharan/Drowsiness_Detection_System/assets/92298181/506b9bb2-3ee4-4473-afc9-59ce7504b94b" width = 250, height = 250> . 
<img src="https://github.com/Dinesh-Manoharan/Drowsiness_Detection_System/assets/92298181/a9846175-217a-465c-a001-a699829445d2" width = 250, height = 250> 


## Objective:
  The project aims to proactively identify early signs of driver drowsiness to prevent accidents caused by fatigue. Utilizing live camera feeds, behavioral cues are analyzed, focusing on the driver's face and eyes using image processing techniques. The algorithm tracks and analyzes eye movements, triggering an alarm if closed eyes persist in consecutive frames. The system also monitors yawning through lip distance, updating a yawn count. The overall goal is to enhance road safety by providing timely alerts to drivers exhibiting signs of drowsiness.

## Process involved in this project:
  This system utilizes a real-time video feed captured by a camera. Through feature extraction, facial detection employs both a haar cascade method for face and eye detection and a pre-trained facial landmark detector (shape_predictor_68_face_landmarks.dat) for detailed facial features. The software monitors the driver's eyes, triggering an alarm if closed for consecutive frames. Yawning is detected by measuring lip distance, updating a yawn count with a counter.

## TechStack used:
* Python
* OpenCV
* dlib
* numpy
* pygame
