import cv2
import dlib
import numpy as np
from pygame import mixer 

mixer.init()
sound = mixer.Sound('mixkit-alarm-tone-996.wav')


face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
left_eye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
right_eye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()


def get_landmarks(img):
    rects = detector(img, 1)
    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])


def annotate_landmarks(img, landmarks):
    img = img.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(img, str(idx), pos, fontFace= cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4, color=(0, 0, 255))
        cv2.circle(img, pos, 3, color=(0, 255, 255))
    return img


def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append( landmarks[i])
    for i in range(61,64):
        top_lip_pts.append( landmarks[i])
    top_lip_all_pts = np.squeeze( np.asarray( top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])


def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append( landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append( landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray( bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])


def mouth_open(image):
    landmarks = get_landmarks(image)
    if landmarks == "error":
        return image, 0
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance

def detect_red(image):
    for (x, y, w, h) in image:
       lower_red = np.array([161, 155, 84], dtype = "uint8") 
       upper_red= np.array([179, 255, 255], dtype = "uint8")
       red_mask = cv2.inRange(hsvimg, lower_red, upper_red)
       detected_output = cv2.bitwise_and(hsvimg, hsvimg, mask = red_mask)
       
       if detected_output.any():
            return True
       else:
            return False 


yawns = 0
yawn_status = False

while True:
    _,img = cap.read()
    height,width = img.shape[:2] 

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    faces = face.detectMultiScale(gray)
    

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        gray_tmp = img[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1, :]
        gray = gray[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1]
        leye = left_eye.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
        reye = right_eye.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))


        if len(leye) == 0:
            cv2.putText(img,"left eye Closed",(10,height-40), font, 1,(255,255,255),1,cv2.LINE_AA)
        else:
            cv2.putText(img,"left eye Open",(10,height-40), font, 1,(255,255,255),1,cv2.LINE_AA)
            
            
            if detect_red(leye):
                cv2.putText(img,"eyes red",(10,80), font, 1,(255,255,255),1,cv2.LINE_AA)
            else:
                cv2.putText(img,"eyes white",(10,80), font, 1,(255,255,255),1,cv2.LINE_AA)
        

        if len(reye) == 0:
            cv2.putText(img,"right eye Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        else:
            cv2.putText(img,"right eye Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

            if detect_red(reye):
                cv2.putText(img,"eyes red",(10,80), font, 1,(255,255,255),1,cv2.LINE_AA)
            else:
                cv2.putText(img,"eyes white",(10,80), font, 1,(255,255,255),1,cv2.LINE_AA)

        if len(leye)==0 and len(reye) == 0:
            sound.play()

        image_landmarks, lip_distance = mouth_open(img)
    
        prev_yawn_status = yawn_status
   
        if lip_distance > 30:

            yawn_status = True
            cv2.putText(img, "person is Yawning", (10,50), font, 1,(255,0,0),1,cv2.LINE_AA)
            output_text = "Yawn Count: " + str(yawns + 1)
            cv2.putText(img, output_text, (450,50), font, 1,(255,0,0),1,cv2.LINE_AA)
        else:
                yawn_status = False
            
        if prev_yawn_status == True and yawn_status == False:
            yawns += 1
    
    cv2.imshow('output' , img) 
    
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()