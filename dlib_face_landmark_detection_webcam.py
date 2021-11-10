import cv2 as cv
import numpy as np
import dlib

# Download shape_predictor model
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv.VideoCapture(0)

index = list(range(0, 68)) 

while cap.isOpened():

    ret, frame = cap.read()

    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detector(gray_img, 1)

    for face in faces:

        facial_points = face_predictor(frame, face)

        facial_points_list = []
        for p in facial_points.parts():
            facial_points_list.append([p.x, p.y])

        facial_points_list = np.array(facial_points_list)

        for i,pt in enumerate(facial_points_list[index]):
            pt_pos = (pt[0], pt[1])
            cv.circle(frame, pt_pos, 1, (255, 0, 0), -1)
        
        cv.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()),
            (0, 0, 255), 1)

    cv.imshow('dlib-Face Landmark Detection', frame)
    
    key = cv.waitKey(1)

    if key == 27:
        break

cap.release()