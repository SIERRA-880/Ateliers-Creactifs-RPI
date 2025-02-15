import cv2
import sys
import logging as log
import datetime as dt
import face_recognition
import numpy as np
from time import sleep

# Charger les images de référence et apprendre leurs encodages
known_face_encodings = []
known_face_names = []

# Ajouter ici les images de référence (exemple : "person1.jpg")
reference_images = {"Barack_Obama": "img_1.jpg", "Jean_Bourgies": "img_2.jpg"}
for name, img_path in reference_images.items():
    image = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(image)
    if encoding:
        known_face_encodings.append(encoding[0])
        known_face_names.append(name)

log.basicConfig(filename='webcam.log', level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        continue

    ret, frame = video_capture.read()
    if not ret:
        continue
    
    # Convertir l'image en RGB pour face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Trouver les visages et leurs encodages
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    # Identifier les visages
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        # Trouver la meilleure correspondance
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None
        if best_match_index is not None and matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        face_names.append(name)
    
    # Dessiner les rectangles et les noms
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
    
    if anterior != len(face_locations):
        anterior = len(face_locations)
        log.info(f"Faces detected: {len(face_locations)} at {dt.datetime.now()}")
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
