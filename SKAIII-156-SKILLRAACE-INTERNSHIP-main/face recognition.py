pip install opencv-python
pip install face_recognition


import cv2
import face_recognition
import os

# Initialize variables
video_capture = cv2.VideoCapture(0)  # Use 0 for the default camera or specify the path to a video file
known_face_encodings = []
known_face_names = []

# Load sample pictures and learn how to recognize them.
known_faces_dir = 'data/known_faces'  # Directory where known face images are stored
for filename in os.listdir(known_faces_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(known_faces_dir, filename)
        img = face_recognition.load_image_file(img_path)
        img_encoding = face_recognition.face_encodings(img)[0]
        known_face_encodings.append(img_encoding)
        known_face_names.append(os.path.splitext(filename)[0])

while True:
    ret, frame = video_capture.read()
    
    if not ret:
        print("Error: Could not read frame from video source.")
        break
    
    # Convert the frame from BGR (OpenCV format) to RGB (face_recognition format)
    rgb_frame = frame[:, :, ::-1]
    
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        # If a match was found in known_face_encodings, use the first one
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
