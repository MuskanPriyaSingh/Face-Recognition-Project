import cv2
import numpy as np
import os

# Set up constants and configurations
size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
unknown_threshold = 100  # Number of continuous unknown frames before saving as 'Unknown'

# Training the model
print('Training...')

(images, labels, names, id) = ([], [], {}, 0)

# Load all datasets (folders) as individual labeled classes
for subdirs, dirs, _ in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir  # Assign an ID to each subfolder (person)
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            images.append(cv2.imread(path, 0))  # Read the grayscale image
            labels.append(id)  # Append the label ID
        id += 1

(width, height) = (130, 100)  # Resize dimensions

# Convert lists to numpy arrays for training
(images, labels) = [np.array(lis) for lis in [images, labels]]

# Initialize the LBPH recognizer
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

# Face detection with OpenCV Haar Cascade
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

# Counter for unknown faces
unknown_counter = 0

while True:
    _, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))

        # Predict using the model
        label, confidence = model.predict(face_resize)

        if confidence < 100:  # Adjust threshold for better accuracy (lower is stricter)
            name = names[label]  # Recognized person
            cv2.putText(frame, f'{name} - {confidence:.0f}', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
            print(f"Recognized: {name} (Confidence: {confidence:.0f})")
            unknown_counter = 0  # Reset unknown counter
        else:
            unknown_counter += 1
            cv2.putText(frame, 'Unknown', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            print(f"Unknown face detected (Counter: {unknown_counter})")

            if unknown_counter > unknown_threshold:  # Save after consistent unknown detection
                print("Unknown Person Detected - Saving image.")
                cv2.imwrite("Unknown.jpg", frame)
                unknown_counter = 0  # Reset counter after saving

    # Show video feed
    cv2.imshow('Face Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
