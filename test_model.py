import cv2
import pickle
from utils import get_face_landmarks

with open('./model', 'rb') as f:
    model = pickle.load(f)

# Extract the exact numbers the model learned (e.g., [1.0, 2.0, 3.0, 4.0])
trained_classes = model.classes_
emotions_names = ['HAPPY', 'NEUTRAL', 'SAD', 'SURPRISED']

# Create a dictionary to safely map the model's output to the text
# This prevents the IndexError completely
emotion_dict = dict(zip(trained_classes, emotions_names))

# Number 0 because I only have 1 camera in my computer
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # visualize the frame
    face_landmarks = get_face_landmarks(frame, static_image_mode=False)

    # SAFETY CHECK: Only predict if a full face is detected
    if len(face_landmarks) == 1404:
        output = model.predict([face_landmarks])
        predicted_number = output[0]

        # Safely grab the emotion text from the dictionary
        emotion_text = emotion_dict.get(predicted_number, "UNKNOWN")

        cv2.putText(frame,
                    emotion_text,
                    (10, frame.shape[0] - 20),  # Lifted it up slightly so it doesn't get cut off
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 255, 0),
                    5)


    cv2.imshow('frame', frame)

    # wait 25 milliseconds to make the visualization look real-time
    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()