import os
import mediapipe as mp
import cv2
import sklearn
import numpy as np
from utils import get_face_landmarks

data_dir = './data'

"""
I could use less data and optimize its usage by focusing on the
landmarks that truly matter, which are the mouth and eye areas, but I don't
think it matters in the case of this project, so I am going to leave it like that.
Working with all the data helps deal with more scenarios like when there's no
face at all, or fewer landmarks are being worked with.
"""

# Open the file at the start in 'append' mode ('a')
with open('data.txt', 'a') as f:
    # sorted(...) is used so that the emotions are sorted alphabetically
    for emotion_indx, emotion in enumerate(sorted(os.listdir(data_dir))):
        if emotion.startswith('.'):
            continue

        emotion_folder_path = os.path.join(data_dir, emotion)

        for image_name in os.listdir(emotion_folder_path):
            if image_name.startswith('.'):
                continue

            image_path = os.path.join(emotion_folder_path, image_name)
            image = cv2.imread(image_path)

            if image is None:
                continue

            face_landmarks = get_face_landmarks(image)

            if len(face_landmarks) == 1404:
                face_landmarks.append(int(emotion_indx))
                # Save this single face directly to the file
                np.savetxt(f, np.asarray([face_landmarks]))

            # Manually delete the image from memory to prevent PyCharm lag
            del image
        
