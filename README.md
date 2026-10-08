# Real-Time Emotion Classifier with Custom UI

A machine learning pipeline that captures real-time webcam feeds, extracts facial landmarks, and classifies human emotions using a Random Forest model. The project features a custom-engineered UI that dynamically displays original vector artwork corresponding to the detected emotion.

<img width="2550" height="5095" alt="emotion class vector" src="https://github.com/user-attachments/assets/5ccdbb92-158e-4ff7-8c10-a55efaf9b4d3" />

### Custom Vector Art Assets
*Designed using Procreate and Adobe Illustrator.*

| Happy | Neutral | Sad | Surprised |
| :---: | :---: | :---: | :---: |
| ![Happy](happy_vector.png) | ![Neutral](neutral_vector.png) | ![Sad](sad_vector.png) | ![Surprised](surprised_vector.png) |

---

## 🚀 Features & Technical Highlights

* **Real-Time Classification:** Processes live video feeds using OpenCV and MediaPipe to extract 1,404 precise facial landmarks per frame.
* **Custom UI & Asset Mapping:** Integrates custom, Duolingo-inspired vector art. The UI dynamically maps the model's prediction to the corresponding artwork in a secondary OpenCV window.
* **Crash-Proof Logic:** Implemented dictionary mapping for model predictions to strictly prevent `IndexError` crashes, alongside a landmark validation check (requiring exactly 1404 points) before pushing frames to the predictive model.
* **Data Normalization:** Raw coordinate data is normalized by subtracting minimum values, ensuring the model focuses on facial geometry rather than absolute position in the frame.

## 🛠️ Tech Stack

* **Language:** Python 3
* **Computer Vision:** OpenCV-python (ver 4.9.0.80), MediaPipe (ver 0.10.9)
* **Machine Learning:** Scikit-learn (ver 1.4.0, Random Forest Classifier), NumPy (ver 1.26.3)
* **Design Tools:** Procreate, Adobe Illustrator

## 📊 Dataset

The model was trained on a custom-selected Kaggle dataset: https://www.kaggle.com/datasets/alyyan/emotion-detection. This differs from standard tutorials to ensure a more robust variety of facial structures. 

## 📁 Project Structure

* `prepare_data.py`: Iterates through the raw image dataset, extracts facial landmarks, normalizes the data, and securely saves it to a text file while actively managing memory.
* `train_model.py`: Loads the extracted coordinate data, splits it into training and testing sets, trains the Random Forest Classifier, and exports the serialized model using `pickle`.
* `utils.py`: Contains the core facial landmark extraction logic and coordinate normalization mathematics.
* `test_model.py`: The live execution script. Captures the webcam feed, runs the live prediction, and triggers the synchronized custom UI.

## ⚙️ How to Run

1. Clone this repository.
2. Install the required dependencies:
   - OpenCV-python (ver 4.9.0.80)
   - MediaPipe (ver 0.10.9)
   - Scikit-learn (ver 1.4.0)
   - NumPy (ver 1.26.3)
  
     
