Voice Recognition and Person Identification Application
This project involves a machine learning model that identifies specific people by extracting MFCC (Mel-Frequency Kepstrum Coefficients) features from audio files and instantaneous speech from a microphone.

Features:
MFCC Feature Extraction: Extracting MFCC features from audio files.
SVM Model: Training an SVM (Support Vector Machine) model using the extracted features.
Identification: Identifying people from the test audio files or from the audio taken instantaneously from the microphone.
Evaluation: Calculate Accuracy and F1 Score.
Visualisation: MFCC histogram and Mel-frequency spectrogram visualisation.
Google Speech API: Use Google Speech API to identify instant microphone data.
Requirements:
Python 3.x
Librosa
Scikit-learn
Matplotlib
SpeechRecognition
Installation:
Run the following command to install the required Python packages:

pip install librosa scikit-learn matplotlib SpeechRecognition

Usage:
Place the audio files to be used as training data appropriately in the variable data_dir and specify the file names in the people list.
Run the script to prepare the training data and train the model.
Use the recognise_from_file function to identify and evaluate the person with the test file.
Use the recognise_from_microphone function to identify instant speech.

# Training data and model training
# ...
# Identifying the person from the test file
actual_person_index = people.index("EnesSesKaydi.wav")
recognise_from_file(test_file_path, actual_person_index)
# Identifying instantaneous speech
recognise_from_microphone()

Licence:
This project is licensed under the MIT Licence.
