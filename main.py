from flask import Flask, render_template, request, jsonify
from joblib import load
import pandas as pd
import numpy as np
import librosa
import os

def extract_mfcc(path, n_mfcc=10):
    audio, sample_rate = librosa.load(path)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc).T, axis=0)
    return mfcc

    # Ensure that the 'temp' directory exists
if not os.path.exists('temp'):
   os.makedirs('temp')


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file uploaded'})

    file = request.files['audio_file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded file to a temporary location
        temp_path = os.path.join('temp', file.filename)
        file.save(temp_path)

        loaded_model = load('mlp.joblib')

        scaler = load('scaler.joblib')

        # Extract features from the uploaded audio file
        mfcc_features = extract_mfcc(temp_path)

        col = ['mfcc'+str(i) for i in range(10)]

        mfcc_df = pd.DataFrame(mfcc_features.reshape(1, -1), columns = col)

        # Scale the features using your scaler
        scaled_mfcc_features = pd.DataFrame(scaler.transform(mfcc_df), columns=mfcc_df.columns)

        # Use the loaded model for prediction
        prediction = loaded_model.predict(scaled_mfcc_features)

        # Clean up: delete the temporary file
        os.remove(temp_path)

        # Return the prediction result as JSON
        return jsonify({'prediction': prediction.tolist()})

    return jsonify({'error': 'Unknown error'})

if __name__ == '__main__':
    app.run(debug=True)
