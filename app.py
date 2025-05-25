import streamlit as st
import librosa
import numpy as np
import joblib
import soundfile as sf

# Function to extract MFCC, RMS, Spectral Centroid, Spectral Bandwidth, Spectral Rolloff, and Zero Crossing Rate features from audio
def extract_features(audio_path):
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=22050)
        # Extract MFCC features (21 coefficients to match model requirements)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=21)
        # Compute mean across time for each MFCC coefficient
        mfcc_mean = np.mean(mfcc, axis=1)
        # Extract RMS feature
        rms = librosa.feature.rms(y=y)
        # Compute mean of RMS
        rms_mean = np.mean(rms)
        # Extract Spectral Centroid feature
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        # Compute mean of Spectral Centroid
        spectral_centroid_mean = np.mean(spectral_centroid)
        # Extract Spectral Bandwidth feature
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        # Compute mean of Spectral Bandwidth
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        # Extract Spectral Rolloff feature
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        # Compute mean of Spectral Rolloff
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        # Extract Zero Crossing Rate feature
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
        # Compute mean of Zero Crossing Rate
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)
        # Concatenate MFCC, RMS, Spectral Centroid, Spectral Bandwidth, Spectral Rolloff, and Zero Crossing Rate features
        features = np.concatenate([mfcc_mean, [rms_mean], [spectral_centroid_mean], [spectral_bandwidth_mean], [spectral_rolloff_mean], [zero_crossing_rate_mean]])
        return features
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# Streamlit app
st.title("Real-Time AI-Generated Voice Detection")
st.write("Upload an audio file to check if the voice is Real or Fake.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display audio player
    st.audio(uploaded_file)
    
    # Extract features
    features = extract_features("temp_audio.wav")
    
    if features is not None:
        try:
            # Load the pre-trained model
            model = joblib.load("catboost_model.pkl")
            
            # Reshape features for prediction
            features = features.reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Map prediction to label (assuming 0 = Real, 1 = Fake)
            label = "Fake" if prediction == 1 else "Real"
            
            # Display result
            st.subheader("Result")
            st.write(f"The voice is predicted to be: **{label}**")
            
        except Exception as e:
            st.error(f"Error loading model or making prediction: {e}")
else:
    st.info("Please upload an audio file to proceed.")