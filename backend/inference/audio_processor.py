import librosa
import numpy as np

class AudioProcessor:
    def __init__(self, samplerate=22050):
        self.samplerate = samplerate

    def extract_features(self, audio_data, sr=None):
        """Extracts MFCC features from raw audio data."""
        target_sr = sr or self.samplerate
        try:
            # MFCC extraction
            mfccs = librosa.feature.mfcc(y=audio_data, sr=target_sr, n_mfcc=13)
            
            # Ensure we have enough frames for delta calculation (min 9)
            if mfccs.shape[1] < 9:
                 # Pad or just return raw MFCCs
                 features = np.concatenate((mfccs, np.zeros_like(mfccs), np.zeros_like(mfccs)), axis=0)
                 return features.T

            # Delta and Delta-Delta features
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            
            # Combine
            features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs), axis=0)
            return features.T # Shape (Time, Features)
        except Exception as e:
            print(f"Extraction error: {e}")
            return None

    def fuse_features(self, visual_features, audio_features):
        """
        Naive fusion: Concatenates flattened audio features with visual sequence.
        In a real AVSR, we'd use cross-attention, but for this prototype, 
        we'll use a gated fusion approach.
        """
        # Ensure visual is (Time, Features)
        # Ensure audio is (Time, Features) and matching length
        # For simplicity, we'll return visual for now until we retrain
        return visual_features
