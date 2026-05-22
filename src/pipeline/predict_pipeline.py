import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object, read_yaml_config

class PredictPipeline:
    def __init__(self, model_name="BiLSTM"):
        self.config = read_yaml_config("config.yaml")
        self.model_name = model_name
        
        # Determine which model to load based on user selection
        if model_name == "BiLSTM":
            self.model_path = self.config.get("model_training", {}).get("bilstm_model_path", "artifacts/bilstm_model.h5")
        elif model_name == "LSTM":
            self.model_path = self.config.get("model_training", {}).get("lstm_model_path", "artifacts/lstm_model.h5")
        elif model_name == "CNN":
            self.model_path = self.config.get("model_training", {}).get("cnn_model_path", "artifacts/cnn_model.h5")
        else:
            self.model_path = "artifacts/bilstm_model.h5"

        self.tokenizer_path = self.config.get("data_transformation", {}).get("tokenizer_path", "artifacts/tokenizer.pkl")
        self.label_encoder_path = self.config.get("data_transformation", {}).get("label_encoder_path", "artifacts/label_encoder.pkl")
        self.max_len = self.config.get("data_transformation", {}).get("max_len", 150)

    def predict(self, text: str):
        try:
            logging.info(f"Loading tokenizer, label encoder, and {self.model_name} model for prediction.")
            tokenizer = load_object(self.tokenizer_path)
            label_encoder = load_object(self.label_encoder_path)
            model = load_model(self.model_path)

            # Preprocess the input text
            seq = tokenizer.texts_to_sequences([text])
            padded_seq = pad_sequences(seq, maxlen=self.max_len, padding="post", truncating="post")

            # Predict
            predictions = model.predict(padded_seq)[0]
            predicted_class_index = np.argmax(predictions)
            
            # Map index back to string label
            predicted_category = label_encoder.inverse_transform([predicted_class_index])[0]
            confidence_score = predictions[predicted_class_index]
            
            # Create a dictionary of all classes and their probabilities
            all_classes = label_encoder.classes_
            confidence_dict = {all_classes[i]: float(predictions[i]) for i in range(len(all_classes))}

            logging.info(f"Prediction successful: {predicted_category} ({confidence_score:.4f})")
            
            return predicted_category, confidence_score, confidence_dict, predicted_class_index

        except Exception as e:
            raise CustomException(e, sys)