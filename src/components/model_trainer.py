import os
import sys
import numpy as np
from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Conv1D, GlobalMaxPooling1D, Dense, Dropout

from src.logger import logging
from src.exception import CustomException
from src.utils import read_yaml_config

@dataclass
class ModelTrainerConfig:
    bilstm_model_path: str = os.path.join("artifacts", "bilstm_model.h5")
    lstm_model_path: str = os.path.join("artifacts", "lstm_model.h5")
    cnn_model_path: str = os.path.join("artifacts", "cnn_model.h5")

class ModelTrainer:
    def __init__(self, vocab_size, embedding_matrix, num_classes):
        self.trainer_config = ModelTrainerConfig()
        self.config = read_yaml_config("config.yaml")
        
        self.vocab_size = vocab_size
        self.embedding_matrix = embedding_matrix
        self.num_classes = num_classes
        self.max_len = self.config.get("data_transformation", {}).get("max_len", 150)
        self.embedding_dim = self.config.get("data_transformation", {}).get("embedding_dim", 100)

    def _get_embedding_layer(self):
        """Returns a pre-configured Keras Embedding layer using GloVe."""
        return Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            weights=[self.embedding_matrix],
            input_length=self.max_len,
            trainable=False # Keep GloVe weights frozen initially
        )

    def build_bilstm(self):
        model = Sequential([
            self._get_embedding_layer(),
            Bidirectional(LSTM(64, return_sequences=True)),
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def build_lstm(self):
        model = Sequential([
            self._get_embedding_layer(),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def build_cnn(self):
        model = Sequential([
            self._get_embedding_layer(),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            epochs = self.config.get("model_training", {}).get("epochs", 5)
            batch_size = self.config.get("model_training", {}).get("batch_size", 64)

            models = {
                "BiLSTM": (self.build_bilstm(), self.trainer_config.bilstm_model_path),
                "LSTM": (self.build_lstm(), self.trainer_config.lstm_model_path),
                "CNN": (self.build_cnn(), self.trainer_config.cnn_model_path)
            }

            training_history = {}

            for model_name, (model, save_path) in models.items():
                logging.info(f"--- Starting Training for {model_name} ---")
                
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test, y_test),
                    verbose=1
                )
                
                # Save the trained model
                model.save(save_path)
                logging.info(f"Saved {model_name} at {save_path}")
                
                # Store final validation accuracy for logging
                val_acc = history.history['val_accuracy'][-1]
                training_history[model_name] = val_acc
                logging.info(f"{model_name} Final Validation Accuracy: {val_acc:.4f}")

            return training_history

        except Exception as e:
            raise CustomException(e, sys)