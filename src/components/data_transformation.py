import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, read_yaml_config

@dataclass
class DataTransformationConfig:
    tokenizer_path: str = os.path.join("artifacts", "tokenizer.pkl")
    label_encoder_path: str = os.path.join("artifacts", "label_encoder.pkl")
    glove_matrix_path: str = os.path.join("artifacts", "glove_matrix.npy")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        self.config = read_yaml_config("config.yaml")

    def get_data_transformer_object(self):
        try:
            # Fetch parameters from config
            max_words = self.config.get("data_transformation", {}).get("max_words", 20000)
            tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
            label_encoder = LabelEncoder()
            return tokenizer, label_encoder
        except Exception as e:
            raise CustomException(e, sys)

    def load_glove_embeddings(self, word_index, embedding_dim=100):
        """
        Loads pre-trained GloVe embeddings and creates an embedding matrix.
        Assumes glove.6B.100d.txt is present in the root directory.
        """
        embeddings_index = {}
        glove_path = f"glove.6B.{embedding_dim}d.txt"
        
        logging.info("Building GloVe embedding matrix.")
        try:
            if os.path.exists(glove_path):
                with open(glove_path, encoding="utf8") as f:
                    for line in f:
                        values = line.split()
                        word = values[0]
                        coefs = np.asarray(values[1:], dtype='float32')
                        embeddings_index[word] = coefs
                logging.info(f"Loaded {len(embeddings_index)} word vectors from GloVe.")
            else:
                logging.warning(f"{glove_path} not found. Proceeding with random embeddings initialization.")

            # Create embedding matrix
            max_words = self.config.get("data_transformation", {}).get("max_words", 20000)
            vocab_size = min(max_words, len(word_index) + 1)
            embedding_matrix = np.zeros((vocab_size, embedding_dim))
            
            for word, i in word_index.items():
                if i >= max_words:
                    continue
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # Words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
            
            # Save the matrix
            np.save(self.transformation_config.glove_matrix_path, embedding_matrix)
            logging.info("Saved GloVe embedding matrix to artifacts.")
            
            return embedding_matrix, vocab_size

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Starting Data Transformation")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Drop NaNs in the text column
            text_col = "consumer_complaint_narrative"
            label_col = "product" # We encode the text category

            train_df = train_df.dropna(subset=[text_col])
            test_df = test_df.dropna(subset=[text_col])

            tokenizer, label_encoder = self.get_data_transformer_object()

            # 1. Transform Target Labels
            logging.info("Encoding target labels.")
            y_train = label_encoder.fit_transform(train_df[label_col])
            y_test = label_encoder.transform(test_df[label_col])

            # 2. Transform Text Data
            logging.info("Fitting Tokenizer on training text.")
            tokenizer.fit_on_texts(train_df[text_col])

            X_train_seq = tokenizer.texts_to_sequences(train_df[text_col])
            X_test_seq = tokenizer.texts_to_sequences(test_df[text_col])

            # 3. Pad Sequences
            max_len = self.config.get("data_transformation", {}).get("max_len", 150)
            logging.info(f"Padding sequences to max length of {max_len}.")
            X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
            X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")

            # 4. Generate & Save Embeddings Matrix
            embedding_dim = self.config.get("data_transformation", {}).get("embedding_dim", 100)
            embedding_matrix, vocab_size = self.load_glove_embeddings(tokenizer.word_index, embedding_dim)

            # 5. Save Tokenizer and Label Encoder
            save_object(self.transformation_config.tokenizer_path, tokenizer)
            save_object(self.transformation_config.label_encoder_path, label_encoder)
            
            logging.info("Data Transformation completed successfully.")

            return (
                X_train_pad, X_test_pad,
                y_train, y_test,
                embedding_matrix, vocab_size
            )

        except Exception as e:
            raise CustomException(e, sys)