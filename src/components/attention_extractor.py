import sys
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.exception import CustomException

class AttentionExtractor:
    def __init__(self, model, tokenizer, max_len):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len

    def get_word_importances(self, text: str, predicted_class_index: int):
        """
        Calculates word importance using Occlusion Sensitivity.
        Masks each word and measures the drop in prediction probability.
        """
        try:
            words = text.split()
            if not words:
                return []

            # Base prediction probability
            base_seq = self.tokenizer.texts_to_sequences([" ".join(words)])
            base_pad = pad_sequences(base_seq, maxlen=self.max_len, padding="post", truncating="post")
            base_prob = self.model.predict(base_pad, verbose=0)[0][predicted_class_index]

            importances = []
            
            for i in range(len(words)):
                # Create a version of the text with the i-th word removed/masked
                masked_words = words.copy()
                masked_words[i] = "<OOV>" # Replace with out-of-vocabulary token
                
                masked_text = " ".join(masked_words)
                masked_seq = self.tokenizer.texts_to_sequences([masked_text])
                masked_pad = pad_sequences(masked_seq, maxlen=self.max_len, padding="post", truncating="post")
                
                # Get prediction with the word masked
                masked_prob = self.model.predict(masked_pad, verbose=0)[0][predicted_class_index]
                
                # The importance is the drop in probability. 
                # If removing the word drops the probability a lot, it was highly important.
                drop = base_prob - masked_prob
                importances.append((words[i], float(drop)))

            # Normalize importances between 0 and 1 for heatmap visualization
            max_drop = max([imp[1] for imp in importances] + [1e-9]) # prevent div by zero
            normalized_importances = [(word, max(0, val / max_drop)) for word, val in importances]

            return normalized_importances

        except Exception as e:
            raise CustomException(e, sys)