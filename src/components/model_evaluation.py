import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object, read_yaml_config

@dataclass
class ModelEvaluationConfig:
    report_path: str = os.path.join("artifacts", "evaluation_report.json")
    comparison_img_path: str = os.path.join("artifacts", "model_comparison.png")
    confusion_matrix_path: str = os.path.join("artifacts", "confusion_matrix.png")

class ModelEvaluation:
    def __init__(self):
        self.eval_config = ModelEvaluationConfig()
        self.config = read_yaml_config("config.yaml")

    def _save_plot(self, fig, path):
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)

    def initiate_evaluation(self, X_test, y_test):
        logging.info("Initiating Model Evaluation")
        try:
            # Paths to models
            model_paths = {
                "BiLSTM": self.config.get("model_training", {}).get("bilstm_model_path", "artifacts/bilstm_model.h5"),
                "LSTM": self.config.get("model_training", {}).get("lstm_model_path", "artifacts/lstm_model.h5"),
                "CNN": self.config.get("model_training", {}).get("cnn_model_path", "artifacts/cnn_model.h5")
            }

            label_encoder = load_object(self.config.get("data_transformation", {}).get("label_encoder_path", "artifacts/label_encoder.pkl"))
            class_names = label_encoder.classes_

            report = {}
            best_model_name = None
            best_f1 = 0
            best_y_pred = None

            # Evaluate each model
            for name, path in model_paths.items():
                logging.info(f"Loading and evaluating {name} from {path}")
                model = load_model(path)
                
                y_pred_probs = model.predict(X_test)
                y_pred = np.argmax(y_pred_probs, axis=1)

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='macro')
                rec = recall_score(y_test, y_pred, average='macro')
                f1 = f1_score(y_test, y_pred, average='macro')

                report[name] = {
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1_Score": f1
                }

                if f1 > best_f1:
                    best_f1 = f1
                    best_model_name = name
                    best_y_pred = y_pred

            logging.info(f"Saving evaluation report to {self.eval_config.report_path}")
            with open(self.eval_config.report_path, "w") as f:
                json.dump(report, f, indent=4)

            # 1. Generate Model Comparison Bar Chart
            logging.info("Generating Model Comparison Chart")
            models = list(report.keys())
            f1_scores = [report[m]["F1_Score"] for m in models]
            acc_scores = [report[m]["Accuracy"] for m in models]

            x = np.arange(len(models))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(x - width/2, acc_scores, width, label='Accuracy', color='skyblue')
            ax.bar(x + width/2, f1_scores, width, label='F1 Score (Macro)', color='salmon')
            
            ax.set_ylabel('Scores')
            ax.set_title('Deep Learning Models Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend()
            self._save_plot(fig, self.eval_config.comparison_img_path)

            # 2. Generate Confusion Matrix for the Best Model
            logging.info(f"Generating Confusion Matrix for best model: {best_model_name}")
            cm = confusion_matrix(y_test, best_y_pred)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_ylabel('Actual Category')
            ax.set_xlabel('Predicted Category')
            ax.set_title(f'Confusion Matrix ({best_model_name})')
            self._save_plot(fig, self.eval_config.confusion_matrix_path)

            logging.info("Model Evaluation completed successfully.")
            return report

        except Exception as e:
            raise CustomException(e, sys)