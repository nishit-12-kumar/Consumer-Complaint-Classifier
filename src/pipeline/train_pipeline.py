import sys
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("========== Starting Training Pipeline ==========")
            
            # Step 1: Data Ingestion
            ingestion = DataIngestion()
            train_data_path, test_data_path = ingestion.initiate_data_ingestion()
            
            # Step 2: Data Transformation
            transformation = DataTransformation()
            (X_train_pad, X_test_pad, y_train, y_test, 
             embedding_matrix, vocab_size) = transformation.initiate_data_transformation(train_data_path, test_data_path)
            
            # Extract number of unique classes for the output layer
            num_classes = len(set(y_train))
            
            # Step 3: Model Training
            trainer = ModelTrainer(vocab_size=vocab_size, embedding_matrix=embedding_matrix, num_classes=num_classes)
            training_history = trainer.initiate_model_trainer(X_train_pad, y_train, X_test_pad, y_test)
            
            # Step 4: Model Evaluation
            evaluator = ModelEvaluation()
            eval_report = evaluator.initiate_evaluation(X_test_pad, y_test)
            
            logging.info("========== Training Pipeline Completed Successfully ==========")
            return eval_report
            
        except Exception as e:
            logging.error("Pipeline failed!")
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()