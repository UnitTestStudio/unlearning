from src.logger import setup_logging, log_config
from src.config_loader import load_config
from src.model_handler import ModelAnalyzer, ModelTrainer
from src.tester import test_model
import os
import logging

def main():
    # Load configuration
    config = load_config()
    setup_logging()

    # Log configuration parameters
    log_config(config)

    try:
        # Identify top concept neurons
        model_analyzer = ModelAnalyzer(config) # Create an instance of ModelAnalyzer
        activations = model_analyzer.load_activations() # Updated to base_model_path
        test_model(config, "base")

        # Prune the model
        model_trainer = ModelTrainer(config, activations) # Create an instance of ModelTrainer
        if not os.path.exists(config["neural_probing"]['pruned_model_path']):
            model_trainer.prune()
        test_model(config, "pruned")

        #Retrain the model
        if not os.path.exists(config["retraining"]['retrained_model_path']):
            model_trainer.retrain()
        test_model(config, "retrained")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
