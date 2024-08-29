import sys
import os
import logging
import datetime as d
sys.path.append("..")
from src.models.ModelTrainer import ModelTrainer
from src.visualization.ModelAnalyzer import ModelAnalyzer
from src.models.prune_model import prune_model
from src import ACTIVATIONS_PATH, MODEL_PATH, NUMBER_TO_PRUNE, PRUNED_MODEL_PATH

def setup_logging():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('ablate.log', mode='a')

    # Set levels for handlers
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    # Create formatters and add to handlers
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_format = logging.Formatter('%(message)s')  # Simplified format for file
    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)

    # Filter to print INFO messages to console
    class InfoFilter(logging.Filter):
        def filter(self, record):
            return record.levelno == logging.INFO

    # Add filter to console handler
    console_handler.addFilter(InfoFilter())

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Add a timestamped heading to the log file
    timestamp = d.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_handler.emit(logging.LogRecord(
        name='', level=logging.WARNING,
        pathname='', lineno=0,
        msg=f"\n\n{'='*50}\nScript Run: {timestamp}\n{'='*50}\n",
        args=(), exc_info=None
    ))

    logging.info(f"Logging warnings and errors to file: {os.path.abspath('warnings.log')}")

if __name__ == "__main__":
    setup_logging()  # Initialize logging
    basic_analyser = ModelAnalyzer(MODEL_PATH, ACTIVATIONS_PATH)
    neurons_to_prune = basic_analyser.identify_concept_neurons()

    # Check if the pruned model file exists
    if os.path.exists(PRUNED_MODEL_PATH):
        # If the file exists, load the pruned model
        model_trainer = ModelTrainer()
        # pruned_model = get_pruned_model(PRUNED_MODEL_PATH, model_trainer)
        logging.info(f"Pruned model exists at {PRUNED_MODEL_PATH}.")
    else:
        # Ablate neurons
        pruned_model = prune_model(MODEL_PATH, neurons_to_ablate=neurons_to_prune[-NUMBER_TO_PRUNE:])
        logging.info(f"Saving pruned model to file: {os.path.abspath(PRUNED_MODEL_PATH)}")
        pruned_model.save_pretrained(PRUNED_MODEL_PATH)