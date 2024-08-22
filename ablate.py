import sys
import os
import logging
import datetime as d
sys.path.append("..")
from src.models.ModelTrainer import ModelTrainer
from src.visualization.ModelAnalyzer import ModelAnalyzer
from src.models.prune_model import prune_model
from src import NEURONS_PER_LAYER, NUM_LAYERS

def setup_logging():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('unlearning.log', mode='a')

    # Set levels for handlers
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

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
    
    model_path = "openai-community/gpt2"
    activations_path = "gpt-2-activations.json"
    basic_analyser = ModelAnalyzer(model_path, activations_path)
    neurons_to_prune = basic_analyser.identify_concept_neurons()

    # If the model has been saved
    # pruned_model = get_pruned_model(pruned_model_path, model_trainer)
    pruned_model_path = "models/pruned_gpt2_model"
    model_trainer = ModelTrainer()
    # Ablate neurons
    num_prune = (NEURONS_PER_LAYER * NUM_LAYERS) // 3 # this is where the percentage of nuerons to prune is set
    pruned_model = prune_model(model_path, model_trainer, neurons_to_prune[-num_prune:])
    logging.info(f"Saving pruned model to file: {os.path.abspath(pruned_model_path)}")
    pruned_model.save_pretrained(pruned_model_path)