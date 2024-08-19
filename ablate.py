import sys
import logging
sys.path.append("..")
from src.models.ModelTrainer import ModelTrainer
from src.visualization.ModelAnalyzer import ModelAnalyzer
from src.models.prune_model import prune_model
from src import BASIC_MODEL_PATH, BASIC_ACTIVATIONS_PATH, PRUNED_MODEL_PATH, NEURONS_PER_LAYER, NUM_LAYERS,

def setup_logging():
    """Set up logging to output messages to the console."""
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the logging level

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set the level for the console handler

    # Create a formatter and set it for the console handler
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

if __name__ == "__main__":
    setup_logging()  # Initialize logging
    # if len(sys.argv) != 4:
    #     print("Usage: python tokens_labels_from_sentences.py <target_words> <sentences> <output_prefix>")
    #     sys.exit(1)
    
    model_path = "openai-community/gpt2"
    activations_path = "gpt-2-activations.json"
    basic_analyser = ModelAnalyzer(model_path, activations_path)
    neurons_to_prune = basic_analyser.identify_concept_neurons()

    # If the model has been saved
    # pruned_model = get_pruned_model(pruned_model_path, model_trainer)
    pruned_model_path = "models/pruned_gpt2_model"
    model_trainer = ModelTrainer()
    # Ablate neurons
    num_prune = (NEURONS_PER_LAYER * NUM_LAYERS) // 2
    pruned_model = prune_model(model_path, model_trainer, neurons_to_prune[-num_prune:])
    pruned_model.save_pretrained(pruned_model_path)