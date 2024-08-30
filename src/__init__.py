import json
import logging
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Load configuration from config.json
with open('run-config.json', 'r') as config_file:
    config = json.load(config_file)

# Model specific constants
MODEL_PATH = config['model_path']
MODEL_CHECKPOINT = config['model_checkpoint']
DATASET_SIZE = config['dataset_size']
MODEL_NAME = MODEL_PATH.split('/')[1]
ACTIVATIONS_PATH = f"activations/{MODEL_NAME}-{DATASET_SIZE}-activations.json"
NEURONS_PER_LAYER = config['neurons_per_layer']
NUM_LAYERS = config['num_layers']
PRUNE_RATIO = config['prune_ratio']
PRUNED_MODEL_PATH = f"models/pruned-{MODEL_NAME}-{DATASET_SIZE}-{PRUNE_RATIO}"
NUMBER_TO_PRUNE = round((NEURONS_PER_LAYER * NUM_LAYERS) * PRUNE_RATIO)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# Dataset paths
if DATASET_SIZE == "6k":
    TOKENS_INPUT_PATH = "/home/martindisley/Workspace/drw-unlearning/unlearning/data/tcn-combined-tokens.txt"
    TOKENS_LABEL_PATH = "/home/martindisley/Workspace/drw-unlearning/unlearning/data/tcn-combined-labels.txt"
else:
    TOKENS_INPUT_PATH = "/home/martindisley/Workspace/drw-unlearning/unlearning/data/tcn-combined-tokens-2k.txt"
    TOKENS_LABEL_PATH = "/home/martindisley/Workspace/drw-unlearning/unlearning/data/tcn-combined-labels-2k.txt"

# Output paths (you can define these as needed)
BASIC_MODEL_PATH = "models/basic_model"
RETRAINED_MODEL_PATH = "models/retrained_model"

# Concept label (if needed, you can define it here)
CONCEPT_LABEL = "target"

# Log the loaded configuration
logger.info("Loaded configuration:")
logger.info("Model Checkpoint: %s", MODEL_CHECKPOINT)
logger.info("Model Path: %s", MODEL_PATH)
logger.info("Activations Path: %s", ACTIVATIONS_PATH)
logger.info("Neurons per Layer: %d", NEURONS_PER_LAYER)
logger.info("Number of Layers: %d", NUM_LAYERS)
logger.info("Prune Ratio: %d", PRUNE_RATIO)
logger.info("Tokens Input Path: %s", TOKENS_INPUT_PATH)
logger.info("Tokens Label Path: %s", TOKENS_LABEL_PATH)
