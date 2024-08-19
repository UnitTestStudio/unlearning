from transformers import AutoTokenizer

# Model specific constants
NEURONS_PER_LAYER = 768
NUM_LAYERS = 6 # DistilBERT, DistilGPT2
# MODEL_CHECKPOINT = "distilbert-base-cased"
MODEL_CHECKPOINT = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

BASIC_MODEL_PATH = "models/basic_model"
PRUNED_MODEL_PATH = "models/pruned_model"
RETRAINED_MODEL_PATH = "models/retrained_model"

CONCEPT_LABEL = "target"
TOKENS_INPUT_PATH = "/home/martindisley/Workspace/drw-unlearning/unlearning/tcn-combined-tokens.txt"
TOKENS_LABEL_PATH = "/home/martindisley/Workspace/drw-unlearning/unlearning/tcn-combined-labels.txt"
BASIC_ACTIVATIONS_PATH = "basic_activations.json"
RETRAINED_ACTIVATIONS_PATH = "retrained_activations.json"