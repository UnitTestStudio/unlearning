import logging
from transformers import AutoModelForTokenClassification, GPT2LMHeadModel
from src import NEURONS_PER_LAYER, tokenizer
import torch

# Create a logger for this module
logger = logging.getLogger(__name__)

def prune_model(model_path: str, neurons_to_ablate):
    logger.info("Pruning model from path: %s", model_path)
    
    # Load the GPT-2 model
    model = GPT2LMHeadModel.from_pretrained(model_path)
    logger.info("Loaded GPT-2 model successfully.")

    for neuron_pos in neurons_to_ablate:
        layer_id, neuron_index = divmod(neuron_pos, NEURONS_PER_LAYER)

        # FOR GPT2
        weights = model.transformer.h[layer_id - 1].ln_2.weight.data

        # FOR DISTILBERT (commented out)
        # weights = model.distilbert.transformer.layer[
        #     layer_id - 1
        # ].output_layer_norm.weight.data

        # Prune the specified neuron by setting its weight to zero
        weights[neuron_index] = torch.zeros_like(weights[neuron_index])
        weights.requires_grad = False
        logger.debug("Neuron at layer %d, index %d pruned.", layer_id, neuron_index)

    logger.info("Model pruning completed. %d neurons ablated.", len(neurons_to_ablate))
    return model