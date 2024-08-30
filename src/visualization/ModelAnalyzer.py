import os
import logging
import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader
import neurox.interpretation.utils as utils
import neurox.interpretation.probeless as probeless
import neurox.analysis.corpus as corpus
from src import TOKENS_INPUT_PATH, TOKENS_LABEL_PATH, MODEL_CHECKPOINT, CONCEPT_LABEL

# Create a logger for this module
logger = logging.getLogger(__name__)

class ModelAnalyzer:
    def __init__(self, model_path, activations_path) -> None:
        self.activations = None
        self.tokens = None
        self.X = None
        self.y = None
        self.idx2label = None
        self.label2idx = None
        logger.info("Initializing ModelAnalyzer with model_path: %s and activations_path: %s", model_path, activations_path)
        self.load_activations(model_path, activations_path)

    def load_activations(self, model_path, activations_path):
        model_path_type = model_path + "," + MODEL_CHECKPOINT
        if not os.path.exists(activations_path):
            logger.info("Activations path does not exist. Extracting representations...")
            transformers_extractor.extract_representations(
                model_path_type, TOKENS_INPUT_PATH, activations_path, aggregation="average"
            )
            logger.info("Representations extracted to %s", activations_path)
        else:
            logger.info("Activations path exists: %s", activations_path)

        self.activations, _ = data_loader.load_activations(activations_path)
        logger.info("Activations loaded from %s", activations_path)

    def load_tokens(self):
        logger.info("Loading tokens from %s", TOKENS_INPUT_PATH)
        self.tokens = data_loader.load_data(
            TOKENS_INPUT_PATH, TOKENS_LABEL_PATH, self.activations, 512
        )
        logger.info("Tokens loaded. Creating tensors...")
        self.X, self.y, mapping = utils.create_tensors(
            self.tokens, self.activations, "NN"
        )
        self.label2idx, self.idx2label, _, _ = mapping
        logger.info("Tensors created. Label mapping established.")

    def identify_concept_neurons(self):
        if self.tokens is None:
            self.load_tokens()
        logger.info("Identifying concept neurons...")
        top_neurons = probeless.get_neuron_ordering_for_tag(
            self.X, self.y, self.label2idx, CONCEPT_LABEL
        )
        logger.debug("Top neurons: %s", top_neurons[:10])
        logger.info("%d top neurons identified", len(top_neurons))
        return top_neurons

    def show_top_words(self, concept_neurons):
        if self.tokens is None:
            logger.warning("Tokens are None. Loading tokens...")
            self.load_tokens()
        top_words = {}
        logger.info("Showing top words for concept neurons...")
        for neuron_idx in concept_neurons:
            words = corpus.get_top_words(
                self.tokens, self.activations, neuron_idx, 5)
            top_words[neuron_idx] = words
            logger.info("Top words for neuron %d: %s", neuron_idx, words)
            print(f"== {neuron_idx} ==")
            print(words)
        return top_words
