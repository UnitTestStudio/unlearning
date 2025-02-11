import json
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_data(config):
    """Load data from a JSON file specified in the configuration."""
    with open(config["neural_pruning"]["data"], 'r') as f:
        data = json.load(f)
        test_prompts = data.get('test_prompts', [])
        concept_examples = data.get('concept_examples', [])
        background_examples = data.get('background_examples', [])
        return test_prompts, concept_examples, background_examples

def load_model(model, device):
    """Loads a pre-trained causal language model and its tokenizer based on the provided configuration."""
    model = AutoModelForCausalLM.from_pretrained(model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, use_fast=False)
    return model, tokenizer

def save_results(model, tokenizer, model_file_path, concept_neurons = None, concept_neurons_file_path = None):
    """Save the pruned model and neuron indices."""
    # Save model
    model.save_pretrained(model_file_path)
    tokenizer.save_pretrained(model_file_path)
    
    if concept_neurons:
        # Save concept neurons
        with open(concept_neurons_file_path, 'w') as f:
            json.dump(concept_neurons, f)
