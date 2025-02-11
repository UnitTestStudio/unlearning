import json
import os

def generate_concept_neurons_file_path(config):
    model_name = config["base_model"]["base_model_path"].replace("/", "-")
    return f"models/{model_name}_concept_neurons.json"

def generate_model_path(config, model_type):
    model_name = config["base_model"]["base_model_path"].replace("/", "-")
    # prune_ratio = str(config['neural_probing']["prune_percentage"])
    prune_ratio = str(0.9)
    if model_type == "pruned":
        return f"models/{model_name}_{prune_ratio}n_{model_type}"
    elif model_type == "retrained":
        return f"models/{model_name}_{prune_ratio}n_{model_type}_{config['retraining']['num_train_epochs']}_epochs"

def load_config(config_path='config.json'):
    # os.makedirs("data/activations/", exist_ok=True)
    os.makedirs("logs/", exist_ok=True)
    os.makedirs("models/", exist_ok=True)

    with open(config_path) as config_file:
        config = json.load(config_file)

    config["neural_pruning"]["pruned_model_path"] = generate_model_path(config, "pruned")
    config["neural_pruning"]["concept_neurons_path"] = generate_concept_neurons_file_path(config)
    return config