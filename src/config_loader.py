import json
import os

def generate_activations_file_path(config):
    model_name = config["base_model"]["base_model_path"].replace("/", "-")
    activations_label = config["neural_probing"]["activations_label"]
    return f"data/activations/{model_name}_{activations_label}_activations.json"

def generate_model_path(config, model_type):
    model_name = config["base_model"]["base_model_path"].replace("/", "-")
    activations_label = config["neural_probing"]["activations_label"]
    prune_ratio = str(config['neural_probing']["prune_ratio"])
    if model_type == "pruned":
        return f"models/{model_name}_{activations_label}activations_{prune_ratio}pruned_{model_type}"
    elif model_type == "retrained":
        return f"models/{model_name}_{activations_label}activations_{prune_ratio}pruned_{model_type}_{config['retraining']["num_train_epochs"]}epochs"

def load_config(config_path='config.json'):
    os.makedirs("data/activations/", exist_ok=True)
    os.makedirs("logs/", exist_ok=True)
    os.makedirs("models/", exist_ok=True)

    with open(config_path) as config_file:
        config = json.load(config_file)
    # Create necessary file paths based on the configuration
    config["neural_probing"]['activations_file_path'] = generate_activations_file_path(config)
    config["neural_probing"]['pruned_model_path'] = generate_model_path(config, "pruned")
    config["retraining"]['retrained_model_path'] = generate_model_path(config, "retrained")
    config["retraining"]["train_dataset_path"] = "data/filtered_train"
    config["retraining"]["val_dataset_path"] =  "data/filtered_val"
    return config
