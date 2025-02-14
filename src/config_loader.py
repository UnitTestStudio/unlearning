import json
import os

def generate_activations_file_path(config):
    model_name = config["base_model"]["base_model_path"].replace("/", "-")
    dataset_name = config["neural_pruning"]["data"].split("/")[-1].split(".")[0]
    return f"data/{dataset_name}_{model_name}_activations.npz"

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

    if config['neural_pruning']['activations_file_path'] == None:
        config['neural_pruning']['compute_activations'] = True
    else:
        config['neural_pruning']['compute_activations'] = False
    config["neural_pruning"]["pruned_model_path"] = generate_model_path(config, "pruned")
    config["neural_pruning"]["activations_file_path"] = generate_activations_file_path(config)
    return config