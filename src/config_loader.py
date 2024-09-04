import json

def generate_activations_file_path(config):
    model_name = config["base_model"]["base_model_path"].replace("/", "-")
    activations_label = config["neural_probing"]["activations_label"]
    return f"data/activations/{model_name}_{activations_label}_activations.json"

def generate_model_path(config, model_type):
    model_name = config["base_model"]["base_model_path"].replace("/", "-")
    activations_label = config["neural_probing"]["activations_label"]
    prune_ratio = str(int(config['neural_probing']["prune_ratio"] * 100)).zfill(2)  # Convert to percentage and pad with zeros
    return f"models/{model_name}_{activations_label}_{prune_ratio}_{model_type}"

def load_config(config_path='config.json'):
    with open(config_path) as config_file:
        config = json.load(config_file)

    # Create necessary file paths based on the configuration
    config["neural_probing"]['activations_file_path'] = generate_activations_file_path(config)
    config["neural_probing"]['pruned_model_path'] = generate_model_path(config, "pruned")
    config["retraining"]['retrained_model_path'] = generate_model_path(config, "retrained")
    
    return config