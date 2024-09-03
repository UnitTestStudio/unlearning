import json

def generate_activations_file_path(config):
    model_name = config["base_model_path"].replace("/", "-")
    dataset_size = config["activations_dataset_size"]
    return f"data/activations/{model_name}_{dataset_size}_activations.json"

def generate_model_path(config, model_type):
    model_name = config["base_model_path"].replace("/", "-")
    dataset_size = config["activations_dataset_size"]
    prune_ratio = str(int(config["prune_ratio"] * 100)).zfill(2)  # Convert to percentage and pad with zeros
    return f"models/{model_name}_{dataset_size}_{prune_ratio}_{model_type}"

def generate_dataset_path(config):
    if config["activations_dataset_size"] == "2k":
        tokens_input_path = "/home/martindisley/Workspace/drw-unlearning/unlearning/data/tcn-combined-tokens-2k.txt"
        labels_input_path = "/home/martindisley/Workspace/drw-unlearning/unlearning/data/tcn-combined-labels-2k.txt"
    else:
        tokens_input_path = "/home/martindisley/Workspace/drw-unlearning/unlearning/data/tcn-combined-tokens.txt"
        labels_input_path = "/home/martindisley/Workspace/drw-unlearning/unlearning/data/tcn-combined-labels.txt"
    return tokens_input_path, labels_input_path

def load_config(config_path='config.json'):
    with open(config_path) as config_file:
        config = json.load(config_file)

    # Create necessary file paths based on the configuration
    config['activations_file_path'] = generate_activations_file_path(config)
    config['pruned_model_path'] = generate_model_path(config, "pruned")
    config['retrained_model_path'] = generate_model_path(config, "retrained")
    config['tokens_input_path'], config['labels_input_path'] = generate_dataset_path(config)
    
    return config