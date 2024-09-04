# LLM Unlearning

This software analyses, ablates, and retrains a target large language model, effectively "unlearning" a given concept in the target model. It allows users to identify important neurons associated with the target concepts, prune the most salient neurons, and retrain the model using custom datasets.

## Table of Contents

-  [Features](#features)
-  [Requirements](#requirements)
-  [Installation](#installation)
-  [Configuration](#configuration)
-  [Directory Structure](#directory-structure)
-  [Usage](#usage)
-  [Logging](#logging)

## Features

-  Load pretrained models from various sources (e.g., Hugging Face).
-  Identify and rank neurons based on their relevance to the target concept.
-  Prune neurons to remove the concept.
-  Retrain the model with a custom dataset to regain performance.
-  Log all operations and configuration parameters for reproducibility.

## Requirements

Python 3.10.10
Required libraries:
  - torch (for model handling)
  - transformers (for loading models)
  - NeuroX (for neuronal analysis)

If using, set the python version using PyEnv, then create a virtual environment and install the required libraries using pip:

```bash
pyenv local 3.10.10
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/UnitTestStudio/unlearning/
   cd unlearning
   ```

2. Install the required libraries as mentioned above.

## Configuration

The application uses a JSON configuration file located in the config directory. The configuration file (config.json) should be structured following this example:

```json
{
    "base_model": {
        "base_model_path": "openai-community/gpt2",
        "model_type": "gpt2",
        "neurons_per_layer": 768,
        "num_layers": 12
    },
    "neural_probing": {
        "concept_definition": "data/target_words.txt",
        "tokens_input_path": "data/tcn-combined-tokens-2k.txt",
        "labels_input_path": "data/tcn-combined-labels-2k.txt",
        "target_label": "target",
        "activations_label": "2k",
        "prune_ratio": 0.2
    },
    "retraining": {
        "train_dataset_path": "data/filtered_train_oasst2", 
        "val_dataset_path": "data/filtered_val_oasst2",
        "num_train_epochs": 1,
        "weight_decay": 0.01,
        "batch_size": 4
    },
    "test_prompts": [
        "Paris is the capital city of",
        "I'm tired, I need to rest on this",
        "Q: What is a chair? A:",
        "Q: What is a uuisen? A:"
    ]
}

```

### Configuration Parameters
Base Model
- base_model_path: Path or Hugging Face identifier for the pretrained model.
- model_type: Type of model (e.g., gpt2).
- neurons_per_layer: Number of neurons per layer.
- num_layers: Total number of layers in the model.

Neural Probing
- concept_definition: Path to the file containing a list of target words that define the concept.
- tokens_input_path: Path to the input tokens file.
- labels_input_path: Path to the input labels file.
- target_label: The target label for probing.
- activations_label: Label for the activations.
- prune_ratio: Ratio of neurons to prune.

Retraining
- train_dataset_path: Path to the training dataset.
- val_dataset_path: Path to the validation dataset.
- num_train_epochs: Number of epochs for retraining.
- weight_decay: Weight decay for the optimizer.
- batch_size: Batch size for training.

Testing
- test_prompts: List of prompts for testing the model.


## Directory Structure

```
unlearning/
│
├── config.json
├── data
│   ├── activations
│   ├── ...
├── models
│   └── ...
├── logs/                       
│   └── app.log                 
├── src
│   ├── config_loader.py
│   ├── logger.py
│   ├── model_handler.py
│   └── tester.py
└── main.py     
```

## Usage

To run the application, execute the following command in your terminal:

```bash
source venv/bin/activate
python main.py
```

The application will log all operations and configuration parameters to logs/app.log.

## Logging

The application uses Python's built-in logging module to log information about the operations performed. All configuration parameters and actions are logged for reproducibility. The log file is located in the logs directory.
