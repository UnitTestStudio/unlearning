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
-  [Testing](#testing)
-  [Contributing](#contributing)
-  [License](#license)

## Features

-  Load pretrained models from various sources (e.g., Hugging Face).
-  Identify and rank neurons based on their relevance to the target concept.
-  Prune neurons to remove the concept.
-  Retrain the model with a custom dataset to regain performance.
-  Log all operations and configuration parameters for reproducibility.

## Requirements

-  Python 3.10.10
-  Required libraries:
  - torch (for model handling)
  - transformers (for loading models)
  - NeuroX (for model analysis)

You can install the required libraries using pip:

```bash
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

The application uses a JSON configuration file located in the config directory. The configuration file (config.json) should include the following parameters:

```json
{
    "base_model_path": "<target/base-model-path can be huggging face repo>",
    "model_checkpoint": "<target/base-model-path-checkpoint>",
    "model_type": "<model type e.g gpt2>",
    "train_dataset_path": "<path to retraining data>", 
    "val_dataset_path": "<path to retraining data>",
    "activations_dataset_size": "<activations_dataset_size>",
    "neurons_per_layer": "<neurons_per_layer>",
    "num_layers": "<number of layers>",
    "prune_ratio": "<percentage of neurons to prune>",
    "target_label": "<label specified in the annotation dataset lablels>",
    "test_prompts": [
        "A list",
        "of prompts",
        "to test the models with...",
    ]
}

```

### Parameters

-  model_path: Path or identifier for the pretrained model.
-  model_checkpoint: Checkpoint for the model.
-  model_type: Type of model (e.g., gpt2).
-  dataset_size: Size of the dataset (e.g., 2k).
-  neurons_per_layer: Number of neurons per layer.
-  num_layers: Total number of layers in the model.
-  prune_ratio: Ratio of neurons to prune.
-  test_prompts: List of prompts for testing the model.

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
