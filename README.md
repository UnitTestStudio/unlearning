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

-  Load a target pretrained model from various sources (e.g., Hugging Face).
-  Create datsets for neuronal probing and analysis based on a target concept .
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
# compatability issue with tqdm install NeuroX first
pip install git+https://github.com/mee-kell/NeuroX@344f2cd1f45a73c39923eaa88d2caf68baa31c0e
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
- activations_label: A label for the activations file, used to name the pruned and retrained model
- prune_ratio: The percentage of neurons in the model to prune.

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
│   ├── target_words.txt
│   ├── filtering-huggingface-dataset.py
│   └── tokens_labels_from_sentences.py
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
### Data Preparation
#### For Analysis
To run analysis with NeuroX, we need first to prepare our data. NeuroX expects a pair of dataset files, one containing the sentences and another with labels for each token in each sentence. 

tokens.txt
```txt
She 's yet to pay for an upper class seat with the airline she uses most , Virgin Atlantic .
Much of the furniture , including a dining table , master bed , plywood sofa and a coffee table , also by Stummel .
Lance Mason was removed from the bench as a sitting Cuyahoga County Common Pleas judge at the time .
This is the time , if there 's one thing this week , a week of worry , a week of turmoil ( ph ) , a week of chaos , but a week where children , teenagers can sit ( ph ) forward and say enough is enough and things may finally change .
Pour the hot cream over the semi-sweet chocolate chips and let it sit for 3 minutes .
A table nearby said they were going and we could sit there but we were just left hanging , not knowing if we could stay or not .
```
labels.txt
```txt
N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A target N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A
N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A target N/A N/A N/A N/A N/A N/A N/A N/A N/A
N/A N/A N/A N/A N/A N/A target N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A
N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A target N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A
N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A target N/A N/A N/A N/A
N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A target N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A
```

The "target" label in `labels.txt` denotes a target token in the sentence. `tokens.txt` and `labels.txt` files can be generated using `data/tokens_labels_from_sentences.py`, supplying a initial list of sentences.

Set the `tokens.txt` and `labels.txt` in the `config.json` to use them in the application.

#### For Retraining
To retrain the model, without reintroducing the target concept we need to filter a dataset. Using `data/filtering-huggingface-dataset.py` we can create filtered training and validation dataset that filters out sentence from a Hugging Face dataset that contain our target concept. 

### Run
Once the datasets have been produced and theire locations specified in the configuration file, we can run the application. To run it, execute the following command in your terminal:

```bash
source venv/bin/activate
python main.py
```

The application will log all operations and configuration parameters to logs/app.log.

## Logging

The application uses Python's built-in logging module to log information about the operations performed. All configuration parameters and actions are logged for reproducibility. The log file is located in the logs directory.
