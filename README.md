# LLM Pruning and Retraining

This project provides a framework for analyzing, pruning, and retraining neural network models in a model-agnostic manner. It allows users to identify important neurons associated with specific concepts, prune less relevant neurons, and retrain the model using custom datasets. The application is designed to be extensible and maintainable, following best practices in Python software development.

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
-  Identify and rank neurons based on their relevance to specified concepts.
-  Prune neurons to reduce model complexity.
-  Retrain the model with a custom dataset.
-  Log all operations and configuration parameters for reproducibility.

## Requirements

-  Python 3.8 or higher
-  Required libraries:
  - torch (for model handling)
  - transformers (for loading models)
  - numpy (for numerical operations)
  - pandas (for dataset handling)
  - json (for configuration management)

You can install the required libraries using pip:

```bash
pip install torch transformers numpy pandas
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/my_ml_project.git
   cd my_ml_project
   ```

2. Install the required libraries as mentioned above.

## Configuration

The application uses a JSON configuration file located in the config directory. The configuration file (config.json) should include the following parameters:

```json
{
    "model_path": "openai/gpt-2",
    "model_checkpoint": "openai/gpt-2",
    "model_type": "gpt2",
    "dataset_size": "2k",
    "neurons_per_layer": 2048,
    "num_layers": 24,
    "prune_ratio": 0.2,
    "test_prompts": [
        "Paris is the capital city of",
        "I'm tired, I think I'll rest on this",
        "Q: What is a chair? A:",
        "Q: What is a poticam? A:"
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
my_ml_project/
│
├── config/                     # Configuration files
│   └── config.json             # Main configuration file
│
├── data/                       # Data files (datasets, activations)
│   └── dataset.csv             # Example dataset
│
├── logs/                       # Log files
│   └── app.log                 # Application log file
│
├── src/                        # Source code
│   ├── __init__.py             # Makes src a package
│   ├── logger.py                # Logging setup
│   ├── config_loader.py         # Configuration loading logic
│   ├── model_handler.py         # Model loading, saving, and pruning logic
│   ├── neuron_identifier.py     # Logic for identifying concept neurons
│   ├── retrainer.py             # Logic for retraining the model
│   ├── tester.py                # Logic for testing the model
│
└── main.py                     # Entry point for the application
```

## Usage

To run the application, execute the following command in your terminal:

```bash
python main.py
```

The application will log all operations and configuration parameters to logs/app.log.

## Logging

The application uses Python's built-in logging module to log information about the operations performed. All configuration parameters and actions are logged for reproducibility. The log file is located in the logs directory.

## Testing

Unit tests can be added in the tests directory. Ensure that you write tests for each module to verify the functionality of the application. You can use the unittest framework or any other testing framework of your choice.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
