from transformers import GPT2LMHeadModel, GPTNeoForCausalLM, GPT2Tokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk, DatasetDict, load_dataset
import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.interpretation.probeless as probeless
import neurox.interpretation.utils as utils
import neurox.data.loader as data_loader
import logging
import torch
import os

# Create a logger for this module
logging = logging.getLogger(__name__)

def load_model(model_path, model_type):
    logging.info(f"Loading model from {model_path}")
    if model_type == "gpt_neo": #SOMETHING LIKE THIS
        model = GPTNeoForCausalLM.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        logging.info("Loaded GPT-NEO model successfully.")
    elif model_type == "gpt2":
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        logging.info("Loaded GPT-2 model successfully.")

    return model, tokenizer

class ModelAnalyzer:
    def __init__(self, config):
        self.config = config
        self.model = None

    def load_activations(self):
        model_path_type = self.config["base_model_path"] + "," + self.config["model_type"]
        if not os.path.exists(self.config["activations_file_path"]):
            logging.info("Activations path does not exist. Extracting representations...")
            transformers_extractor.extract_representations(
                model_path_type, self.config['tokens_input_path'], 
                self.config["activations_file_path"], 
                aggregation="average"
            )
            logging.info("Representations extracted to %s", self.config["activations_file_path"])
        else:
            logging.info("Activations path exists: %s", self.config["activations_file_path"])

        self.activations, _ = data_loader.load_activations(self.config["activations_file_path"])
        logging.info("Activations loaded from %s", self.config["activations_file_path"])
        return self.activations

class ModelTrainer:
    def __init__(self, config, activations):
        self.config = config
        self.activations = activations
        self.model = None
        self.tokenizer = None
        self.tokens = None
        self.top_neurons = None
        self.X = None
        self.y = None
        self.idx2label = None
        self.label2idx = None

    def load_tokens(self):
        logging.info("Loading tokens from %s", self.config['tokens_input_path'])

        print(self.config['tokens_input_path'])
        print(self.config['labels_input_path'])
        print(self.activations)

        self.tokens = data_loader.load_data(
            self.config['tokens_input_path'], self.config['labels_input_path'], self.activations, 512
        )
        logging.info("Tokens loaded. Creating tensors...")
        self.X, self.y, mapping = utils.create_tensors(
            self.tokens, self.activations, "NN"
        )
        self.label2idx, self.idx2label, _, _ = mapping
        logging.info("Tensors created. Label mapping established.")
        
    def identify_concept_neurons(self):
        if self.tokens is None:
            self.load_tokens()
        logging.info("Identifying concept neurons...")
        top_neurons = probeless.get_neuron_ordering_for_tag(
            self.X, self.y, self.label2idx, self.config["target_label"]
        )
        logging.debug("Top neurons: %s", top_neurons[:10])
        logging.info("%d top neurons identified", len(top_neurons))
        self.top_neurons = top_neurons
    
    def prune(self):
        logging.info("Pruning model from path: %s", self.config['base_model_path'])
        self.model, self.tokenizer = load_model(self.config['base_model_path'], self.config['model_type'])
        self.identify_concept_neurons()
        max_no_neurons_to_prune = round((self.config['neurons_per_layer'] * self.config['num_layers']) * self.config['prune_ratio'])

        for neuron_pos in self.top_neurons[-max_no_neurons_to_prune:]:
            layer_id, neuron_index = divmod(neuron_pos, (self.config['neurons_per_layer']))
            weights = self.model.transformer.h[layer_id - 1].ln_2.weight.data

            # Prune the specified neuron by setting its weight to zero
            weights[neuron_index] = torch.zeros_like(weights[neuron_index])
            weights.requires_grad = False
            # logging.debug("Neuron at layer %d, index %d pruned.", layer_id, neuron_index)

        logging.info("Model pruning completed. %d neurons ablated.", len(self.top_neurons))
        self.model.save_pretrained(self.config['pruned_model_path'])
    
    def load_datasets(self, train_dataset_path, val_dataset_path):
        logging.info("Loading training dataset from path: %s", train_dataset_path)
        try:
            train_dataset = load_from_disk(train_dataset_path)
            logging.info("Successfully loaded training dataset with %d samples.", len(train_dataset))
        except Exception as e:
            logging.error("Error loading training dataset: %s", e)
            raise

        logging.info("Loading validation dataset from path: %s", val_dataset_path)
        try:
            val_dataset = load_from_disk(val_dataset_path)
            logging.info("Successfully loaded validation dataset with %d samples.", len(val_dataset))
        except Exception as e:
            logging.error("Error loading validation dataset: %s", e)
            raise

        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
    
    def tokenize_dataset(self, max_length=1024):
        logging.info("Tokenizing datasets...")

        # Define the tokenization function
        def tokenize_function(examples):
            return self.tokenizer(
            examples["text"],  # Use the 'text' column for tokenization
            padding=False,  # Do not pad here; the DataCollator will handle it
            truncation=True,  # Enable truncation
            max_length=max_length  # Set max_length to prevent exceeding model limits
            )

        # Apply the tokenization function to the entire dataset
        self.tokenized_dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            batch_size=10,
            remove_columns=self.dataset["train"].column_names  # Remove original columns
        )

        logging.info("Tokenization completed successfully.")


    def retrain(self):
        logging.info("Retraining pruned model from path: %s", self.config['pruned_model_path'])
        if self.tokenizer == None:
            self.model, self.tokenizer = load_model(self.config['pruned_model_path'], self.config['model_type'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.resize_token_embeddings(len(self.tokenizer))
        logging.info("Pad token added and model token embeddings resized.")

        self.dataset = self.load_datasets(self.config['train_dataset_path'], self.config['val_dataset_path'])
        self.tokenize_dataset()
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        args = TrainingArguments(
            "retrained_model",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=12,
            weight_decay=0.01,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
        )

        logging.info("Initializing Trainer...")
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=args,
            data_collator=self.data_collator,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"],
        )
        logging.info("Trainer initialized successfully.")

        trainer.train()
        logging.info("Retraining completed successfully.")