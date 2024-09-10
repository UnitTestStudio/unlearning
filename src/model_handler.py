from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel, GPTNeoForCausalLM, GPT2Tokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, MistralForCausalLM
from datasets import load_from_disk, DatasetDict, load_dataset
import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.interpretation.probeless as probeless
import neurox.interpretation.utils as utils
import neurox.data.loader as data_loader
import logging
import torch
import os
import glob
import shutil

# Create a logger for this module
logging = logging.getLogger(__name__)

def load_model(model_path, model_type):
    """
    Load a pre-trained GPT model and its corresponding tokenizer.

    Args:
        model_path (str): The file path to the pre-trained model.
        model_type (str): The type of model to load ('gpt_neo' or 'gpt2').

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    logging.info(f'Loading model from {model_path}')
    try:
        if model_type == 'gpt_neo':
            model = GPTNeoForCausalLM.from_pretrained(model_path)
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            logging.info('Loaded GPT-NEO model successfully.')
        elif model_type == 'gpt2':
            model = GPT2LMHeadModel.from_pretrained(model_path)
            tokenizer = GPT2Tokenizer.from_pretrained(model_type)
            logging.info('Loaded GPT-2 model successfully.')
        elif model_type == 'mistral':
            model = MistralForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            logging.info('Loaded GPT-2 model successfully.')
        else:
            model = AutoModel.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            logging.info('Loaded AutoModel model successfully.')
    except Exception as e:
        logging.error(f'Failed to load model of type {model_type} from path {model_path}: {e}')

    return model, tokenizer

class ModelAnalyzer:
    def __init__(self, config):
        self.config = config
        self.model = None

    def load_activations(self):
        """
        Load the NeuroX activations file. If the file does not exist, 
        it triggers the analysis process to extract representations.

        Returns:
            activations: The loaded activations data.
        """
        if not os.path.exists(self.config['neural_probing']['activations_file_path']):
            logging.info('Activations path does not exist. Extracting representations...')
            transformers_extractor.extract_representations(
                self.config['base_model']['base_model_path'], 
                self.config['neural_probing']['tokens_input_path'], 
                self.config['neural_probing']['activations_file_path'], 
                aggregation='average'
            )
            logging.info('Representations extracted to %s', self.config['neural_probing']['activations_file_path'])
        else:
            logging.info('Activations path exists: %s', self.config['neural_probing']['activations_file_path'])

        self.activations, _ = data_loader.load_activations(self.config['neural_probing']['activations_file_path'])
        logging.info('Activations loaded from %s', self.config['neural_probing']['activations_file_path'])
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
        """
        Load the analysis dataset for the NeuroX analysis and convert 
        token strings to tensors.

        Raises:
            Exception: If there is an error loading the data.
        """
        logging.info('Loading tokens from %s', self.config['neural_probing']['tokens_input_path'])

        print(self.config['neural_probing']['tokens_input_path'])
        print(self.config['neural_probing']['labels_input_path'])
        print(self.activations)

        self.tokens = data_loader.load_data(
            self.config['neural_probing']['tokens_input_path'], self.config['neural_probing']['labels_input_path'], self.activations, 512
        )
        logging.info('Tokens loaded. Creating tensors...')
        self.X, self.y, mapping = utils.create_tensors(
            self.tokens, self.activations, 'NN'
        )
        self.label2idx, self.idx2label, _, _ = mapping
        logging.info('Tensors created. Label mapping established.')

    def identify_concept_neurons(self):
        """
        Identify the model's neurons based on their saliency to the target concept.

        This method loads tokens if they are not already loaded and 
        logs the identified top neurons.
        """
        if self.tokens is None:
            self.load_tokens()
        logging.info('Identifying concept neurons...')
        top_neurons = probeless.get_neuron_ordering_for_tag(
            self.X, self.y, self.label2idx, self.config['neural_probing']['target_label']
        )
        logging.debug('Top neurons: %s', top_neurons[:10])
        logging.info('%d top neurons identified', len(top_neurons))
        self.top_neurons = top_neurons
    
    def prune(self):
        """
        Ablate the model by setting the weights of the target neurons to zero.

        This method loads the model, identifies concept neurons, and 
        prunes the specified number of neurons based on the configuration.
        """
        logging.info('Pruning model from path: %s', self.config['base_model']['base_model_path'])
        self.model, self.tokenizer = load_model(self.config['base_model']['base_model_path'], self.config['base_model']['model_type'])
        self.identify_concept_neurons()
        max_no_neurons_to_prune = round((self.config['base_model']['neurons_per_layer'] * self.config['base_model']['num_layers']) * self.config['neural_probing']['prune_ratio'])

        for neuron_pos in self.top_neurons[-max_no_neurons_to_prune:]:
            layer_id, neuron_index = divmod(neuron_pos, (self.config['base_model']['neurons_per_layer']))
            if self.config['base_model']['model_type'] == 'mistral':
                weights = self.model.model.layers[layer_id -1].post_attention_layernorm.weight # access the ln_2 weights of a Mistral model
                cloned_weights = weights.clone()
                cloned_weights[neuron_index] = 0
                weights.data.copy_(cloned_weights)
            else:
                weights = self.model.transformer.h[layer_id - 1].ln_2.weight.data
                weights[neuron_index] = torch.zeros_like(weights[neuron_index])
                weights.requires_grad = False


        logging.info('Model pruning completed. %d neurons ablated.', max_no_neurons_to_prune)
        self.model.save_pretrained(self.config['neural_probing']['pruned_model_path'])
        self.tokenizer.save_pretrained(self.config['neural_probing']['pruned_model_path'])
    
    def load_datasets(self, train_dataset_path, val_dataset_path):
        """
        Load training and validation datasets for retraining.

        Args:
            train_dataset_path (str): The path to the training dataset.
            val_dataset_path (str): The path to the validation dataset.

        Returns:
            DatasetDict: A dictionary containing the training and validation datasets.

        Raises:
            Exception: If there is an error loading the datasets.
        """
        logging.info('Loading training dataset from path: %s', train_dataset_path)
        try:
            train_dataset = load_from_disk(train_dataset_path)
            logging.info('Successfully loaded training dataset with %d samples.', len(train_dataset))
        except Exception as e:
            logging.error('Error loading training dataset: %s', e)
            raise

        logging.info('Loading validation dataset from path: %s', val_dataset_path)
        try:
            val_dataset = load_from_disk(val_dataset_path)
            logging.info('Successfully loaded validation dataset with %d samples.', len(val_dataset))
        except Exception as e:
            logging.error('Error loading validation dataset: %s', e)
            raise

        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
    
    def tokenize_dataset(self, max_length=1024):
        """
        Tokenize the retraining dataset.

        Args:
            max_length (int, optional): The maximum length for tokenization. Defaults to 1024.
        """
        logging.info('Tokenizing datasets...')

        # Define the tokenization function
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],  # Use the 'text' column for tokenization
                padding=False,  # Do not pad here; the DataCollator will handle it
                truncation=True,  # Enable truncation
                max_length=max_length  # Set max_length to prevent exceeding model limits
            )

        # Apply the tokenization function to the entire dataset
        self.tokenized_dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            batch_size=10,
            remove_columns=self.dataset['train'].column_names  # Remove original columns
        )

        logging.info('Tokenization completed successfully.')

    def retrain(self):
        """
        Retrain the pruned model using the specified training and validation datasets.

        This method initializes the Trainer, sets up the training arguments, 
        and performs the training process.
        """
        torch.cuda.empty_cache()
        logging.info('Retraining pruned model from path: %s', self.config['neural_probing']['pruned_model_path'])
        if self.tokenizer is None:
            self.model, self.tokenizer = load_model(self.config['neural_probing']['pruned_model_path'], self.config['base_model']['model_type'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.resize_token_embeddings(len(self.tokenizer))
        logging.info('Pad token added and model token embeddings resized.')

        self.dataset = self.load_datasets(self.config['retraining']['train_dataset_path'], self.config['retraining']['val_dataset_path'])
        self.tokenize_dataset()
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        args = TrainingArguments(
            self.config['retraining']['retrained_model_path'],
            evaluation_strategy='epoch',
            save_strategy='epoch',
            learning_rate=2e-5,
            num_train_epochs=self.config['retraining']['num_train_epochs'],
            weight_decay=self.config['retraining']['weight_decay'],
            per_device_train_batch_size=self.config['retraining']['batch_size'],
            per_device_eval_batch_size=self.config['retraining']['batch_size']
        )

        logging.info('Initializing Trainer...')
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=args,
            data_collator=self.data_collator,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['validation'],
        )
        logging.info('Trainer initialized successfully.')

        trainer.train()
        logging.info('Retraining completed successfully.')

        # Copy the final checkpoint to the parent directory
        final_checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-*')  # Pattern for the checkpoint directory
        final_checkpoint_path = max(glob.glob(final_checkpoint_dir), key=os.path.getctime)  # Get the latest checkpoint directory

        # Copy contents of the final checkpoint to the parent directory
        for item in os.listdir(final_checkpoint_path):
            s = os.path.join(final_checkpoint_path, item)
            d = os.path.join(self.config['retraining']['retrained_model_path'], item)
            if os.path.isdir(s):
                shutil.copytree(s, d)  # Copy directory
                logging.info(f'Copied directory {s} to {d}')
            else:
                shutil.copy2(s, d)  # Copy file
                logging.info(f'Copied file {s} to {d}')


        if self.config['retraining']['push_to_hub']:
            model, tokenizer = load_model(self.config['retraining']['retrained_model_path'], 
                                          self.config['base_model']['model_type'])
            model.push_to_hub(os.path.basename(self.config['retraining']['retrained_model_path']), 
                              token=self.config['retraining']['hf_token'], 
                              max_shard_size="5GB", 
                              safe_serialization=True)
            logging.info(f"Model '{os.path.basename(self.config['retraining']['retrained_model_path'])}' pushed to the Hugging Face Hub successfully.")
