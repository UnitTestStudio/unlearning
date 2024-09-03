from transformers import GPTNeoForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk, DatasetDict, load_dataset
import logging

# Create a logger for this module
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, train_dataset_path, val_dataset_path):
        logger.info("Initializing ModelTrainer...")
        
        # Define the tokenizer as an instance variable
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load the datasets
        self.dataset = self.load_datasets(train_dataset_path, val_dataset_path)
        print(self.dataset)

        # Tokenize the dataset 
        self.tokenized_dataset = self.tokenize_dataset()
        print(self.tokenized_dataset)

    
        logger.info("ModelTrainer initialized successfully.")

        # Initialize the data collator
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Set to False for causal language modeling
        )

    def load_datasets(self, train_dataset_path, val_dataset_path):
        """
        Load and return the training and validation datasets.

        Args:
            train_dataset_path (str): Path to the training dataset.
            val_dataset_path (str): Path to the validation dataset.

        Returns:
            DatasetDict: A dictionary containing 'train' and 'validation' datasets.
        """
        logger.info("Loading training dataset from path: %s", train_dataset_path)
        try:
            train_dataset = load_from_disk(train_dataset_path)
            logger.info("Successfully loaded training dataset with %d samples.", len(train_dataset))
        except Exception as e:
            logger.error("Error loading training dataset: %s", e)
            raise

        logger.info("Loading validation dataset from path: %s", val_dataset_path)
        try:
            val_dataset = load_from_disk(val_dataset_path)
            logger.info("Successfully loaded validation dataset with %d samples.", len(val_dataset))
        except Exception as e:
            logger.error("Error loading validation dataset: %s", e)
            raise

        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })

    def tokenize_dataset(self, max_length=1024):
        """
        Tokenizes the loaded datasets.

        Args:
            max_length (int, optional): Maximum length of the tokenized sequences. 
                                        If None, uses the default max length of the tokenizer.

        Returns:
            DatasetDict: A dictionary containing 'train' and 'validation' datasets with 'input_ids' and 'attention_mask' tensors.
        """
        logger.info("Tokenizing datasets...")

        # Define the tokenization function
        def tokenize_function(examples):
            return self.tokenizer(
            examples["text"],  # Use the 'text' column for tokenization
            padding=False,  # Do not pad here; the DataCollator will handle it
            truncation=True,  # Enable truncation
            max_length=max_length  # Set max_length to prevent exceeding model limits
            )

        # Apply the tokenization function to the entire dataset
        tokenized_dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            batch_size=10,
            remove_columns=self.dataset["train"].column_names  # Remove original columns
        )

        logger.info("Tokenization completed successfully.")
        return tokenized_dataset

    def retrain_pruned_model(self, pruned_model_path):
        logger.info("Retraining pruned model from path: %s", pruned_model_path)
        try:
            model = GPTNeoForCausalLM.from_pretrained(pruned_model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error("Error loading model: %s", e)
            raise

        model.resize_token_embeddings(len(self.tokenizer))
        logger.info("Pad token added and model token embeddings resized.")

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

        logger.info("Initializing Trainer...")
        trainer = Trainer(
            model=model,
            tokenizer=self.tokenizer,
            args=args,
            data_collator=self.data_collator,  # Use the data collator here
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"],
        )
        logger.info("Trainer initialized successfully.")

        logger.info("Starting retraining...")
        trainer.train()
        logger.info("Retraining completed successfully.")
        return model
