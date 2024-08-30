from transformers import GPTNeoForCausalLM, GPT2Tokenizer, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from src import tokenizer, RETRAIN_DATASET_PATH
import logging

# Create a logger for this module
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        logger.info("Initializing ModelTrainer...")
        self.tokenized_dataset = self.tokenize_dataset()
        logger.info("ModelTrainer initialized successfully.")

    def tokenize_dataset(self, test_size=0.2, max_length=None):
        """
        Tokenizes sentences from a text file using the GPT-2 tokenizer and splits them into training and validation sets.

        Args:
            test_size (float): Proportion of the dataset to include in the validation split (default is 0.2).
            max_length (int, optional): Maximum length of the tokenized sequences. 
                                        If None, uses the default max length of the tokenizer.

        Returns:
            dict: A dictionary containing 'train' and 'validation' datasets with 'input_ids' and 'attention_mask' tensors.
        """
        logger.info("Loading sentences from dataset...")
        try:
            with open(RETRAIN_DATASET_PATH, 'r', encoding='utf-8') as file:
                sentences = file.readlines()
            logger.info("Successfully loaded %d sentences.", len(sentences))
        except Exception as e:
            logger.error("Error loading dataset: %s", e)
            raise

        # Remove any leading/trailing whitespace characters (like newlines)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        logger.info("Filtered sentences. Remaining count: %d", len(sentences))

        # Step 2: Split the dataset into training and validation sets
        logger.info("Splitting dataset into training and validation sets...")
        train_sentences, val_sentences = train_test_split(sentences, test_size=test_size, random_state=42)
        logger.info("Split completed. Training set size: %d, Validation set size: %d", len(train_sentences), len(val_sentences))

        if tokenizer.pad_token is None:
            logger.info("Adding pad token to tokenizer...")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
        # Step 4: Tokenize the training and validation sentences
        logger.info("Tokenizing training sentences...")
        tokenized_train = tokenizer(
            train_sentences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'  # Return as PyTorch tensors
        )
        logger.info("Tokenization of training sentences completed. Tokenized count: %d", tokenized_train['input_ids'].size(0))

        logger.info("Tokenizing validation sentences...")
        tokenized_val = tokenizer(
            val_sentences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'  # Return as PyTorch tensors
        )
        logger.info("Tokenization of validation sentences completed. Tokenized count: %d", tokenized_val['input_ids'].size(0))

        # Step 5: Create a dataset dictionary
        tokenized_dataset = {
            'train': {
                'input_ids': tokenized_train['input_ids'],
                'attention_mask': tokenized_train['attention_mask']
            },
            'validation': {
                'input_ids': tokenized_val['input_ids'],
                'attention_mask': tokenized_val['attention_mask']
            }
        }

        logger.info("Tokenized dataset created successfully.")
        return tokenized_dataset

    def retrain_pruned_model(self, pruned_model_path):
        logger.info("Retraining pruned model from path: %s", pruned_model_path)
        try:
            model = GPTNeoForCausalLM.from_pretrained(pruned_model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error("Error loading model: %s", e)
            raise

        model.resize_token_embeddings(len(tokenizer))
        logger.info("Pad token added and model token embeddings resized.")

        args = TrainingArguments(
            "retrained_model",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=12,
            weight_decay=0.01,
        )

        logger.info("Initializing Trainer...")
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"],
            tokenizer=tokenizer,
        )
        logger.info("Trainer initialized successfully.")

        logger.info("Starting retraining...")
        trainer.train()
        logger.info("Retraining completed successfully.")
        return model
