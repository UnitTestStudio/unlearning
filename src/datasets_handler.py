from datasets import load_dataset
import os
import logging
import sys

# Create a logger for this module
logging = logging.getLogger(__name__)

def load_text_file(target_file):
    """Load target words from a specified text file."""
    try:
        logging.info(f"Loading target words from file: {target_file}")
        with open(target_file, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip()]  # Remove trailing newlines and empty lines
            logging.info(f"Loaded {len(lines)} target words from {target_file}")
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading the file: {e}")
        raise

    return lines

def load_hf_dataset(dataset_name):
        """Load a dataset from Hugging Face."""
        try:
            logging.info(f"Loading the dataset: {dataset_name}...")
            dataset = load_dataset(dataset_name)
            logging.info("Dataset loaded successfully.")
            return dataset
        except Exception as e:
            logging.error(f"Failed to load dataset: {e}")
            raise

class Datasets:
    def __init__(self, config):
        self.hugging_face_dataset = load_hf_dataset(config['datasets']['dataset_path'])
        self.target_words = load_text_file(config['datasets']['concept_definition'])

    # def generate_labels(self, sentences, target_words):
    #     """Filter the tokens based on target words and return their indices in the word lists."""
    #     ### TURNS OUT NOT EVERY DATASET IS AS GOOD FOR ACTIVATION PROBING BASED ON TARGET CONCEPT
    #     # DONT USE THIS FOR NOW

    #     filtered_sentences = []  # List to hold filtered sentences
    #     labels = []  # List to hold labels of target words in the current sentence
    #     inverse_sentences = []  # List to hold sentences without any target words

    #     # Iterate over each sentence and its index
    #     for i, sentence in enumerate(sentences):
    #         # logging.debug(f"Processing sentence {i + 1}/{len(sentences)}: '{sentence}'")
            
    #         # Split the sentence into words
    #         words = sentence.split(" ")

    #         # Check for any target word in the list of words
    #         found_target = False
    #         for keyword in target_words:
    #             if keyword in words:
    #                 found_target = True
    #                 # Get the index of the first occurrence of the keyword
    #                 index = words.index(keyword)
    #                 label_line = ["N/A"] * len(words)
    #                 label_line[index] = "target"
    #                 labels.append(" ".join(label_line))

    #                 filtered_sentences.append(sentence)
    #                 logging.debug(f"Found target word '{keyword}' in sentence {i + 1} at index {index}.")
    #                 break  # No need to check other keywords if one is found

    #         # If no target word was found, add to inverse sentences
    #         if not found_target:
    #             inverse_sentences.append(sentence)

    #     # Log the number of filtered sentences
    #     logging.info(f"Found {len(filtered_sentences)} sentences with target words from {len(sentences)} total sentences.")

    #     return filtered_sentences, labels

    # def make_activations_dataset(self):
    #     ### TURNS OUT NOT EVERY DATASET IS AS GOOD FOR ACTIVATION PROBING BASED ON TARGET CONCEPT
    #     # DONT USE THIS FOR NOW

    #     logging.info("Creating activations dataset...")
    #     all_sentences = self.hugging_face_dataset['train']['text'] + self.hugging_face_dataset['validation']['text']
    #     logging.debug(len(all_sentences))

    #     filtered_sentences, labels = self.generate_labels(all_sentences, self.target_words)
 
    #     try:
    #         with open('data/tokens.txt', 'w') as f:
    #             f.write('\n'.join(filtered_sentences) + '\n')
    #         logging.info(f"Tokens file created at: {os.path.abspath('data/tokens.txt')}")
    #     except Exception as e:
    #         logging.error(f"Error saving activations sentences: {e}")
    #         sys.exit(1)

    #     try:
    #         with open('data/labels.txt', 'w') as f:
    #                     f.write('\n'.join(labels) + '\n')
    #         logging.info(f"Labels file created at: {os.path.abspath('data/labels.txt')}")
    #     except Exception as e:
    #         logging.error(f"Error saving activations labels: {e}")
    #         sys.exit(1)
    
    def make_retrain_dataset(self):
        logging.info("Creating retraining dataset...")
        def contains_target_words(example):
            # Check if the 'text' field contains any of the target words
            return not any(word in example['text'].lower() for word in self.target_words)

        # Apply the filtering function to the dataset
        try:
            logging.info("Filtering the training dataset...")
            filtered_train = self.hugging_face_dataset['train'].filter(contains_target_words)
            logging.info("Filtering completed for training dataset.")

            logging.info("Filtering the validation dataset...")
            filtered_val = self.hugging_face_dataset['validation'].filter(contains_target_words)
            logging.info("Filtering completed for validation dataset.")
        except Exception as e:
            logging.error(f"Error during filtering: {e}")
            sys.exit(1)

        logging.info(f"Original train size: {len(self.hugging_face_dataset['train'])}, Filtered train size: {len(filtered_train)}")
        logging.info(f"Original validation size: {len(self.hugging_face_dataset['validation'])}, Filtered validation size: {len(filtered_val)}")

        try:
            filtered_train.save_to_disk("data/filtered_train")
            filtered_val.save_to_disk("data/filtered_val")
            logging.info("Filtered datasets saved successfully.")
        except Exception as e:
            logging.error(f"Error saving filtered datasets: {e}")
            sys.exit(1)