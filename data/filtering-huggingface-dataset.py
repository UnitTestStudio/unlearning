from datasets import load_dataset
import logging
import sys

def setup_logging():
    """Set up logging to output messages to the console."""
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the logging level

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set the level for the console handler

    # Create a formatter and set it for the console handler
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

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
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred while loading the file: {e}")
        sys.exit(1)

    return lines

# Step 3: Load the OpenAssistant dataset
try:
    logging.info("Loading the OpenAssistant dataset...")
    dataset = load_dataset("OpenAssistant/oasst2")
    logging.info("Dataset loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load dataset: {e}")
    sys.exit(1)

# Step 4: Define the list of target words
target_words = load_text_file("data/target_words.txt")

# Step 5: Define a function to filter out messages containing any target words
def contains_target_words(example):
    # Check if the 'text' field contains any of the target words
    return not any(word in example['text'].lower() for word in target_words)

# Step 6: Apply the filtering function to the dataset
try:
    logging.info("Filtering the training dataset...")
    filtered_train = dataset['train'].filter(contains_target_words)
    logging.info("Filtering completed for training dataset.")

    logging.info("Filtering the validation dataset...")
    filtered_val = dataset['validation'].filter(contains_target_words)
    logging.info("Filtering completed for validation dataset.")
except Exception as e:
    logging.error(f"Error during filtering: {e}")
    sys.exit(1)

# Step 7: Use or save the modified dataset
original_train_size = len(dataset['train'])
original_val_size = len(dataset['validation'])
filtered_train_size = len(filtered_train)
filtered_val_size = len(filtered_val)

logging.info(f"Original train size: {original_train_size}, Filtered train size: {filtered_train_size}")
logging.info(f"Original validation size: {original_val_size}, Filtered validation size: {filtered_val_size}")

# Optionally, save the modified datasets
try:
    filtered_train.save_to_disk("data/filtered_train_oasst2")
    filtered_val.save_to_disk("data/filtered_val_oasst2")
    logging.info("Filtered datasets saved successfully.")
except Exception as e:
    logging.error(f"Error saving filtered datasets: {e}")
    sys.exit(1)
