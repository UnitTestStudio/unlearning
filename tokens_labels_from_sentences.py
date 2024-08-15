import json
import logging
import sys

def setup_logging():
    """Set up logging to output messages to the console."""
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the logging level

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Set the level for the console handler

    # Create a formatter and set it for the console handler
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

def load_json_file(file_path):
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                logging.info(f"Loaded JSON file: {file_path} with {len(data)} items.")
                return data  # Return the list if the data is a list
            else:
                logging.warning(f"The file {file_path} does not contain a list.")
                return []
    except FileNotFoundError:
        logging.error(f"The file {file_path} was not found.")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from file {file_path}: {e}")
        return []

def load_text_file(target_file):
    """Load target words from a specified text file."""
    try:
        logging.info(f"Loading target words from file: {target_file}")
        with open(target_file, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]  # Remove trailing newlines
            logging.info(f"Loaded {len(lines)} target words from {target_file}")
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        sys.exit(1)

    return lines

def filter_lists(sentences, target_words):
    """Filter the tokens based on target words and return their indices in the word lists."""
    filtered_sentences = []  # List to hold filtered sentences
    labels = []  # List to hold labels of target words in the current sentence

    # Iterate over each sentence and its index
    for i, sentence in enumerate(sentences):
        logging.debug(f"Processing sentence {i + 1}/{len(sentences)}: '{sentence}'")
        
        # Split the sentence into words
        words = sentence.split(" ")

        # Check for each target word in the list of words
        for keyword in target_words:
            if keyword in words:
                # Get the index of the first occurrence of the keyword
                index = words.index(keyword)
                label_line = ["N/A"] * len(words)
                label_line[index] = "target"
                labels.append(" ".join(label_line))

                filtered_sentences.append(sentence)
                logging.info(f"Found target word '{keyword}' in sentence {i + 1} at index {index}.")

    # Log the number of filtered sentences
    logging.info(f"Filtered {len(filtered_sentences)} sentences from {len(sentences)} total sentences.")

    return filtered_sentences, labels

def save_output(tokens, labels):
    """Save tokens and labels to their respective output files."""
    token_output_file = 'tcn-tokens.txt'
    label_output_file = 'tcn-labels.txt'

    if tokens and labels:
        with open(token_output_file, 'w') as f:
            f.write('\n'.join(tokens) + '\n')
        logging.info(f"Tokens file created at: {os.path.abspath(token_output_file)}")

        with open(label_output_file, 'w') as f:
            f.write('\n'.join(labels) + '\n')
        logging.info(f"Labels file created at: {os.path.abspath(label_output_file)}")

        logging.info(f"Saved output files")
    else:
        logging.warning(f"No data to save")

if __name__ == "__main__":
    setup_logging()  # Initialize logging