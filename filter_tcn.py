import json
import sys
import os
import logging
import datetime as d

# Logging
def setup_logging():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('warnings.log', mode='a')

    # Set levels for handlers
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.WARNING)

    # Create formatters and add to handlers
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_format = logging.Formatter('%(message)s')  # Simplified format for file
    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)

    # Filter to print INFO messages to console
    class InfoFilter(logging.Filter):
        def filter(self, record):
            return record.levelno == logging.INFO

    # Add filter to console handler
    console_handler.addFilter(InfoFilter())

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Add a timestamped heading to the log file
    timestamp = d.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_handler.emit(logging.LogRecord(
        name='', level=logging.WARNING,
        pathname='', lineno=0,
        msg=f"\n\n{'='*50}\nScript Run: {timestamp}\n{'='*50}\n",
        args=(), exc_info=None
    ))

    logging.info(f"Logging warnings and errors to file: {os.path.abspath('warnings.log')}")
# Constants for file paths
CLUSTER_FILE = 'data/external/albert-base-v1/layer0/clusters-600.txt'
SENTENCE_FILE = 'data/external/albert-base-v1/sentences.json'
ANNOTATIONS_FILE = 'data/external/albert-base-v1/layer0/annotations.json'

def load_clusters(cluster_file):
    """Load clusters from a specified file into a dictionary."""
    clusters = {}
    with open(cluster_file, 'r') as f:
        for line in f:
            parts = line.strip().split('|||')
            if len(parts) >= 5:
                word, frequency, sentence_id, word_id, cluster_id = parts
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append((word, sentence_id, word_id))
    logging.info(f"Loaded {len(clusters)} clusters")
    return clusters

def load_sentences(sentence_file):
    """Load sentences from a specified JSON file."""
    with open(sentence_file, 'r') as f:
        sentences = json.load(f)
    logging.info(f"Loaded {len(sentences)} sentences")
    return sentences

def load_definitions(definition_file):
    """Load concept definitions from a specified JSON file."""
    with open(definition_file, 'r') as f:
        definitions = json.load(f)
    logging.info(f"Loaded {len(definitions)} definitions")
    return definitions

def filter_dataset(concept_id, clusters, sentences):
    """Filter the dataset based on the provided concept ID."""
    tokens = []
    labels = []
    skipped_count = 0

    if concept_id in clusters:
        logging.info(f"Found concept_id {concept_id} in clusters")
        for word, sentence_id, word_id in clusters[concept_id]:
            sentence_index = int(sentence_id)
            if 0 <= sentence_index < len(sentences):
                sentence = sentences[sentence_index]
                words = sentence.split()
                word_index = int(word_id)

                if 0 <= word_index < len(words):
                    tokens.append(sentence)
                    label_line = ["N/A"] * len(words)
                    label_line[word_index] = concept_id
                    labels.append(" ".join(label_line))
                else:
                    skipped_count += 1
                    logging.debug(f"Skipped: Word_id {word_id} is out of range for sentence '{sentence}' with length {len(words)}.")
            else:
                skipped_count += 1
                logging.debug(f"Skipped: Sentence index {sentence_index} is out of range.")
    else:
        logging.warning(f"Concept_id {concept_id} not found in clusters")
    
    logging.info(f"Filtered {len(tokens)} tokens for concept_id {concept_id}")
    if skipped_count > 0:
        logging.warning(f"Skipped {skipped_count} entries due to out-of-range indices.")
    return tokens, labels

def save_output(tokens, labels, concept_id):
    """Save tokens and labels to their respective output files."""
    token_output_file = os.path.dirname(CLUSTER_FILE) + '/tcn-tokens-' + concept_id + '.txt'
    label_output_file = os.path.dirname(CLUSTER_FILE) + '/tcn-labels-' + concept_id + '.txt'

    if tokens and labels:
        with open(token_output_file, 'w') as f:
            f.write('\n'.join(tokens) + '\n')
        logging.info(f"Tokens file created at: {os.path.abspath(token_output_file)}")

        with open(label_output_file, 'w') as f:
            f.write('\n'.join(labels) + '\n')
        logging.info(f"Labels file created at: {os.path.abspath(label_output_file)}")

        logging.info(f"Saved output files for concept_id {concept_id}")
    else:
        logging.warning(f"No data to save for concept_id {concept_id}")

def main():
    """Main function to execute the script."""
    setup_logging()
    
    if len(sys.argv) != 2:
        logging.error("Usage: python3 filter_dataset.py <concept_id>")
        sys.exit(1)

    concept_id = sys.argv[1]
    logging.info(f"Processing concept_id: {concept_id}")
    
    # Load the definitions and check for the concept ID
    definitions = load_definitions(ANNOTATIONS_FILE)
    
    if concept_id in definitions:
        definition = definitions[concept_id]
        logging.info(f"Concept Definition for {concept_id}: {definition}")
    else:
        logging.error(f"Error: concept_id {concept_id} not found in definitions.")
        sys.exit(1)

    sentences = load_sentences(SENTENCE_FILE)
    clusters = load_clusters(CLUSTER_FILE)

    concept_id = concept_id[1:]
    logging.info(f"Processing concept_id without leading character: {concept_id}")
    tokens, labels = filter_dataset(concept_id, clusters, sentences)
    
    if tokens:
        save_output(tokens, labels, concept_id)
        logging.info(f"Filtered dataset for concept ID {concept_id} ({definition}) created.")
    else:
        logging.warning(f"No tokens found for concept_id {concept_id}")

if __name__ == "__main__":
    main()