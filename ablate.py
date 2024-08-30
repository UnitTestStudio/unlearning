import sys
import os
import logging
import torch
import json
import datetime as d
sys.path.append("..")
from src.models.ModelTrainer import ModelTrainer
from src.visualization.ModelAnalyzer import ModelAnalyzer
from src.models.prune_model import prune_model
from src import ACTIVATIONS_PATH, MODEL_PATH, NUMBER_TO_PRUNE, PRUNED_MODEL_PATH, MODEL_NAME
from transformers import GPT2LMHeadModel, GPTNeoForCausalLM, GPT2Tokenizer

def setup_logging():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('ablate.log', mode='a')

    # Set levels for handlers
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

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

    logging.info(f"Logging warnings and errors to file: {os.path.abspath('ablate.log')}")

def generate_text_with_prompt(model, prompt, temperature=0.7, seed=42):
    # Set the seed for reproducibility
    # torch.manual_seed(seed)
    

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    generation_params = {
        "max_length": 100,
        "num_return_sequences": 1,
        "do_sample": True,
        "temperature": temperature,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # Generate text with the specified prompt and temperature
    with torch.no_grad():
        output = model.generate(input_ids, **generation_params, attention_mask=input_ids.ne(0))
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    logging.info(f"Generated Text: {generated_text}.")

if __name__ == "__main__":
    setup_logging()  # Initialize logging

    # Check if the pruned model file exists
    if os.path.exists(PRUNED_MODEL_PATH):
        # If the file exists, load the pruned model
        # model_trainer = ModelTrainer()
        # pruned_model = get_pruned_model(PRUNED_MODEL_PATH, model_trainer)
        logging.info(f"Pruned model exists at {PRUNED_MODEL_PATH}.")
    else:
        # Ablate neurons
        basic_analyser = ModelAnalyzer(MODEL_PATH, ACTIVATIONS_PATH)
        neurons_to_prune = basic_analyser.identify_concept_neurons()
        pruned_model = prune_model(MODEL_PATH, neurons_to_ablate=neurons_to_prune[-NUMBER_TO_PRUNE:])
        logging.info(f"Saving pruned model to file: {os.path.abspath(PRUNED_MODEL_PATH)}")
        pruned_model.save_pretrained(PRUNED_MODEL_PATH)
        
    if MODEL_NAME == "gpt-neo-1.3B":
        model = GPTNeoForCausalLM.from_pretrained(PRUNED_MODEL_PATH)
        logging.info("Loaded GPT-NEO model successfully.")
    else:
        model = GPT2LMHeadModel.from_pretrained(PRUNED_MODEL_PATH)
        logging.info("Loaded GPT-2 model successfully.")


    # Load configuration from config.json
    with open('run-config.json', 'r') as config_file:
        config = json.load(config_file)

    for prompt in config["test_prompts"]:
        logging.info(f"Prompt: {prompt}")
        generated_text = generate_text_with_prompt(model, prompt, 0.6)