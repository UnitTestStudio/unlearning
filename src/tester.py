from src.model_handler import load_model
from transformers import GPT2Tokenizer
import logging
import torch

# Create a logger for this module
logging = logging.getLogger(__name__)

def generate_text_with_prompt(model, tokenizer, prompt, temperature=0.7, seed=42):
    # Set the seed for reproducibility
    # torch.manual_seed(seed)
    
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

def test_model(config, stage):
    if stage == "base":
        model_path = config['base_model']['base_model_path']
    if stage == "pruned":
        model_path = config['neural_probing']['pruned_model_path']
    if stage == "retrained":
        model_path = config['retraining']['retrained_model_path']
    
    model, tokenizer = load_model(model_path, config['base_model']['model_type'])
    logging.info(f"Testing model with prompts: {config['test_prompts']}")
    for prompt in config['test_prompts']:
        logging.info(f"Prompt: {prompt}")
        generate_text_with_prompt(model, tokenizer, prompt, 0.6)