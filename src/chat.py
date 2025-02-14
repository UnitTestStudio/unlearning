from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import logging

# Get logger
logger = logging.getLogger()

def generate_responses(test_prompts, model, tokenizer,  max_length, num_return_sequences, temperature):
    # Set pad_token_id to eos_token_id if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Initialize the pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Loop through the test prompts and generate responses
    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        output = pipe(messages, max_length=max_length, num_return_sequences=num_return_sequences, temperature=temperature)    
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Output: {output[0]['generated_text'][1]['content']}\n")