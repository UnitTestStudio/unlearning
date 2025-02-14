import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import argparse

# Constants
concept_example_prompt = "Generate a sentence that includes the word 'red'. It could be a description or an extract of dialogue. Your response should only include the example sentence."
background_example_prompt = "Generate a sentence that does not include the word 'red'. Include a colour that is not 'red'. Your response should only include the example sentence."

# Load the tokenizer and model using the Transformers library.
# Adjust the model path or additional parameters as needed.
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set pad_token_id to eos_token_id if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Create a text generation pipeline from the loaded model and tokenizer.
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Initialize a set to track unique sentences
concept_set = set()
background_set = set()

def has_escape_character(text: str) -> bool:
    """
    Returns True if the text contains newline, carriage return, tab, or a backslash.
    """
    forbidden = ["\n", "\r", "\t", "\\"]
    return any(forbidden_char in text for forbidden_char in forbidden)

def generate_sentence(prompt: str,  max_length: int, num_return_sequences: int, temperature: float) -> str:
    """
    Use the Hugging Face generation pipeline to generate a sentence.
    You can adjust the generation parameters (e.g., max_length, temperature) as needed.
    """
    messages = [{"role": "user", "content": prompt}]
    outputs = generator(messages, max_length=max_length, num_return_sequences=num_return_sequences, temperature=temperature)
    # Extract the generated text from the output
    sentences = []
    for output in outputs:
        # Check if the 'generated_text' key is a list
        generated_part = output.get("generated_text")
        if isinstance(generated_part, list) and len(generated_part) > 0:
            # Assume that the assistant's answer is the last item
            message = generated_part[-1]
            # Ensure the assistant message has a 'content' key
            if isinstance(message, dict) and "content" in message:
                sentence = message["content"].strip()
                sentences.append(sentence)
            else:
                print("Unexpected format for generated message:", message)
    return sentences

def write_json(data, filename):
    """
    Write the current dataset JSON object to a file.
    """
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4)

def main(target_dataset_size: int, max_length: int, num_return_sequences: int, temperature: float, out_name: str):
    total_sentences = 0
    num_skipped = 0
    pbar = tqdm(total=target_dataset_size, desc="Generating concept examples")
    while len(concept_set) < target_dataset_size:
        # Generate a sentence using the prompt
        sentences = generate_sentence(concept_example_prompt, max_length, num_return_sequences, temperature)
        total_sentences += len(sentences)
        
        # Check if the sentence contains the word "red" (case-insensitive check)
        for sentence in sentences:
            if " red " in sentence.lower():
                sentence = sentence.replace('\"', '')
                if has_escape_character(sentence):
                    num_skipped += 1
                if sentence not in concept_set:
                    concept_set.add(sentence)
                    pbar.update(1)
                    # Build the JSON object and write it to file after each addition
                    dataset_json = {"concept_examples": list(concept_set)}
                    write_json(dataset_json, out_name)
                else:
                    num_skipped += 1
            else:
                num_skipped += 1

    pbar.close()
    print("Concept examples generated!")
    print(f"Total unique sentences: {len(concept_set)}")
    print(f"Pecentage skipped: {num_skipped / total_sentences:.2%}")

    total_sentences = 0
    num_skipped = 0
    pbar = tqdm(total=target_dataset_size, desc="Generating background examples")
    while len(background_set) < target_dataset_size:
        # Generate a sentence using the prompt
        sentences = generate_sentence(background_example_prompt, max_length, num_return_sequences, temperature)
        total_sentences += len(sentences)
        
        # Check if the sentence contains the word "red" (case-insensitive check)
        for sentence in sentences:
            if " red " not in sentence.lower():
                sentence = sentence.replace('\"', '')
                if has_escape_character(sentence):
                    num_skipped += 1
                if sentence not in background_set:
                    background_set.add(sentence)
                    pbar.update(1)
                    # Build the JSON object and write it to file after each addition
                    dataset_json = {"concept_examples": list(concept_set),
                                    "background_examples": list(background_set)}
                    write_json(dataset_json, out_name)
                else:
                    num_skipped += 1
            else:
                num_skipped += 1

    pbar.close()
    print("Background examples generated!")
    print(f"Total unique sentences: {len(background_set)}")
    print(f"Pecentage skipped: {num_skipped / total_sentences:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dataset_size", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=70)
    parser.add_argument("--num_return_sequences", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=2.75)
    parser.add_argument("--out_name", type=str, default="data.json")
    args = parser.parse_args()

    main(
        args.target_dataset_size,
        args.max_length,
        args.num_return_sequences,
        args.temperature,
        args.out_name
    )