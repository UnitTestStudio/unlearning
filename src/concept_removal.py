import torch
import torch.nn as nn
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer  # Changed to LLAMA-specific classes
import logging

def identify_concept_neurons(model, tokenizer, concept_examples, layer_nums, prune_percentage=0.01, device='cuda'):
    """
    Identify neurons that are highly activated by concept-related inputs.
    
    Args:
        model: The pre-trained LLAMA model
        tokenizer: Associated LLAMA tokenizer
        concept_examples: List of strings containing examples of the concept
        layer_nums: List of transformer layer numbers to analyze
        prune_percentage: Percentage of top neurons to prune (default: 1%)
        device: Device to run the model on (default: 'cuda')
    
    Returns:
        Dictionary mapping layer numbers to lists of neuron indices
    """
    concept_neurons = {}
    
    inputs = tokenizer(concept_examples, return_tensors="pt", padding=True).to(device)
    
    # Changed to access LLAMA's layers
    for layer_num in layer_nums:
        layer = model.model.layers[layer_num]  # Modified layer access for LLAMA
        
        activations = []
        def hook_fn(module, input, output):
            activations.append(output.detach())
        
        # Hook is now attached to LLAMA's MLP output
        handle = layer.mlp.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        layer_activations = torch.cat(activations)
        mean_activations = layer_activations.mean(dim=(0,1))
        
        # Determine the top neurons based on the specified prune percentage
        top_neurons = torch.topk(mean_activations, k=int(len(mean_activations) * prune_percentage)).indices
        concept_neurons[layer_num] = top_neurons.tolist()
        
        handle.remove()
    
    return concept_neurons

def prune_neurons(model, concept_neurons):
    """
    Prune identified neurons by zeroing out their weights in LLAMA's MLP structure.
    
    Args:
        model: The pre-trained LLAMA model
        concept_neurons: Dictionary mapping layer numbers to neuron indices
    """
    for layer_num, neurons in concept_neurons.items():
        layer = model.model.layers[layer_num]  # Modified for LLAMA's layer structure
        
        # Zero out weights in LLAMA's MLP structure
        layer.mlp.gate_proj.weight.data[neurons, :] = 0
        layer.mlp.up_proj.weight.data[neurons, :] = 0
        layer.mlp.down_proj.weight.data[:, neurons] = 0

def evaluate_concept_removal(model, tokenizer, test_prompts, device='cuda'):
    """
    Evaluate if the concept has been successfully removed.
    
    Args:
        model: The pruned LLAMA model
        tokenizer: Associated LLAMA tokenizer
        test_prompts: List of prompts to test concept understanding
        device: Device to run the model on (default: 'cuda')
    
    Returns:
        Float: Score indicating concept presence (lower is better)
    """
    scores = []
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=inputs['input_ids'].shape[1] + 50,  # Adjust max_length as needed
                num_beams=5,  # Number of beams for beam search
                early_stopping=True
            )

        # Decode the generated output back into text
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Log the natural language output from the model
        logging.info(f"Model output for prompt '{prompt}': {decoded_output}")
        
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        
        score = probs.max().item()
        scores.append(score)
    
    return np.mean(scores)