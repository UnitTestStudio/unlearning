import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import logging

# Get logger
logger = logging.getLogger()
        

class ConceptNeuronSaliencyAnalyzer:
    def __init__(self, model: nn.Module, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def total_neurons(self, total_salient_neurons):
        total_neurons_in_model = 0
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None and len(module.weight.shape) == 2:
                total_neurons_in_model += module.weight.shape[0]

        if total_neurons_in_model > 0:
            percentage = 100.0 * total_salient_neurons / total_neurons_in_model
        else:
            percentage = 0.0

        logger.info(f"Total salient neurons: {total_salient_neurons} "
                    f"({percentage:.2f}% of {total_neurons_in_model} total neurons)")

    def extract_layer_activations(
        self, 
        texts: List[str], 
        layer_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Extract mean layer activations for input texts.
        
        Args:
            texts: List of input texts
            layer_names: Optional list of specific layers to extract
        
        Returns:
            Dictionary of layer activations
        """
        # Store layer-wise results
        layer_activations = {}
        
        # Create hooks for specified layers
        hooks = {}
        layer_acts = {layer: [] for layer in layer_names}
        
        def create_hook(layer_name):
            def hook_fn(module, input, output):
                # Capture mean activations across sequence
                if isinstance(output, tuple):
                    output = output[0]
                layer_acts[layer_name].append(output.mean(dim=1).detach())
            return hook_fn
        
        # Register hooks
        hook_handles = []
        for layer_name in layer_names:
            for name, module in self.model.named_modules():
                if name == layer_name:
                    hook = create_hook(layer_name)
                    handle = module.register_forward_hook(hook)
                    hook_handles.append(handle)
                    break
        
        # Process texts
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                _ = self.model(**inputs)
        
        # Remove hooks
        for handle in hook_handles:
            handle.remove()
        
        # Convert to numpy arrays
        for layer_name in layer_names:
            if layer_acts[layer_name]:
                # Concatenate and compute mean across texts
                layer_activations[layer_name] = torch.cat(layer_acts[layer_name]).cpu().numpy()
        
        # Print the activations for each layer
        for layer_name, activation in layer_activations.items():
            logger.debug(f"{layer_name}: {activation.shape}")

        return layer_activations

    def extract_activations(
            self,
            concept_texts: List[str],
            background_texts: List[str],
            num_layers: int = 10,
            output_path: Optional[str] = "data/activations.npz"):

        # Print model structure for debugging
        logger.debug(f"Model type: {self.model.__class__.__name__.lower()}")

        layer_names = [
                name for name, module in self.model.named_modules()
                if 'layers' in name and isinstance(module, nn.Module) and 
                any(isinstance(submodule, nn.Linear) 
                    for submodule in module.children())
                    ]
        
        # Select the top `num_layers` layers
        layer_names = layer_names[-num_layers:]
        logger.info(f"Selected top {num_layers} layers: {layer_names}")
        # Extract activations
        logger.info("Extracting layer activations for concept texts")
        concept_activations = self.extract_layer_activations(concept_texts, layer_names)
        logger.info("Extracting layer activations for background texts")
        background_activations = self.extract_layer_activations(background_texts, layer_names)

        # Save activations to file
        np.savez(output_path, concept=concept_activations, background=background_activations)
        logger.info(f"Activations saved to {output_path}")
    
    def analyze_concept_saliency(
            self,
            activations_path: str,
            top_k: int = 10,
            regularisation_strength: float = 10.0,
            statistical_test: bool = True
        ) -> Dict[str, List[Tuple[int, float]]]:
        """
        Analyze neuron saliency for a specific concept across layers.
        
        Args:
            activations_path: Path to the saved activations file
            top_k: Number of top neurons to return for each layer
            regularisation_strength: Regularisation strength for logistic regression
            statistical_test: Whether to apply statistical significance test
        
        Returns:
            Dictionary of submodule names to their top salient neurons
        """
        
        # Analyze saliency for each layer
        submodule_saliency = {}
        logger.info("Analyzing saliency for each layer...")

        activations = np.load(activations_path, allow_pickle=True)
        concept_activations = activations["concept"].item()
        background_activations = activations["background"].item()
        
        for layer_name in concept_activations:
            
            # Get counts from activations
            concept_count = concept_activations[layer_name].shape[0]
            background_count = background_activations[layer_name].shape[0]
            
            # Prepare data for logistic regression
            X = np.vstack([
            concept_activations[layer_name], 
            background_activations[layer_name]
            ])
            y = np.concatenate([
            np.ones(concept_count), 
            np.zeros(background_count)
            ])
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Logistic regression with L1 regularization
            lr = LogisticRegression(
                penalty='l1',
                solver='liblinear',
                max_iter=1000,
                class_weight='balanced',
                C = regularisation_strength
            )
            lr.fit(X_scaled, y)
            
            # Get neuron importances
            neuron_importances = np.abs(lr.coef_[0])
            
            # Optional statistical significance testing
            if statistical_test:
                # Perform t-test between concept and background activations
                pvalues = []
                for i in range(X.shape[1]):
                    t_stat, p_val = stats.ttest_ind(
                        concept_activations[layer_name][:, i],
                        background_activations[layer_name][:, i]
                    )
                    pvalues.append(p_val)
                
                # Adjust importances based on statistical significance
                neuron_importances *= -np.log10(pvalues)
            
            # Get top-k neurons
            top_neurons = sorted(
                enumerate(neuron_importances), 
                key=lambda x: x[1], 
                reverse=True
            )[:top_k]

            # Filter neurons with scores greater than zero
            filtered_neurons = [(idx, importance) for idx, importance in top_neurons if importance > 0]
            
            # Calculate additional statistics
            if filtered_neurons:
                highest_score = max(importance for _, importance in filtered_neurons)
                average_score = sum(importance for _, importance in filtered_neurons) / len(filtered_neurons)
            else:
                highest_score = 0
                average_score = 0
            
            logger.info(f"Layer {layer_name} has {len(filtered_neurons)} neurons with scores greater than zero.")
            logger.info(f"Highest score in layer {layer_name}: {highest_score}")
            logger.info(f"Average score in layer {layer_name}: {average_score}")

            # Store saliency results with submodule names
            for idx, importance in filtered_neurons:
                # Access the layer module
                layer_module = dict(self.model.named_modules())[layer_name]
                
                # Get the hidden size from the layer module
                if 'self_attn' in layer_name:
                    hidden_size = layer_module.q_proj.weight.size(0)  # Assuming q_proj defines the hidden size
                elif 'mlp' in layer_name:
                    hidden_size = layer_module.up_proj.weight.size(0)  # Assuming up_proj defines the hidden size
                else:
                    logger.debug(f"Layer {layer_name} does not have self_attn or mlp attributes.")
                    continue  # Skip this layer if it doesn't fit expected structure

                if 'self_attn' in layer_name:
                    # Determine the correct projection based on the index
                    if idx < hidden_size:
                        submodule_name = f"{layer_name}.q_proj"
                        logger.debug(f"Neuron index {idx} corresponds to q_proj")
                    elif idx < 2 * hidden_size:
                        submodule_name = f"{layer_name}.k_proj"
                        logger.debug(f"Neuron index {idx} corresponds to k_proj")
                    else:
                        submodule_name = f"{layer_name}.v_proj"
                        logger.debug(f"Neuron index {idx} corresponds to v_proj")
                elif 'mlp' in layer_name:
                    # Assuming that idx corresponds to the up_proj and down_proj
                    if idx < hidden_size:
                        submodule_name = f"{layer_name}.up_proj"
                        logger.debug(f"Neuron index {idx} corresponds to up_proj")
                    else:
                        submodule_name = f"{layer_name}.down_proj"
                        logger.debug(f"Neuron index {idx} corresponds to down_proj")

                # Append the neuron index and importance to the corresponding submodule
                submodule_saliency.setdefault(submodule_name, []).append((idx, float(importance)))

        total_salient_neurons = sum(len(neurons) for neurons in submodule_saliency.values())
        self.total_neurons(total_salient_neurons)

        return submodule_saliency

    def zero_out_neurons(self, submodule_saliency: Dict[str, List[Tuple[int, float]]]):
        """
        Zero out the weights and biases of the neurons specified in the saliency results.
        
        Args:
            submodule_saliency: Dictionary mapping submodule names to lists of (neuron_index, importance)
        """
        for submodule_name, neurons in submodule_saliency.items():
            # Access the corresponding submodule in the model to zero out the weights and biases of the specified neurons
            submodule = dict(self.model.named_modules())[submodule_name]
            
            logger.debug(f"Handling weights in {submodule_name}")
            # Check if the submodule has weights and biases
            if hasattr(submodule, 'weight'):
                for neuron_index, _ in neurons:
                    with torch.no_grad():
                        # Zero out the weights for the specified neuron index
                        if 0 <= neuron_index < submodule.weight.size(0):
                            submodule.weight[neuron_index, :] = 0.0  # Zero out weights
                            logger.debug(f"Zeroed out weights for {submodule_name} at neuron index {neuron_index}.")
                        else:
                            logger.debug(f"Neuron index {neuron_index} is out of bounds for {submodule_name} weights.")

        
            # Check if the submodule has a bias tensor
            logger.debug(f"Handling biases in {submodule_name}")
            if hasattr(submodule, 'bias'):
                if submodule.bias is not None:  # Check if bias is not None
                    # Check if bias is a tensor and has a length
                    if isinstance(submodule.bias, torch.Tensor):
                        for neuron_index, _ in neurons:
                            # Zero out the biases for the specified neuron index
                            if 0 <= neuron_index < submodule.bias.numel():  # Use numel() to check the number of elements
                                submodule.bias[neuron_index] = 0.0  # Zero out biases
                                logger.info(f"Zeroed out biases for {submodule_name} at neuron index {neuron_index}.")
                            else:
                                logger.warning(f"Neuron index {neuron_index} is out of bounds for {submodule_name} biases.")
                # else:
                #     logger.info(f"No bias in {submodule_name}")