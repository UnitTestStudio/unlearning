# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from src.neuron_saliency_analysis import ConceptNeuronSaliencyAnalyzer
import torch

# Load the tokenizer and model
model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Specify the model version you want to use
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
#%%
# # Extract the names of the transformer layers
# layer_names = []

# # Iterate through the layers and construct the names
# for i, layer in enumerate(model.model.layers):
#     # Construct a name based on the layer index
#     layer_name = f"layer.{i}"  # You can customize the naming convention as needed
#     layer_names.append(layer_name)

# # Print the list of layer names
# print(layer_names)

# %%
cns = ConceptNeuronSaliencyAnalyzer(model, tokenizer, "cuda")
layer_names = cns._detect_transformer_layers(model)


#Print the list of layer names
print(layer_names)

# %%
neurons = [0, 1, 2, 3, 4]

for layer_name, neuron in zip(layer_names, neurons):
    for name, module in model.named_modules():
        if name == layer_name:
            print(f"Zeroing out neurons in layer: {layer_name}")
            print(neurons)
            for neuron_idx in neurons:
                print(neuron_idx)
                if hasattr(module, 'weight'):
                    print(f"Zeroing out neuron {neuron_idx} in layer {layer_name}")
            #         with torch.no_grad():
            #             print(f"Zeroing out neuron {neuron_idx} in layer {layer_name}")
            #             # module.weight[neuron_idx].zero_()
            #             # if module.bias is not None:
            #             #     module.bias[neuron_idx].zero_()
# %%
# Access the 9th layer's self_attn
layer_index = 9
self_attention = model.model.layers[layer_index].self_attn

# Check if self_attn has weight attributes in its components
has_weights = (
    hasattr(self_attention, 'q_proj') and hasattr(self_attention.q_proj, 'weight') or
    hasattr(self_attention, 'k_proj') and hasattr(self_attention.k_proj, 'weight') or
    hasattr(self_attention, 'v_proj') and hasattr(self_attention.v_proj, 'weight')
)

if has_weights:
    print(f"Layer {layer_index} self_attn has weight attributes in its components.")
else:
    print(f"Layer {layer_index} self_attn does not have weight attributes in its components.")


# %%
