import logging
import traceback
from src.utils import load_data, load_model, save_results
from src.neuron_saliency_analysis import ConceptNeuronSaliencyAnalyzer
from src.chat import generate_responses
from src.config_loader import load_config
from src.logger import setup_logging, log_config
import torch

def main():
    # Load configuration
    setup_logging()
    logger = logging.getLogger()
    config = load_config()
    log_config(config)

    try:
        #Load the concept examples and test prompts
        test_prompts, concept_examples, background_examples = load_data(config)

        # Load the model
        model, tokenizer = load_model(config["base_model"]["base_model_path"], config["base_model"]["device"])

        # # Generate responses using the chat module
        # generate_responses(test_prompts, model, tokenizer)
    
    except Exception as e:
        logger.error(f"An error occurred during target model testing: {e}")
        logger.debug(traceback.format_exc())

    logger.info("Loading model to determine layer count...")
    num_layers = len(model.model.layers)
    layer_nums = list(range(num_layers - config['neural_pruning']['num_layers'], num_layers))
    logger.info(f"Model has {num_layers} layers. Analyzing layers {layer_nums}...")

    logger.info("Analysing concept neurons...")
    analyzer = ConceptNeuronSaliencyAnalyzer(model, tokenizer, config["base_model"]["device"])

    # Analyze concept saliency for the top 10 layers
    results = analyzer.analyze_concept_saliency(
        concept_texts=concept_examples,
        background_texts=background_examples,
        num_layers=config['neural_pruning']['num_layers'],
        top_k=config['neural_pruning']['neurons_per_layer'],
        statistical_test=True
    )
    
    # Zero out neurons
    analyzer.zero_out_neurons(results)
        
    try:
        logger.info("Saving results...")
        save_results(analyzer.model, 
                     tokenizer,
                     model_file_path = config["neural_pruning"]["pruned_model_path"])
        logger.info(f"Results saved to {config['neural_pruning']['pruned_model_path']}")

        #Load the pruned model and generate responses
        generate_responses(test_prompts, analyzer.model, tokenizer)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    main()