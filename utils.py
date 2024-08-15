# %%
import glob
import random
import sys
import json
import os
import argparse

from collections import Counter, defaultdict
from typing import Dict, List

# %%
def cluster_read(fname):
    """
    Given a txt file containing the latent concepts of a corresponding layer, The
    function loads all data in the file and returns it in form of lists. These lists 
    will be then used to create mappings between clusters, words, and sentences 

    Parameters
    ----------
    fname : str
        Path to where the latent concepts data is stored for a corresponding
        layer (Usually saved in a .txt file).

    Returns
    -------
    words: List
        A list of words corresponding to latent concepts of the passed data.
        Each word will be associated with a latent concept (also called a cluster)
    words_idx: List
        A list of word indices corresponding to occurence location of each word
        in the sentences.
    cluster_idx: List
        A list of cluster ids corresponding to the data passed. Each layer will
        have a group of clusters and each cluster contains a group of words
    sent_idx: List
        A list of sentence ids corresponding to which sentences the concept appears in
    """
    words = []
    words_idx = []
    cluster_idx = []
    sent_idx = []
    with open(fname) as f:
        for line in f:
            line  = line.rstrip('\r\n')
            parts = line.split("|||")
            words.append(parts[0])
            cluster_idx.append(int(parts[4]))
            words_idx.append(int(parts[3]))
            sent_idx.append(int(parts[2]))
    return words, words_idx, sent_idx, cluster_idx

def read_cluster_data(fname):
    """
    Given a .txt file corresponding to latent concepts of a layer, the function
    returns a mapping between cluster ids and words. The words corresponding to
    each cluster are returned in a list

    Parameters
    ----------
    fname : str
        Path to where the latent concepts data is stored for a corresponding
        layer (Usually saved in a .txt file).

    Returns
    -------
    clusterToWords: Dict
        A mapping (or dictionary) between cluster and words. The keys of the dictionary
        are clusters, and the corresponding values are words corresponding to that cluster.  
    """
    clusterToWords  = defaultdict(list)
    words, words_idx, sent_idx, cluster_idx = cluster_read(fname)
    for i, elem in enumerate(cluster_idx):
        cluster = "c" + str(cluster_idx[i])
        clusterToWords[cluster].append(words[i])
    return clusterToWords


def read_annotations(path):
    """
    Given a path to the annotations file; the function 
    returns the LLM annotations for the clusters in the form of 
    a dictionary
    
    Parameters
    ----------
    fname : str
        Path to JSON annotations file
    Returns
    -------
    labels: Dict
        LLM labels for the clusters. 
    """
    with open(path, "r") as reader: 
        labels = json.load(reader)
    return labels

def read_sentences(path_to_sentences: str) -> List: 
    """
    Given a path to the sentences file, the function returns
    a list of sentences

    Parameters
    ----------
    path_to_sentences: str :
        A path to where the sentences file is stored

    Returns
    -------
    sentences: List
        A list of sentences 
    """
    sentences = []
    with open(path_to_sentences, "r") as reader: 
        data = json.load(reader) 
        for line in data: 
            l = line.rstrip('\r\n')
            sentences.append(l) 
    return sentences


def load_all_cluster_data(clusters_path): 
    """
    Given a path to where the cluster data is stored, the function 
    returns a dictionary where each key is a cluster id, and each value 
    is a list of tuples consisting of tokens, sentence ids, and token ids 
    (corresponding to the cluster)

    Parameters
    ----------
    clusters_path: str :
        A path to where the cluster data is stored
        
    Returns
    -------
    clusters: Dict
        A dictionary containing the cluster data. 
    """
    clusters = defaultdict(list)
    with open(clusters_path) as fp:
        for line_idx, line in enumerate(fp):
            token, _, sentence_idx, token_idx, cluster_idx = line.strip().rsplit("|||")

            sentence_idx = int(sentence_idx)
            token_idx = int(token_idx)
            cluster_idx = int(cluster_idx)

            clusters["c" + str(cluster_idx)].append((token, sentence_idx, token_idx))
    return clusters


# %%
def cluster_to_json(clusters_path):
    all_cluster_data =  load_all_cluster_data(clusters_path)
    # Specify the file name
    file_name = clusters_path[:-4] + ".json"


    # Dump the dictionary to a JSON file
    with open(file_name, 'w') as json_file:
        json.dump(all_cluster_data, json_file, indent=4)


# %%
def append_path_to_keys(data, file_path):
    """Recursively append the file path to each key in the JSON data."""
    if isinstance(data, dict):
        return {f"{file_path}.{key}": append_path_to_keys(value, file_path) for key, value in data.items()}
    elif isinstance(data, list):
        return [append_path_to_keys(item, file_path) for item in data]
    else:
        return data  # Return the value as is if it's neither a dict nor a list


# %%
def combine_json_files_by_name_recursive(directory, target_filename, output_file):
    combined_data = []

    # Normalize the target filename to lowercase for case-insensitive comparison
    target_filename_lower = target_filename.lower()

    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file matches the target filename (case-insensitive)
            if file.lower() == target_filename_lower and file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                        # Append the file path to each key in the JSON data
                        modified_data = append_path_to_keys(data, file_path)
                        combined_data.append(modified_data)  # Append modified data
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from file {file_path}: {e}")

    # Write the combined data to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)

# %%
def filter_json_by_keywords(data, keywords):
    """Recursively filter JSON data for key/value pairs where the value contains any of the keywords."""
    filtered_data = []

    # Convert keywords to lowercase for case-insensitive comparison
    keywords_lower = [keyword.lower() for keyword in keywords]

    for item in data:
        if isinstance(item, dict):
            filtered_item = {}
            for key, value in item.items():
                if isinstance(value, str):
                    # Check if any keyword is in the value
                    if any(keyword in value.lower() for keyword in keywords_lower):
                        filtered_item[key] = value  # Add the key/value pair if any keyword matches
            if filtered_item:  # Only add non-empty results
                filtered_data.append(filtered_item)

    return filtered_data

def filter_json_file(input_file, keywords, output_file):
    """Filter a JSON file for key/value pairs containing any of the specified keywords and save the results."""
    with open(input_file, 'r') as f:
        try:
            data = json.load(f)
            filtered_data = filter_json_by_keywords(data, keywords)

            # Write the filtered data to a new JSON file
            with open(output_file, 'w') as out_f:
                json.dump(filtered_data, out_f, indent=4)

            print(f"Filtered data saved to {output_file}.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {input_file}: {e}")

# Example usage
# filter_json_file('input.json', ['keyword1', 'keyword2'], 'filtered_output.json')

# %%
def extract_and_split_keys(data):
    """Extract keys from a list of dictionaries in the JSON structure and split the keys."""
    result_dict = {}

    for item in data:
        if isinstance(item, dict):
            for key, value in item.items():
                # Split the key at the last dot
                if '.' in key:
                    base_key = key.rsplit('.', 1)[0]  # Get everything before the last dot
                    value_key = key.rsplit('.', 1)[1]  # Get everything after the last dot
                    result_dict[base_key] = value_key  # Store in the result dictionary

    return result_dict

def extract_keys_from_file(input_file, output_file):
    """Load a JSON file, extract keys, split them into a dictionary, and save to a new JSON file."""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)  # Load the JSON data
            result_dict = extract_and_split_keys(data)  # Extract and split keys
            
            # Save the result to a new JSON file
            with open(output_file, 'w') as out_f:
                json.dump(result_dict, out_f, indent=4)  # Write the dictionary to the output file
            
            print(f"Output saved to {output_file}.")
            return result_dict
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {input_file}: {e}")
        return {}

# %%
