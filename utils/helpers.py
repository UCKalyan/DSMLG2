import os
import yaml
import numpy as np

def load_config(config_path='config/config.yaml'):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def ensure_dir(directory):
    """Ensures that a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_npy(data, path):
    """Saves a numpy array to a .npy file."""
    np.save(path, data)

def load_npy(path):
    """Loads a numpy array from a .npy file."""
    return np.load(path)
