"""
Runs experiment.
"""
import json
from typing import Optional
import fire                         # Library for building command line interfaces.
from llama.generation import Llama  # Importing Llama model from the llama package
import traceback

# Imports for experiment setup and data handling
from src.lib.link_prediction.new_experiment_driver import run_experiment
from src.lib.dataset_loaders import *
from src.lib.link_prediction.assistant_response_parsers import *
from src.lib.link_prediction.contextual_assistant_response_parser import *

def main(
    path_to_cfg: str,                   # Path to the JSON config file
    ckpt_dir: str,                      # Directory where model chekcpoints are       
    tokenizer_path: str,                # Path to tokenizer
    temperature: float = 0.6,           # Temperature parameter for generation
    top_p: float = 0.9,                 # Top-p parameter for nucleus sampling
    run_on_validation = False,          # Flag to run expirements on validation set
    run_on_test = False,                # Flag to run expirements on test set
    max_batch_size: int = 8,            # Maximum number of samples per batch
    max_seq_len: int = 512,             # Maximum sequence length for inputs
    max_gen_len: Optional[int] = None   # Optional maximum generation length
):
    # Parse config from a JSON, run experiments.
    cfg = None
    with open(path_to_cfg, 'r', encoding='utf-8') as f:
        # TODO: Validate config against schema
        cfg = json.load(f)
        
    # Load dataset using config
    data_loader = globals()[cfg['dataset_loader']](
            cfg['train_path'],
            cfg['validation_path'],
            cfg['test_path'],
            cfg['use_propositions'],
            cfg['seed'],
        )
    
    # Load response parser class based on config
    response_parser: SupportResponseParser = globals()[cfg['response_parser']]()
    
    # Build LLAMA generator
    generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size)
    
    # Iterate over experiment defined in the config
    for experiment_cfg in cfg['experiments']:
        print(f"Running Experiment {experiment_cfg['name']}")
        data = data_loader.get_splits(experiment_cfg['num_examples'])
        print("Data: ============================")
        print(data)
        print("==========================")
        try:
            run_experiment(
                    generator,
                    response_parser,
                    data,
                    experiment_cfg['system_prompt'],
                    experiment_cfg['user_prompt_format'],
                    temperature,
                    top_p,
                    max_batch_size,
                    max_gen_len,
                    run_on_validation,
                    run_on_test)
        except Exception as e:
            print(e)
            traceback.print_exc()

if __name__ == '__main__':
    fire.Fire(main)
