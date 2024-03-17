"""
Runs experiment.
"""
import json
from typing import Optional
import fire
from llama.generation import Llama

from src.lib.link_prediction.experiment_driver import run_experiment
from src.lib.dataset_loaders import *
from src.lib.link_prediction.assistant_response_parsers import *

def main(
    path_to_cfg: str,
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    run_on_validation = False,
    run_on_test = False,
    max_batch_size: int = 8,
    max_seq_len: int = 128,
    max_gen_len: Optional[int] = None
):
    # Parse config, run experiments.
    cfg = None
    with open(path_to_cfg, 'r', encoding='utf-8') as f:
        # TODO: Validate config against schema
        cfg = json.load(f)
    data_loader = globals()[cfg['dataset_loader']](
            cfg['dataset_dir_path'],
            cfg['use_propositions'],
            cfg['seed'],
            cfg['train'],
            cfg['validation'],
            cfg['test']
        )
    response_parser: BaseResponseParser = globals()[cfg['response_parser']]()
    generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size)
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

if __name__ == '__main__':
    fire.Fire(main)
