"""
(Parses config? and)
Prepares dataset, runs experiments, and collects results.
"""

from typing import List, Optional
import pandas as pd
import numpy as np

from llama import Llama, Dialog

from src.dataset_loaders import SplitData
from .dialog_formatters import get_dialogs

ANSWER_TOKEN = '<answer>'
ANSWER_FORMAT = "Classification: {}"

def get_parsed_assistant_response(response: str) -> str:
    valid_options = ["fact", "testimony", "policy", "value", "reference"]
    search_token = "Classification:"
    idx = response.index(search_token)
    possible_ans = response[idx+len(search_token):].strip().replace('"', '').lower()
    return possible_ans if possible_ans in valid_options else ""

def run_experiment(
    generator: Llama,
    data: SplitData,
    system_prompt: str,
    user_prompt_format: str,
    temperature: float,
    top_p: float,
    max_batch_size: int,
    max_gen_len: Optional[int],
    run_on_validation = False,
    run_on_test = False
):
    """
    Runs prompting experiments.

    Args:
        path_to_dataset (str): Path to Amazon Review dataset.
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    # TODO: Validate args?
    splits_to_use = [data['train']]
    if run_on_validation and data['validation']:
        splits_to_use.append(data['validation'])
    if run_on_test and data['test']:
        splits_to_use.append(data['test'])
    for split in splits_to_use:
        # TODO: Log to see the token input to the model.
        user_prompts = [user_prompt_format.format(sample.text, ANSWER_FORMAT.format(ANSWER_TOKEN)) for sample in split]
        examples = [
                (user_prompt_format.format(example.text, ANSWER_FORMAT.format(ANSWER_TOKEN)),
                 ANSWER_FORMAT.format(example.type))
            for example in data['examples']
        ]
        expected_results = [sample.type for sample in split]
        # TODO: Change this to just be a function
        # TODO: This also needs data['examples']
        dialogs: List[Dialog] = get_dialogs(system_prompt, user_prompts, examples, True)
        results = []
        for i in range(0, len(dialogs), max_batch_size):
            dialogs_batch = dialogs[i:i+max_batch_size] if i+max_batch_size < len(dialogs) else dialogs[i:]
            batch_results = generator.chat_completion(
                dialogs_batch,  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            results += [get_parsed_assistant_response(result['generation']['content']) for result in batch_results]

        # Save results
        # TODO: Record the success rate of each proposition type
        df = pd.DataFrame(data={"Id": [t.id for t in split], "Actual": results, "Expected": expected_results})
        df["AreEqual"] = np.where(df["Actual"] == df["Expected"], 1, 0)
        accuracy = sum(df["AreEqual"]) / len(df["AreEqual"])
        print(f"Accuracy: {accuracy}")
