"""
(Parses config? and)
Prepares dataset, runs experiments, and collects results.
"""

from typing import List, Optional
from functools import reduce
#from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

from llama import Llama, Dialog

import link_prediction.park_data_parser as rp
from .dialog_formatters import ZeroShotDialogFormatter

def get_parsed_assistant_response(response: str) -> str:
    valid_options = ["fact", "testimony", "policy", "value", "reference"]
    search_token = "Classification:"
    idx = response.index(search_token)
    possible_ans = response[idx+len(search_token):].strip().replace('"', '').lower()
    return possible_ans if possible_ans in valid_options else ""

def run_experiments(
    path_to_dataset: str,
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
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
    run_on_validation = False
    run_on_test = False

    # load dataset once (<1 mb)
    training_reviews = rp.deserialize_amazon_reviews_jsonlist(path_to_dataset)
    training_propositions = []
    for review in training_reviews:
        for prop in review.propositions:
            training_propositions.append(prop)

    # TODO: 1/7 holdout for examples, 3/7 for training, 3/7 validation
    training_set = training_propositions
    validation_set = []
    test_set = []

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Possible TODO: Log to see the token input to the model.
    system_prompt = 'The four types of propositions are "fact", "testimony", "policy", "value", and "reference". "Fact" is an objective proposition, meaning there are no subjective interpretations or judgements. "Testimony" is also an objective proposition that is experiential. "Policy" is a subjective proposition that insists on a specific course of action. "Value" is a subjective proposition that is a personal opinion or expression of feeling. "Reference" refers to a resource containing objective evidence. In product reviews, reference is usually a URL to another product page, image or video.'
    answer_token = "<answer>"
    answer_format = f"Classification: {answer_token}"
    user_prompt_format = 'Classify the following proposition as "fact", "testimony", "policy", "value", or "reference": "{}"\n' +\
        f'With no other text, answer in the desired format:\n{answer_format}'
    user_prompts = [user_prompt_format.format(training_sample.text) for training_sample in training_set]
    expected_results = [training_sample.type for training_sample in training_set]
    # TODO: Change this to just be a function
    dialogs: List[Dialog] = ZeroShotDialogFormatter(system_prompt, user_prompts, True).get_dialogs()
    results = []
    for i in range(0, len(dialogs), max_batch_size):
        dialogs_batch = dialogs[i:i+max_batch_size] if i+max_batch_size < len(dialogs) else dialogs[i:]
        # TODO: May need to clean the output if result is not in expected format
        batch_results = generator.chat_completion(
            dialogs_batch,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        #print([result['generation']['content'] for result in batch_results])
        results += [get_parsed_assistant_response(result['generation']['content']) for result in batch_results]
    #print(results)

    # Save results
    df = pd.DataFrame(data={"Id": [t.id for t in training_set], "Actual": results, "Expected": expected_results})
    df["AreEqual"] = np.where(df["Actual"] == df["Expected"], 1, 0)
    accuracy = sum(df["AreEqual"]) / len(df["AreEqual"])
    print(f"Accuracy: {accuracy}")
