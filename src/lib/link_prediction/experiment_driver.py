"""
Module containing function to run a prompting experiment.
"""

from typing import List, Optional
import pandas as pd
import numpy as np

from llama import Llama, Dialog

from src.lib.dataset_loaders import SplitData
from .dialog_formatters import get_dialogs, Example
from .assistant_response_parsers import BaseResponseParser

from sklearn.metrics import precision_score, recall_score, f1_score

def run_experiment(
    generator: Llama,
    response_parser: BaseResponseParser,
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
        generator (Llama): Generates Llama assistant responses.
        response_parser (BaseResponseParser): Object to parse assistant responses.
        data (SplitData): Typed dict mapping data type (train, validation, test) to a dataset.
        system_prompt (str): A system prompt to provide the model with context.
        user_prompt_format (str): A format string for the user prompt.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
        run_on_validation (bool): Whether to also generate responses for prompts from the validation set.
            Defaults to False.
        run_on_test (bool): Whether to also generate responses for prompts from the test set.
            Defaults to False.
    """
    # TODO: Validate args?
    splits_to_use = [data['train']]
    if run_on_validation and data['validation']:
        splits_to_use.append(data['validation'])
    if run_on_test and data['test']:
        splits_to_use.append(data['test'])
    for split in splits_to_use:
        # TODO: Log to see the token input to the model.
        user_prompts = [user_prompt_format.format(sample.text,
                                                  response_parser.answer_format.format(response_parser.answer_token))
                        for sample in split]
        examples = [Example(
            user_prompt=user_prompt_format.format(example.text, response_parser.answer_format.format(response_parser.answer_token)),
            assistant_response=response_parser.answer_format.format(example.type))
                    for example in data['examples']]
        expected_results = [sample.type.lower() for sample in split]
        dialogs: List[Dialog] = get_dialogs(system_prompt, user_prompts, examples, True)
        results = []
        # Have to submit prompts in batches
        for i in range(0, len(dialogs), max_batch_size):
            dialogs_batch = dialogs[i:i+max_batch_size] if i+max_batch_size < len(dialogs) else dialogs[i:]
            print(f"Dialogs: {dialogs_batch}")
            batch_results = generator.chat_completion(
                dialogs_batch,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p
            )
            print(f"Generated results: {[result['generation']['content'] for result in batch_results]}")
            parsed_results = [response_parser.get_parsed_response(result['generation']['content']) for result in batch_results]
            print(f"Parsed results: {parsed_results}")
            results += parsed_results
        print("=================================================================")

        # After collecting all results
        actual_types_list = expected_results  # This should already be populated with actual types
        predicted_types_list = results  # This should be populated with the predicted types as parsed from the generator's responses

        # Compute precision, recall, and F1 score
        precision = precision_score(actual_types_list, predicted_types_list, average='weighted', labels=np.unique(predicted_types_list))
        recall = recall_score(actual_types_list, predicted_types_list, average='weighted', labels=np.unique(predicted_types_list))
        f1 = f1_score(actual_types_list, predicted_types_list, average='weighted', labels=np.unique(predicted_types_list))

        # Print or log the performance metrics
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
