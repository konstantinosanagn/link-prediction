"""
Module to run prompting experiments using the LLAMA model, evaluate responses, and calculate performance metrics.
"""


from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from llama import Llama, Dialog

from src.lib.dataset_loaders import SplitData
from .dialog_formatters import get_dialogs, Example
from .assistant_response_parsers import BaseResponseParser

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
    
    # Determine which data splits to use based on flags
    splits_to_use = [data['train']]
    if run_on_validation and data['validation']:
        splits_to_use.append(data['validation'])
    if run_on_test and data['test']:
        splits_to_use.append(data['test'])
        
    # Returns a list of Comments when use_propositions: false in config, otherwise a list of Propositions in a given dataset that consists of Comments
    for split in splits_to_use:
       
        # TODO: Log to see the token input to the model.
        
        # Generate user prompts for the split
        user_prompts = [user_prompt_format.format(sample.text,
                                                  response_parser.answer_format.format(response_parser.answer_token))
                        for sample in split] 
        # print('===================================USER_PROMPTS===========================================')
        # print(user_prompts)
        # print('===================================END_USER_PROMPTS===========================================')
        
        # Create dialog instances for the prompts
        examples = [Example(
            user_prompt=user_prompt_format.format(example.text, response_parser.answer_format.format(response_parser.answer_token)),
            assistant_response=response_parser.answer_format.format(example.type))
                    for example in data['examples']]
        
        expected_results = [sample.type.lower() for sample in split]
        
        dialogs: List[Dialog] = get_dialogs(system_prompt, user_prompts, examples, True)
        results = []
        # Execute experiments in batches due to potential memory constraints
        for i in range(0, len(dialogs), max_batch_size):
            dialogs_batch = dialogs[i:i+max_batch_size] if i+max_batch_size < len(dialogs) else dialogs[i:]
            print('===========================DIALOGS_BATCH==============================')
            print(f"Dialogs batch: {dialogs_batch}")  # Inspect the batch content
            print('===========================END_DIALOGS_BATCH==============================')
            
            
            print(len(dialogs_batch))
            print(generator)
            print(max_gen_len)
            print(temperature)
            print(top_p)
            
            batch_results = generator.chat_completion(
                dialogs_batch,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p
            )
            
            # print(f"Generated results: {[result['generation']['content'] for result in batch_results]}")
            # parsed_results = [response_parser.get_parsed_response(result['generation']['content']) for result in batch_results]
            # print(f"Parsed results: {parsed_results}")
            # results += parsed_results
        print("=================================================================")

        # Save results
        # TODO: Record the success rate of each proposition type
        df = pd.DataFrame(data={"Id": [t.id for t in split], "Actual": results, "Expected": expected_results})
        
        # Calculate precision, recall, and F1 scores
        precision = precision_score(df["Expected"], df["Actual"], average='weighted')
        recall = recall_score(df["Expected"], df["Actual"], average='weighted')
        f1 = f1_score(df["Expected"], df["Actual"], average='weighted')

        # Print results
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        
        # TODO: Consider exporting results to external files or databases for further analysis
