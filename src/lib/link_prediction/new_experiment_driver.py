# Experiment_driver.py
from typing import List, Optional
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import traceback

from llama import Llama, Dialog
from src.lib.dataset_loaders import SplitData, EnhancedDatasetLoader
from .dialog_formatters import get_dialogs, Example
from .assistant_response_parsers import BaseResponseParser
from .contextual_assistant_response_parser import SupportResponseParser

from src.lib.dataset_loaders.park_data_parser import generate_proposition_pairs, create_reasons_map

def run_experiment(
    generator: Llama,
    response_parser: SupportResponseParser,
    data: SplitData,
    system_prompt: str,
    user_prompt_format: str,
    temperature: float,
    top_p: float,
    max_batch_size: int,
    max_gen_len: Optional[int],
    run_on_validation=False,
    run_on_test=False
):
    splits_to_use = [data['train']]
    if run_on_validation and data['validation']:
        splits_to_use.append(data['validation'])
    if run_on_test and data['test']:
        splits_to_use.append(data['test'])
        
    support_data = []  # List to keep track of proposition support information
    
    # Create the reasons map to parse actual/true answer from reasons attribute of each proposition
    reasons_map = create_reasons_map([comment for split in splits_to_use for comment in split])
    
    # A list of Comments when use_propositions: false in config, otherwise a list of Propositions in a given dataset that consists of Comments
    for split in splits_to_use:
        user_prompts = []
        for comment in split:
            # Generate proposition pairs for each comment
            pairs = generate_proposition_pairs(comment)
            for prop, surrounding in pairs:
                for s_prop in surrounding:
                    user_prompt = user_prompt_format.format(prop.text, s_prop.text,
                                                  response_parser.answer_format.format(response_parser.answer_token))
                    user_prompts.append(user_prompt)
                    # Track proposition support information
                    support_data.append({
                        'comment_id': comment.id,
                        'proposition_id': prop.id,
                        'sup_proposition_id': s_prop.id,
                        'support_boolean': None
                    })
                    
                    # print("============================USER_PROMPT========================================")
                    # print(user_prompt)
                    # print("==============================END_USER_PROMPT======================================")
                    
        # print('===================================USER_PROMPTS===========================================')
        # print(user_prompts)
        # print('===================================END_USER_PROMPTS===========================================')

        examples = [] #TODO: Fix how to pull examples from config
        
        dialogs: List[Dialog] = get_dialogs(system_prompt, user_prompts, examples, True)
        print('------------------------DIALOGS LOADED----------------------')
        
        results = []
        for i in range(0, len(dialogs), max_batch_size):
            dialogs_batch = dialogs[i:i+max_batch_size] if i+max_batch_size < len(dialogs) else dialogs[i:]
            print(f'Processing batch {i//max_batch_size + 1}/{(len(dialogs) + max_batch_size - 1)//max_batch_size}')
            
            print('===========================DIALOGS_BATCH==============================')
            print(f"Dialogs batch: {dialogs_batch}")  # Inspect the batch content
            print('===========================END_DIALOGS_BATCH==============================')
            
            try:
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

                print("-------------------------------------------------------------")
                
                # Update support_boolean in support_data based on parsed_results
                for idx, parsed_result in enumerate(parsed_results):
                    support_data[i + idx]['support_boolean'] = True if 'yes' in parsed_result.lower() else False
                    
            except Exception as e:
                print(f"Error during batch processing: {e}")
                # traceback.print_exc()  # Print full traceback 
        
        print("=================================================================")
        
        # Optionally convert support_data to a DataFrame for easier processing
        df_support = pd.DataFrame(support_data)

        # Add the support_actual column
        df_support['support_actual'] = df_support.apply(
            lambda row: row['sup_proposition_id'] in reasons_map.get((row['comment_id'], row['proposition_id']), []),
            axis=1
        )
        
        # print(df_support.head(n=100))  # Print a sample of the support data
        
        # Print entire generated dataset 
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(df_support)  # Print a sample of the support data
        
        # Print rows where support_actual is True
        # print(df_support[df_support['support_actual'] == True])
        
        # Extract true and predicted labels
        y_true = df_support['support_actual']
        y_pred = df_support['support_boolean']
        
        # Compute precision, recall, F1 score, and accuracy
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Print results
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Accuracy: {accuracy:.4f}')
