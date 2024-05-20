from typing import List, Optional
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from llama import Llama, Dialog
from src.lib.dataset_loaders import SplitData, EnhancedDatasetLoader
from .dialog_formatters import get_dialogs, Example
from .assistant_response_parsers import BaseResponseParser
from .contextual_assistant_response_parser import SupportResponseParser


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
    
    for split in splits_to_use:
        user_prompts = []
        for sample in split:
            surrounding_texts = sample.surrounding
            surrounding_str = ', '.join([f'"{text}"' if text else 'None' for text in surrounding_texts])
            user_prompts.append(user_prompt_format.format(sample.text, surrounding_str, "{}"))

        examples = [Example(
            user_prompt=user_prompt_format.format(example.text, ', '.join([f'"{text}"' if text else 'None' for text in example.surrounding]), "{}"),
            assistant_response=response_parser.answer_format.format(example.surrounding, "yes/no"))
                    for example in data['examples']]
        
        expected_results = [sample.type.lower() for sample in split]
        
        dialogs: List[Dialog] = get_dialogs(system_prompt, user_prompts, examples, True)
        results = []
        for i in range(0, len(dialogs), max_batch_size):
            dialogs_batch = dialogs[i:i+max_batch_size] if i+max_batch_size < len(dialogs) else dialogs[i:]
            batch_results = generator.chat_completion(
                dialogs_batch,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p
            )
            parsed_results = [response_parser.get_parsed_response(result['generation']['content']) for result in batch_results]
            results += parsed_results

        df = pd.DataFrame(data={"Id": [t.id for t in split], "Actual": results, "Expected": expected_results})
        
        # Convert labels to a binary format for multi-label classification
        mlb = MultiLabelBinarizer()
        y_true = mlb.fit_transform(df["Expected"].apply(lambda x: [x] if isinstance(x, str) else x))
        y_pred = mlb.transform(df["Actual"].apply(lambda x: [x] if isinstance(x, str) else x))
        
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Print results
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        
        # TODO: Consider exporting results to external files or databases for further analysis
