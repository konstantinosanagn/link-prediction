from typing import List, Optional
from llama import Llama, Dialog

import fire
import json
import random
import pandas as pd

# Sklearn:
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Parse json and return comments
def parse_dataset(file_path):
    comments = []
    with open(file_path, 'r') as file:
        for line in file:
            comment = json.loads(line)
            comments.append(comment)
    return comments

# Pass comments and return formatted comments
def format_comment(comment):
    propositions = [format_proposition(prop) for prop in comment['propositions']]
    return {
        "commentID": comment['commentID'],
        "propositions": propositions
    }

# Pass propositions and return formatted propositions
def format_proposition(prop):
    return {
        "id": prop['id'],
        "text": prop['text'],
        "type": prop['type'],
        "reasons": prop['reasons'],
        "evidence": prop['evidence']
    }

# # Print formatted test set
# formatted_test_set = [format_comment(comment) for comment in test_set]
# for formatted_comment in formatted_test_set:
#     print(json.dumps(formatted_comment, indent=2))

# # Print formatted only the first comment on the test_set
# formatted_first_comment = format_comment(test_set[0])
# print(json.dumps(formatted_first_comment, indent=2))

validation_set1 = parse_dataset('validation_set1.json')

# Main: run model
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    validation_set: Optional[List[Dict]] = validation_set1
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
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
        validation_set (List[Dict], optional): List of dictionaries representing the validation set. Each dictionary corresponds to a comment.

    """
    
    # Check whether a validation set is provided
    if validation_set is None:
        raise ValueError("Validation set must be provided.")

    # Initialize generator
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    
    dialogs: List[Dialog] = []

    # Iterate through the validation set to create dialogs
    dialogs = []  # List to store dialogs

    for user_comment in validation_set:
        propositions = user_comment['propositions']

        for proposition in propositions:
            # Define the system prompt
            system_prompt = {
                "role": "system",
                "content": 'The four types of propositions are "fact", "testimony", "policy", "value", and "reference". \
                "Fact" is an objective proposition, meaning there are no subjective interpretations or judgments. \
                "Testimony" is also an objective proposition that is experiential. \
                "Policy" is a subjective proposition that insists on a specific course of action. \
                "Value" is a subjective proposition that is a personal opinion or expression of feeling. \
                "Reference" refers to a resource containing objective evidence. In product reviews, reference is usually a URL to another product page, image or video.'
            }

            # Define the user prompt asking the model to predict the type
            user_prompt = {
                "role": "user",
                "content": f'Classify the following proposition as "fact", "testimony", "policy", "value", or "reference": "{proposition["text"]}". Only state ONE type.'
            }

            # Add system and user prompts to the dialog
            dialog = [system_prompt, user_prompt]
            dialogs.append(dialog)

    # Generate predictions
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p
    )

    # Create a DataFrame from the validation set
    records = []

    for user_comment in validation_set:
        comment_id = user_comment['commentID']
        propositions = user_comment['propositions']

        for proposition in propositions:
            prop_id = proposition['id']
            actual_type = proposition['type']
            records.append([comment_id, prop_id, actual_type])

    df = pd.DataFrame(records, columns = ['commentID', 'propositionID', 'actualType'])

    # Add predicted types to the DataFrame
    df['predictedType'] = [result['generation']['content'] for result in results]

    # Append predicted_types as the last column in the DataFrame
    predicted_types = df['predictedType']

    # Convert columns to lists for sklearn metrics
    actual_types_list = actual_types.tolist()
    predicted_types_list = predicted_types.tolist()

    # Calculate precision, recall, and F1 scores
    precision = precision_score(actual_types_list, predicted_types_list, average='weighted')
    recall = recall_score(actual_types_list, predicted_types_list, average='weighted')
    f1 = f1_score(actual_types_list, predicted_types_list, average='weighted')

    # Print results
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

