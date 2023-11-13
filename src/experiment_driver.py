"""
(Parses config? and)
Prepares dataset, runs experiments, and collects results.
"""

from typing import List, Optional
from sklearn.model_selection import KFold
import fire

from llama import Llama, Dialog

import src.amazon_review_parser as rp

def main(
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

    # load dataset once (<1 mb)
    training_reviews = rp.deserialize_amazon_reviews_jsonlist(path_to_dataset)
    training_propositions = []
    for review in training_reviews:
        training_propositions += review.propositions

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Optionally do k-fold cross validation w/some split defined by param.
    k = 5
    seed = 1
    # Split w/sklearn, do k-fold CV myself.
    kf = KFold(n_splits=k, random_state=seed, shuffle=True)
    # For each fold, sample from training_reviews' _propositions_ to get a single training and validation split.
    # Then, format Dialogs list so System role has definition, no assistant (zero-shot).
    # The actual prompt will be an instruction to the model to classify the proposition (requires some formatting).
    #
    # Possible TODO: Log to see the token input to the model.
    for i, (train_idxs, validation_idxs) in enumerate(kf.split(training_propositions)):
        prompt_prefix = "Classify the following proposition as policy, fact, value, or testimony: "
        user_prompts = [f"{prompt_prefix}{training_propositions[j].text}" for j in train_idxs]
        system_prompt = "The four types of propositions are policy, fact, value, and testimony. Fact is an objective proposition, meaning it does not leave any room for subjective interpretations or judgements. Testimony is also an objective proposition. However, it differs from fact in that it is experiential, i.e., it describes a personal state or experience. Policy is a subjective proposition that insists on a specific course of action. Value is a subjective proposition that is not policy. It is a personal opinion or expression of feeling. Reference is the only non-proposition elementary unit that refers to a resource containing objective evidence. In product reviews, reference is usually a URL to another product page, image or video."

    # What is the independent variable? The format of the List of Dialogs (prompt format).
    dialogs: List[Dialog] = []
    # Try 1 System in first Dialog. Then try System in each Dialog. These can each be an experiment.
    # Pass selected training/validation to experiment input formatter. Pass a max_seq_len so return val can be a list of lists if necessary.
    # Run the model on the formatted prompts from the previous step's input.

    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    # Save results
    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
