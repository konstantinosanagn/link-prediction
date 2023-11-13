"""
Blah
"""
import random
from llama import Llama
from typing import List

import src.review_parser as rp


def example(path_to_dataset: str, chkpt_dir, tokenizer_path, temperature, top_p, max_seq_len, max_gen_len, max_batch_size):
    """

    """
    # load dataset once
    training_reviews = rp.deserialize_amazon_reviews_jsonlist(path_to_dataset)

    # optionally split dataset once
    random.seed(0)
    validation_idxs = set(random.sample(range(len(training_reviews)),
                                        int(0.2 * len(training_reviews))))
    training = []
    validation = []
    for idx, review in enumerate(training_reviews):
        if idx in validation_idxs:
            validation.append(review)
        else:
            training.append(review)

    # Load model once
    generator = Llama.build(chkpt_dir, tokenizer_path, max_seq_len, max_batch_size)

    # Iterate over experiments (in a config eventually).
    # Experiment definition:
    # Has to encapsulate some Prompt structure:
    #   List of Prompt objects. Prompt contains some prefix, suffix, maybe a formatter?
    #
    # Use dataset to generate prompts
    # For review in training, format the prompt as "Here is the definition of each proposition type.
    # What is the proposition type of the following sentence? <sentence>"
    prompts: List[str] = []
    results = generator.text_completion(prompts, max_gen_len, temperature, top_p)
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n========================================\n")
