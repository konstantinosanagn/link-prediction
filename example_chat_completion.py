# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
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
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs: List[Dialog] = [
        [
            {"role": "system", "content": 'The four types of propositions are "fact", "testimony", "policy", "value", and "reference". "Fact" is an objective proposition, meaning there are no subjective interpretations or judgements. "Testimony" is also an objective proposition that is experiential. "Policy" is a subjective proposition that insists on a specific course of action. "Value" is a subjective proposition that is a personal opinion or expression of feeling. "Reference" refers to a resource containing objective evidence. In product reviews, reference is usually a URL to another product page, image or video.'},
            {"role": "user", "content": 'Classify the following proposition as "fact", "testimony", "policy", "value", or "reference": "These damn things died within 3 months." Format your answer as "Classification: <answer>" with no other text.'}
        ],
        [
            {"role": "system", "content": 'The four types of propositions are "fact", "testimony", "policy", "value", and "reference". "Fact" is an objective proposition, meaning there are no subjective interpretations or judgements. "Testimony" is also an objective proposition that is experiential. "Policy" is a subjective proposition that insists on a specific course of action. "Value" is a subjective proposition that is a personal opinion or expression of feeling. "Reference" refers to a resource containing objective evidence. In product reviews, reference is usually a URL to another product page, image or video.'},
            {"role": "user", "content": 'Classify the following proposition as "fact", "testimony", "policy", "value", or "reference": "I wanted a headphone which does not press my hair." Format your answer as "Classification: <answer>" with no other text.'}
        ],
        [
            {"role": "system", "content": 'The four types of propositions are "fact", "testimony", "policy", "value", and "reference". "Fact" is an objective proposition, meaning there are no subjective interpretations or judgements. "Testimony" is also an objective proposition that is experiential. "Policy" is a subjective proposition that insists on a specific course of action. "Value" is a subjective proposition that is a personal opinion or expression of feeling. "Reference" refers to a resource containing objective evidence. In product reviews, reference is usually a URL to another product page, image or video.'},
            {"role": "user", "content": 'Classify the following proposition as "fact", "testimony", "policy", "value", or "reference": "Please think carefully about buying this headset." Format your answer as "Classification: <answer>" with no other text.'}
        ],
        [
            {"role": "system", "content": 'The four types of propositions are "fact", "testimony", "policy", "value", and "reference". "Fact" is an objective proposition, meaning there are no subjective interpretations or judgements. "Testimony" is also an objective proposition that is experiential. "Policy" is a subjective proposition that insists on a specific course of action. "Value" is a subjective proposition that is a personal opinion or expression of feeling. "Reference" refers to a resource containing objective evidence. In product reviews, reference is usually a URL to another product page, image or video.'},
            {"role": "user", "content": 'Classify the following proposition as "fact", "testimony", "policy", "value", or "reference": "Good all around general headphones." Format your answer as "Classification: <answer>" with no other text.'}
        ],
    ]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
