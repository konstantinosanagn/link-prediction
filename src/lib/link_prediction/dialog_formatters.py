"""
Module to create Dialogs that can be input into LLAMA generator.
"""
from dataclasses import dataclass
from typing import List
from llama import Dialog

@dataclass
class Example:
    """
    An example for multi-shot learning.
    Attributes:
        user_prompt (str): The user prompt part of the example.
        assistant_response (str): The assistant part of the example.
    """
    user_prompt: str
    assistant_response: str

def get_dialogs(
    system_prompt: str,
    user_prompts: List[str],
    examples: List[Example],
    prefix_per_prompt: bool = True
) -> List[Dialog]:
    """
    Creates a list of Dialogs from system, user prompts, and examples.
    Params:
        system_prompt (str): System prompt containing context for the model.
        user_prompts (List[str]): List of prompts to generate responses for.
        examples (List[Example]): List of prompts and responses.
        prefix_per_prompt (bool): Whether to prefix the system prompt before each user prompt.
                                  Defaults to True.
    Returns:
        List[Dialog]: Formatted Dialogs based on the input prompts and examples.
    """
    prefix_with_sys_prompt = system_prompt is not None
    dialogs = []
    for user_prompt in user_prompts:
        if not user_prompt:
            continue
        dialog = []
        if prefix_with_sys_prompt:
            # TODO: Should abstract as generate_system_msg(), etc in case format changes.
            dialog.append({"role": "system", "content": system_prompt})
            prefix_with_sys_prompt = prefix_per_prompt
        for example in examples:
            dialog.append({"role": "user", "content": example.user_prompt})
            dialog.append({"role": "assistant", "content": example.assistant_response})
        dialog.append({"role": "user", "content": user_prompt})
        dialogs.append(dialog)
    return dialogs
