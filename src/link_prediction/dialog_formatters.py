"""

"""
from typing import List, Tuple
from llama import Dialog

def get_dialogs(
    system_prompt: str,
    user_prompts: List[str],
    examples: List[Tuple[str, str]],  # TODO: NamedTuple or DataClass
    prefix_per_prompt: bool = True
) -> List[Dialog]:
    """
    Params:
        examples: List of questions and responses.
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
            dialog.append({"role": "user", "content": example[0]})
            dialog.append({"role": "assistant", "content": example[1]})
        dialog.append({"role": "user", "content": user_prompt})
        dialogs.append(dialog)
    return dialogs
