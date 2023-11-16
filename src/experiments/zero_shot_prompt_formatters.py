"""

"""
from typing import List
from llama import Dialog

class ZeroShotPromptFormatter():
    def __init__(self, system_prompt: str, user_prompts: List[str], prefix_per_prompt: bool = False)\
            -> None:
        self.system_prompt = system_prompt
        self.user_prompts = user_prompts
        self.prefix_per_prompt = prefix_per_prompt

    def get_dialogs(self) -> List[Dialog]:
        """

        """
        prefix_with_sys_prompt = self.system_prompt is not None
        dialogs: List[Dialog] = []
        for user_prompt in self.user_prompts:
            if not user_prompt:
                continue
            dialog: Dialog = []
            if prefix_with_sys_prompt:
                # TODO: Should abstract as generate_system_msg(), etc in case format changes.
                dialog.append({"role": "system", "content": self.system_prompt})
                prefix_with_sys_prompt = self.prefix_per_prompt
            dialog.append({"role": "user", "content": user_prompt})
            dialogs.append(dialog)
        return dialogs
