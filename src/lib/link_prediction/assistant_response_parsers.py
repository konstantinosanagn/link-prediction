"""
Module containing parsing clases for a Llama-generated response.
"""

from abc import ABC, abstractmethod
import re

class BaseResponseParser(ABC):
    """
    Abstract base class for parsing an assistant response.
    """
    def __init__(self, answer_token: str, answer_format: str):
        """
        Attributes:
            answer_token (str): The token placeholder for an answer.
            answer_format (str): A format string for the expected assistant response.
        """
        self.answer_token = answer_token
        self.answer_format = answer_format

    @abstractmethod
    def get_parsed_response(self, response: str) -> str:
        """
        Parses an assistant response for a particular subsequence.
        Args:
            response (str): The assistant response to parse.
        Returns:
            A processed subsequence of the response.
        """
        raise NotImplementedError

class PropositionResponseParser(BaseResponseParser):
    """
    Parses the proposition classification from an assistant response.
    """
    def __init__(self,
                 answer_token: str='<answer>',
                 answer_format: str='Classification: {}'):
        """
        See parent.
        """
        super().__init__(answer_token, answer_format)
        # This info is repeated in multiple places
        self.proposition_types = set(['fact', 'testimony', 'policy', 'value', 'reference'])
        # Convert answer_format to regex
        self.answer_format_regex: re.Pattern = re.compile(answer_format.replace('{}', '(\w+)'),
                                                          re.IGNORECASE)

    def get_parsed_response(self, response: str) -> str:
        """
        See parent.
        """
        print(f"Parsing assistant response: {response}")
        search_res: re.Match = self.answer_format_regex.search(response)
        if not search_res:
            raise ValueError(f'response did not match the answer format: {response}')
        # Llama's answer will be somewhere after the search_token
        possible_ans = search_res.group(0).strip().replace('"', '').lower()
        print(f"Possible parsed response: {possible_ans}")
        # BUG: Should throw?
        return possible_ans if possible_ans in self.proposition_types else ""
