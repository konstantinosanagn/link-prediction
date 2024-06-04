import re
from .assistant_response_parsers import BaseResponseParser
from typing import List

class SupportResponseParser(BaseResponseParser):
    def __init__(self,
                 answer_token: str='<answer>',
                 answer_format: str='Answer: {}'):
        super().__init__(answer_token, answer_format)
        
    def _get_answer_using_regex_match(self, pattern: str, response: str) -> str:
        match: re.Match = re.search(pattern, response, re.IGNORECASE)
        result = match.group(1) if match else ""
        return result

    def get_parsed_response(self, response: str) -> str:
        """
        See parent.
        """
        # Define the possible answers
        proposition_types = set(['yes', 'no'])

        # Define regex patterns to capture the response
        patterns = [
            r'".*?\b(yes|no)\b.*?"',  # Capture 'yes' or 'no' within quotes
            r'\b(yes|no)\b', # Capture 'yes' or 'no' as standalone words
            r'\b(support|attack)\b',
            r'".*?\b(support|attack)\b.*?"'
        ]

        print(f"Parsing assistant response: {response}")

        # Try matching the response with each pattern
        for pattern in patterns:
            search_res = self._get_answer_using_regex_match(pattern, response)
            if search_res:
                break
        else:
            return ""

        # Normalize the response
        possible_ans = search_res.strip().lower()
        print(f"Possible parsed response: {possible_ans}")

        return possible_ans if possible_ans in proposition_types else ""

# Usage example
if __name__ == '__main__':
    parser = SupportResponseParser()
    response = 'The answer is "Yes".'
    print(parser.get_parsed_response(response))  # Expected output: 'yes'
