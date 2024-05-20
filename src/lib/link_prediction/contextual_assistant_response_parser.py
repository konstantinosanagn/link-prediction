import re
from .assistant_response_parsers import BaseResponseParser
from typing import List

class SupportResponseParser(BaseResponseParser):
    def __init__(self,
                 answer_token: str='<answer>',
                 answer_format: str='{}: {}'):
        super().__init__(answer_token, answer_format)

    def get_parsed_response(self, response: str) -> List[str]:
        pattern = re.compile(r'"(.*?)":\s*(yes|no)', re.IGNORECASE)
        matches = pattern.findall(response)
        return matches if matches else []
