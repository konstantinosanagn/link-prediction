"""
Dataclasses and JSON deserialization for review dataset.
"""

import json
import os
import random
from typing import List
from dataclasses import dataclass


@dataclass
class Proposition:
    """
    Proposition data object for part of an Amazon Review.
    """
    id: int
    type: str
    text: str
    reasons: List[str]
    evidence: List[str]

@dataclass
class Comment:
    """
    Comment data object containing propositions.
    """
    propositions: List[Proposition]
    id: str

def deserialize(obj) -> Comment:
    propositions_raw = obj['propositions']
    propositions = [Proposition(id=prop['id'],
                                type=prop['type'],
                                text=prop['text'],
                                reasons=prop.get('reasons'),
                                evidence=prop.get('evidence'))
                    for prop in propositions_raw]
    return Comment(id=obj['id'], propositions=propositions)

def deserialize_comments_jsonlist(filepath: str) -> List[Comment]:
    """
    Deserializes file containing a list of JSON Amazon Review objects.
    Expects each review to be on a new line.

    Args:
        filepath (str): Path to file to deserialize.
    Returns:
        List[AmazonReview]: List of deserialized Amazon Reviews.
    """
    
    comments = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                comments.append(deserialize(json.loads(line)))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return comments


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    filepath = args[0]
    reviews = deserialize_comments_jsonlist(filepath)
    for review in reviews:
        print(review)
    print(len(reviews))
