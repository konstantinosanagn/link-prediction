"""
Dataclasses and JSON deserialization for review dataset.
"""

import json
import os
import random
from typing import List
from dataclasses import dataclass

# from src.lib.dataset_loaders.amazon_review_dataset_loader import SplitData

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

def deserialize_comments_jsonlist(filepath: str)\
        -> List[Comment]:
    """
    Deserializes file containing a list of JSON Amazon Review objects.
    Expects each review to be on a new line.

    Args:
        filepath (str): Path to file to deserialize.
    Returns:
        List[AmazonReview]: List of deserialized Amazon Reviews.
    """
    def deserialize(obj):
        propositions_raw = obj['propositions']
        propositions = [Proposition(id=prop['id'],
                                   type=prop['type'],
                                   text=prop['text'],
                                   reasons=prop['reasons'],
                                   evidence=prop['evidence'])
                        for prop in propositions_raw]
        return Comment(propositions=propositions, id=obj['id'])
    comments = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            comments.append(deserialize(json.loads(line)))
    return comments

# def serialize_comments_jsonlists(directory_path: str, comments: List[Comment], train: float, validation: float, test: float):
#     """
#     Serializes comments to traini, validation, and test under directory_path.
#     """
#     shuffled_comments = random.sample(comments, len(comments))
#     with open(split_filepath, 'w', encoding='utf-8') as f:
        


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    filepath = args[0]
    reviews = deserialize_comments_jsonlist(filepath)
    for review in reviews:
        print(review)
    print(len(reviews))
