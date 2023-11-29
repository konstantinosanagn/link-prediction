"""
Dataclasses and JSON (de)serialization for Amazon review dataset.
https://facultystaff.richmond.edu/~jpark/data/am2_emnlp2022.zip.
"""

import json
from typing import List, Dict
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
class AmazonReview:
    """
    Amazon review data object.
    """
    propositions: List[Proposition]
    overall: float
    vote: str
    verified: bool
    review_time: str
    review_id: str
    asin: str
    style: Dict[str, str]
    reviewer_name: str
    summary: str
    unix_review_time: int
    existing: int
    total: int

def deserialize_amazon_reviews_jsonlist(filepath: str)\
        -> List[AmazonReview]:
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
        return AmazonReview(propositions=propositions,
                            overall=obj['overall'],
                            vote=obj.get('vote', '0'),
                            verified=obj['verified'],
                            review_time=obj['reviewTime'],
                            review_id=obj['reviewID'],
                            asin=obj['asin'],
                            style=obj.get('style', None),
                            reviewer_name=obj['reviewerName'],
                            summary=obj['summary'],
                            unix_review_time=obj['unixReviewTime'],
                            existing=obj['existing'],
                            total=obj['total'])
    reviews = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            reviews.append(deserialize(json.loads(line)))
    return reviews

if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    filepath = args[0]
    reviews = deserialize_amazon_reviews_jsonlist(filepath)
    for review in reviews:
        print(review)
    print(len(reviews))
