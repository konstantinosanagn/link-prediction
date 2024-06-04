"""
Dataclasses and JSON deserialization for review dataset.
"""

import json
import os
import random
from typing import List, Tuple
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

def generate_proposition_pairs(comment: Comment) -> List[Tuple[Proposition, List[Proposition]]]:
    """
    Generate pairs of the ith proposition with the five propositions before it
    and the five propositions after it.

    Args:
        comment (Comment): The comment object containing propositions.

    Returns:
        List[Tuple[Proposition, List[Proposition]]]: A list of tuples, where each tuple contains
        a proposition and a list of up to 10 surrounding propositions.
    """
    pairs = []
    propositions = comment.propositions
    for i in range(len(propositions)):
        surrounding = propositions[max(0, i-5):i] + propositions[i+1:i+6]
        pairs.append((propositions[i], surrounding))
    return pairs

def print_proposition_pairs(pairs: List[Tuple[Proposition, List[Proposition]]]):
    """
    Print the proposition pairs in a clear, formatted way.

    Args:
        pairs (List[Tuple[Proposition, List[Proposition]]]): List of proposition pairs to print.
    """
    for proposition, surrounding in pairs:
        print(f"Proposition ID: {proposition.id} - {proposition.type}")
        print(f"  Text: {proposition.text}")
        print("  Surrounding Propositions:")
        for s_prop in surrounding:
            print(f"    ID: {s_prop.id} - {s_prop.type}")
            print(f"      Text: {s_prop.text}")
        print("\n" + "="*80 + "\n")

def print_comments_with_pairs(comments: List[Comment]):
    """
    Print each proposition with its surrounding propositions for all comments.

    Args:
        comments (List[Comment]): List of comments to process.
    """
    for comment in comments:
        pairs = generate_proposition_pairs(comment)
        print(f"Comment ID: {comment.id}")
        for proposition, surrounding in pairs:
            print(f"\nProposition ID: {proposition.id} - {proposition.text}")
            print("Surrounding Propositions:")
            for s_prop in surrounding:
                print(f"  ID: {s_prop.id} - {s_prop.text}")
        print("\n" + "="*80 + "\n")

def create_reasons_map(comments: List[Comment]) -> dict:
    reasons_map = {}
    for comment in comments:
        comment_id = comment.id
        for prop in comment.propositions:
            proposition_id = prop.id
            reasons = prop.reasons if prop.reasons else []
            all_reasons = []
            for reason in reasons:
                if '_' in reason:
                    start, end = map(int, reason.split('_'))
                    all_reasons.extend(range(start, end + 1))
                else:
                    all_reasons.append(int(reason))
            reasons_map[(comment_id, proposition_id)] = all_reasons
    return reasons_map


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    filepath = args[0]
    reviews = deserialize_comments_jsonlist(filepath)
    for review in reviews:
        print(review)
    print(len(reviews))
