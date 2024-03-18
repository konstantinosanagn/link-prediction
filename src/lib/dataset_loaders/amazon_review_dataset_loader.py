"""
Module containing functionality for loading comments and propositions
and splitting into train/validation/test.
"""
import random
from typing import Any, List, TypedDict, Union

from .park_data_parser import (Proposition, Comment,
                               deserialize_comments_jsonlist)

class SplitData(TypedDict):
    """
    Dict to store split data.
    """
    train: List[Any]
    validation: List[Any]
    test: List[Any]
    examples: List[Any]

class DatasetLoader:
    """
    Class to deserialize and split data.
    """
    def __init__(
        self,
        train_path: str,
        validation_path: str,
        test_path: str,
        use_propositions: bool,
        seed: int,
    ) -> None:
        """
        Args:

        Returns:

        """
        random.seed(seed)
        self._seed = seed
        self._loaded_train_data = self.load_data(train_path, use_propositions)
        # shuffle train data
        self._loaded_train_data = random.sample(self._loaded_train_data,
                                                len(self._loaded_train_data))
        self.validation_data = self.load_data(validation_path, use_propositions)
        self.test_data = self.load_data(test_path, use_propositions)

    def get_splits(
        self,
        num_examples: int
    ) -> SplitData:
        """
        Retrieves the examples, train, validation, and test splits of the loaded dataset.
        Args:
            num_examples (int): Number of examples to include in the split data.
        Returns:
            SplitData: Dataset split into examples, train, validation, test
        """
        if num_examples > 0.5 * len(self._loaded_train_data):
            raise ValueError("num_examples cannot be more than half the training set.")
        examples = self._loaded_train_data[:num_examples]
        return {
                "examples": examples,
                "train": self._loaded_train_data[num_examples:],
                "validation": self.validation_data,
                "test": self.test_data
        }

    def load_data(
        self,
        dataset_path: str,
        use_propositions: bool
    ) -> List[Union[Proposition, Comment]]:
        """
        Args:

        Returns:
        """
        orig_dataset = deserialize_comments_jsonlist(dataset_path)
        return orig_dataset if not use_propositions \
                else [prop for review in orig_dataset for prop in review.propositions]

def _test(path_to_dataset_dir):
    dataset_loader = DatasetLoader(path_to_dataset_dir, True, 1)
    data = dataset_loader.get_splits(0)
    print(data)

if __name__ == '__main__':
    import fire
    # Some test
    fire.Fire(_test)
