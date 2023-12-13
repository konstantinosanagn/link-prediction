"""
Module containing functionality for loading comments and propositions
and splitting into train/validation/test.
"""
import random
import os
import re
from typing import Any, List, TypedDict, Optional, Tuple, Union
from sklearn.model_selection import train_test_split

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
        dataset_path: str,
        use_propositions: bool,
        seed: int,
        train: Optional[float] = None,
        validation: Optional[float] = None,
        test: Optional[float] = None
    ) -> None:
        """
        Args:

        Returns:

        """
        self._train = train
        self._validation = validation
        self._test = test

        random.seed(seed)
        self._loaded_train_data, self.test_data = self.load_data_from_dir(dataset_path,
                                                                          use_propositions)
        # shuffle train data
        self._loaded_train_data = random.sample(self._loaded_train_data,
                                                len(self._loaded_train_data))
        # rescale _train and _validation split proportions to be out of 1
        # since test data is set aside
        self._train = train / (train + validation)
        self._validation = 1 - self._train

    def get_splits(
        self,
        num_examples: int
    ) -> SplitData:
        """
        Args:

        Returns:

        """
        if num_examples > 0.5 * len(self._loaded_train_data):
            raise ValueError("num_examples cannot be more than half the training set.")
        examples = self._loaded_train_data[:num_examples]
        train_data, validation_data = train_test_split(self._loaded_train_data[num_examples:],
                                                       test_size=self._validation,
                                                       train_size=self._train)
        return {
                "examples": examples,
                "train": train_data,
                "validation": validation_data,
                "test": self.test_data
        }

    def _deserialize_helper(
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

    def load_data_from_dir(
        self,
        dataset_dir_path: str,
        use_propositions: bool
    ) -> Tuple[List[Union[Proposition, Comment]], List[Union[Proposition, Comment]]]:
        """

        Args:

        Returns:
        """
        # search for all jsonlists in dataset_dir_path
        jsonlists = []
        for filename in os.listdir(dataset_dir_path):
            full_path = os.path.join(dataset_dir_path, filename)
            if os.path.isfile(full_path) and os.path.splitext(full_path)[-1] == '.jsonlist':
                jsonlists.append(full_path)

        if not jsonlists:
            raise ValueError(f"{dataset_dir_path} does not contain jsonlists")

        train_data = None
        test_data = None
        if len(jsonlists) == 1:
            # split single file into train/test
            data = self._deserialize_helper(jsonlists[0], use_propositions)
            train_data, test_data = train_test_split(data,
                                                     self._test, self._train + self._validation,
                                                     self.seed)
        else:
            # first file containing 'train' is training list
            # first file containing 'test' is test list
            for filepath in jsonlists:
                if not train_data and \
                        re.search('train', os.path.basename(filepath), re.IGNORECASE) is not None:
                    train_data = self._deserialize_helper(filepath, use_propositions)
                if not test_data and \
                        re.search('test', os.path.basename(filepath), re.IGNORECASE) is not None:
                    test_data = self._deserialize_helper(filepath, use_propositions)
                if train_data and test_data:
                    break
        return train_data, test_data
