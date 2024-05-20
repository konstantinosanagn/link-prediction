from typing import List, Union

from .amazon_review_dataset_loader import DatasetLoader
from typing import List, Union
from .park_data_parser import Proposition, Comment, deserialize_comments_jsonlist

class EnhancedDatasetLoader(DatasetLoader):
    def __init__(
        self,
        train_path: str,
        validation_path: str,
        test_path: str,
        use_propositions: bool,
        seed: int,
    ) -> None:
        super().__init__(train_path, validation_path, test_path, use_propositions, seed)

    def get_surrounding_propositions(
        self,
        propositions: List[Proposition],
        index: int,
        window_size: int = 5
    ) -> List[Union[str, None]]:
        """
        Get surrounding propositions text for a given index.
        
        Args:
            propositions (List[Proposition]): List of propositions.
            index (int): Index of the target proposition.
            window_size (int): Number of surrounding propositions to include on each side.
        
        Returns:
            List[Union[str, None]]: List of surrounding propositions' texts, None if out of bounds.
        """
        surrounding = []
        for offset in range(-window_size, window_size + 1):
            if offset == 0:
                continue
            new_index = index + offset
            if 0 <= new_index < len(propositions):
                surrounding.append(propositions[new_index].text)
            else:
                surrounding.append(None)
        return surrounding

    def load_data_with_context(
        self,
        dataset_path: str,
        use_propositions: bool,
        window_size: int = 5
    ) -> List[Union[Proposition, Comment]]:
        orig_dataset = deserialize_comments_jsonlist(dataset_path)
        data_with_context = []
        for comment in orig_dataset:
            for idx, prop in enumerate(comment.propositions):
                surrounding_props = self.get_surrounding_propositions(comment.propositions, idx, window_size)
                prop.surrounding = surrounding_props
                data_with_context.append(prop)
        return data_with_context

    def load_data(
        self,
        dataset_path: str,
        use_propositions: bool
    ) -> List[Union[Proposition, Comment]]:
        return self.load_data_with_context(dataset_path, use_propositions)
