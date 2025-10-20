import random
from typing import List

from sklearn.model_selection import train_test_split

from data_loader.base_dataset_loader import BaseDatasetLoader


class DataLoader:
    def __init__(self, dataset_loaders: List[BaseDatasetLoader],
                 test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        """
        Initialize the DataLoader with a list of dataset loaders
        """
        self.dataset_loaders = dataset_loaders
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.data = {'train': [], 'val': [], 'test': []}

    def load_data(self, force_reload=False, all_messages=False):
        """
        Load and split the data for each dataset loader if split is not already defined
        :param force_reload: Flag to force reload the data, if false cache is used
        :param all_messages: Flag to include all messages, not just first sender/scammer messages
        """
        # Load and split each dataset individually
        for loader in self.dataset_loaders:
            loader.load_data(force_reload=force_reload, random_state=self.random_state, all_messages=all_messages)
            if 'all' in loader.data:
                all_data = loader.data['all']
                train_data, test_data = train_test_split(
                    all_data, test_size=self.test_size, random_state=self.random_state)
                train_data, val_data = train_test_split(
                    train_data, test_size=self.val_size / (1 - self.test_size), random_state=self.random_state)
                loader.data = {'train': train_data, 'val': val_data, 'test': test_data}
            else:
                pass
            self.data[loader.dataset_name] = loader.data

    def get_dataset_splits(self, dataset_name):
        """
        Get the train, val, and test splits for a given dataset
        """
        return self.data.get(dataset_name, {'train': [], 'val': [], 'test': []})