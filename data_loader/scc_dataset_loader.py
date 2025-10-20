import json
import os
import random
import re

from data_loader.base_dataset_loader import BaseDatasetLoader


class SCCDatasetLoader(BaseDatasetLoader):
    def __init__(self, data_dir: str, train_data_dir: str, test_data_dir: str, sample_size: int = None,
                 use_skipwords: bool = True, random_state: int = 42):
        """
        Initialize the SCC dataset loader
        """
        self.test_data_dir = test_data_dir
        random.seed(random_state)
        self.train_data_dir = train_data_dir
        super().__init__(data_dir=data_dir, label="scam", sample_size=sample_size, use_skipwords=use_skipwords)

    def process_data(self, all_messages=False):
        """
        Loads the SCC dataset from the train and test directory and processes by removing all non-email messages
        :param all_messages: Flag to include all messages, not just scammer messages
        :return: directory with train and test data splits
        """
        splits = {"train": self.train_data_dir, "test": self.test_data_dir}
        return_data = {}
        for split_name, split in splits.items():
            split_path = os.path.join(self.data_dir, split)
            conversations = []
            for root, _, files in os.walk(split_path):
                for filename in files:
                    if filename.endswith('.json'):
                        file_path = os.path.join(root, filename)
                        with open(file_path, 'r') as f:
                            convo = json.load(f)
                            messages = convo.get('messages', [])
                            if not all_messages:
                                # Only keep messages from Email
                                if any(msg.get("medium") in ["Instagram", "Telegram"] for msg in messages):
                                    continue
                                # Only keep scammers messages
                                messages = [msg for msg in messages if msg.get("is_inbound")]
                            messages = self.remove_file_description(messages)
                            messages = self.preprocess_messages(messages)
                            conversation = {
                                'messages': messages,
                                'label': self.label,
                                'dataset': self.dataset_name,
                            }
                            conversations.append(conversation)
            # Apply sampling if specified
            if self.sample_size and len(conversations) > self.sample_size:
                conversations = random.sample(conversations, self.sample_size)
            return_data[split_name] = conversations
        return return_data

    @staticmethod
    def remove_file_description(messages):
        """
        Remove the file description from the message body that was added by the dataset creators
        """
        for message in messages:
            if 'body' in message:
                text = message['body']
                if text == "" or text is None:
                    continue
                text = text.replace(
                    "This message contains files. If the description for a file does not make sense, ignore it."
                    "Here are descriptions of those files:\nDescription for file 1:",
                    ""
                )
                text = re.sub(r'Description for file \d+:', '', text)
                # Update the message text
                message['body'] = text
        return messages