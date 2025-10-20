import hashlib
import os
import pickle
import nltk
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk_stopwords = set(stopwords.words('english'))
SKIPWORDS = {"cindy", "jenkins", "enron", "u"}
SKIPWORDS.update(nltk_stopwords)

class BaseDatasetLoader:
    def __init__(self, data_dir: str, label: str, sample_size: int = None, dataset_name: str = None,
                 use_skipwords: bool = True):
        """
        Initialize the dataset loader.
        """
        self.data_dir = data_dir
        self.label = label
        self.sample_size = sample_size
        self.dataset_name = self.__class__.__name__ if dataset_name is None else dataset_name
        self.data = {}
        self.num_conversations = 0
        self.num_messages = 0
        self.conversation_lengths = []
        self.vocab = set()
        self.use_skipwords = use_skipwords

        self.cache_dir = os.path.join('data_loading_cache', self.dataset_name)
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_data(self, force_reload=False, random_state=42, all_messages=False):
        """Method to load and process data."""
        cache_file = self.get_cache_filename(random_state, all_messages)
        if force_reload or not os.path.exists(cache_file):
            self.data = self.process_data(all_messages=all_messages)
            self.save_to_cache(cache_file)
        else:
            self.data = self.load_from_cache(cache_file)

    def process_data(self, all_messages=False):
        """Process data and return it. To be implemented by subclasses."""
        raise NotImplementedError

    def get_cache_filename(self, random_state=42, all_messages=False):
        """Generate a unique cache filename based on configuration."""
        config_str = f"{self.data_dir}_{self.label}_{self.sample_size}_{self.use_skipwords}_{random_state}_{all_messages}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        filename = os.path.join(self.cache_dir, f"data_{config_hash}.pkl")
        return filename

    def save_to_cache(self, cache_file):
        """Save data to cache."""
        with open(cache_file, 'wb') as f:
            pickle.dump(self.data, f)

    def load_from_cache(self, cache_file):
        """Load data from cache."""
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data

    def preprocess_messages(self, messages):
        """Preprocess a list of messages."""
        filtered_messages = [msg for msg in messages if msg.get('body') is not None and msg.get('body') != '']
        message_bodies = [msg['body'] for msg in filtered_messages]
        for msg in filtered_messages:
            msg['raw_body'] = msg['body']
        processed_bodies = self.preprocess_message_bodies(message_bodies)
        for idx, msg in enumerate(filtered_messages):
            msg['body'] = processed_bodies[idx]
        return filtered_messages

    def preprocess_message_bodies(self, messages):
        """Preprocess a list of message bodies."""
        lemmatizer = WordNetLemmatizer()
        if self.use_skipwords:
            stop_words = set(SKIPWORDS)
        else:
            stop_words = set(stopwords.words('english'))
        processed_messages = []

        for message in messages:
            tokens = nltk.word_tokenize(message.lower())
            tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
            tokens = [token for token in tokens if token not in stop_words]
            processed_message = ' '.join(tokens)
            processed_messages.append(processed_message)

        return processed_messages