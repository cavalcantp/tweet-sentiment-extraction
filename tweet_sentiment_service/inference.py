from tweet_sentiment_service.model import SentimentModel
from tokenizers import ByteLevelBPETokenizer
import numpy as np
from typing import Tuple
from enum import Enum

class SentimentID(Enum):
    POSITIVE = 1313
    NEGATIVE = 2430
    NEUTRAL = 7974

SENTIMENT_ID = {"positive": 1313, "negative": 2430, "neutral": 7974}
MAX_LEN = 96


class SentimentExtractor:
    def __init__(self, weights_path: str, config_path: str):
        """
        Initializes ByteLevelBPETokenizer with given vocabulary and merges files, 
        and initializes SentimentModel with given configuration files.

        Args:
            weights_path (str): Path to the trained weights for the model.
            config_path (str): Path to the directory with configuration files for the RoBERTa base model and ByteLevelBPETokenizer.
        """
        vocab_path = config_path + 'vocab-roberta-base.json'
        merges_path = config_path + 'merges-roberta-base.txt'

        self.tokenizer = ByteLevelBPETokenizer(
            vocab=vocab_path, 
            merges=merges_path, 
            lowercase=True,
            add_prefix_space=True
        )

        self.model = SentimentModel(
            weights_path=weights_path,
            config_path=config_path,
        )

    def tokenize_and_mask(self, tweet: str, sentiment: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess data before feeding it to the model for prediction.

        Args:
            tweet (str): Tweet text.
            sentiment (str): sentiment expressed by tweet text.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: tokens, attention mask, token type ids (all 0 here)
        """
        input_ids = np.ones((1,MAX_LEN),dtype='int32')
        attention_mask = np.zeros((1,MAX_LEN),dtype='int32')
        token_type_ids = np.zeros((1,MAX_LEN),dtype='int32')

        # INPUT_IDS
        text = " "+" ".join(tweet.split())
        encoded_txt = self.tokenizer.encode(text)
        sentiment_tok = SentimentID[sentiment.upper()].value  
        input_ids[0,:len(encoded_txt.ids)+5] = [0] + encoded_txt.ids + [2,2] + [sentiment_tok] + [2]
        attention_mask[0,:len(encoded_txt.ids)+5] = 1

        return input_ids, attention_mask, token_type_ids



    def extract_sentiment(self, tweet: str, sentiment: str) -> str:
        """
        Extracts and returns sequence that expresses sentiment of given tweet.

        Args:
            tweet (str): Tweet text.
            sentiment (str): Sentiment expressed by tweet text.

        Returns:
            str: Sentiment sequence.
        """
        input_ids, attention_mask, token_type_ids = self.tokenize_and_mask(
            tweet=tweet, 
            sentiment=sentiment
        )

        predictions = self.model.predict(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )

        prediction_start = predictions[0]
        prediction_end = predictions[1]

        start_idx = np.argmax(prediction_start)
        end_idx = np.argmax(prediction_end)


        if start_idx > end_idx:
            return tweet
        
        text = " "+" ".join(tweet.split())
        encoded_txt = self.tokenizer.encode(text)
        selected_text = self.tokenizer.decode(encoded_txt.ids[start_idx-1:end_idx])

        return selected_text






