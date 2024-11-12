import tensorflow as tf
from transformers import *
import numpy as np
from tokenizers import ByteLevelBPETokenizer
from typing import Tuple
from enum import Enum

# Hard Code Parameters
class SentimentID(Enum):
    POSITIVE = 1313
    NEGATIVE = 2430
    NEUTRAL = 7974

MAX_LEN = 96
DISPLAY = 1

class SentimentModel:
    def __init__(self, weights_path: str, config_path: str):
        """
        Initializes the model and loads trained weights.

        Args:
            weights_path: Path to the trained weights for the model.
            config_path: Path to the configuration files for the RoBERTa base model.
        """
        self.max_len = MAX_LEN
        self.config_roberta_base = config_path + "config-roberta-base.json"
        self.pre_trained_roberta_weights = config_path + "pretrained-roberta-base.h5"

        self.model = self.build_model()
        self.model.load_weights(weights_path) 

    def build_model(self) -> tf.keras.Model: # type: ignore
        """
        Builds the RoBERTa based model to extract sentiment from tweets.

        Returns:
            tf.keras.Model: Compiled keras model.
        """
        ids = tf.keras.layers.Input((self.max_len,), dtype=tf.int32)
        att = tf.keras.layers.Input((self.max_len), dtype=tf.int32)
        tok = tf.keras.layers.Input((self.max_len), dtype=tf.int32)

        config = RobertaConfig.from_pretrained(self.config_roberta_base)
        bert_model = TFRobertaModel.from_pretrained(self.pre_trained_roberta_weights, config=config)
        x = bert_model(ids, attention_mask=att, token_type_ids=tok)

        x1 = tf.keras.layers.Dropout(0.1)(x[0]) 
        x1 = tf.keras.layers.Conv1D(1,1)(x1)
        x1 = tf.keras.layers.Flatten()(x1)
        x1 = tf.keras.layers.Activation('softmax')(x1)


        x2 = tf.keras.layers.Dropout(0.1)(x[0]) 
        x2 = tf.keras.layers.Conv1D(1,1)(x2)
        x2 = tf.keras.layers.Flatten()(x2)
        x2 = tf.keras.layers.Activation('softmax')(x2)


        model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        return model


    def predict(self, input_ids: np.ndarray, attention_mask: np.ndarray, token_type_ids: np.ndarray) -> str:
        """
        Runs inference. Extract and returns selected tokens from the input.

        Args:
            input_ids (np.ndarray): 
            attention_mask (np.ndarray): 
            token_type_ids (np.ndarray):

        Returns:
            str: Extracted sentiment of the tweet.
        """
        preds = self.model.predict([input_ids, attention_mask, token_type_ids], verbose=DISPLAY)

        return preds


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
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: 
        """
        input_ids = np.ones((1,MAX_LEN),dtype='int32')
        attention_mask = np.zeros((1,MAX_LEN),dtype='int32')
        token_type_ids = np.zeros((1,MAX_LEN),dtype='int32')

        # INPUT_IDS
        text = " "+" ".join(tweet.split())
        encoded_txt = self.tokenizer.encode(text)
        sentiment_tok = SentimentID[sentiment.upper()]    
        input_ids[0,:len(encoded_txt.ids)+5] = [0] + encoded_txt.ids + [2,2] + [sentiment_tok] + [2]
        attention_mask[0,:len(encoded_txt.ids)+5] = 1

        return input_ids, attention_mask, token_type_ids



    def extract_sentiment(self, tweet: str, sentiment: str) -> str:
        """
        Extracts and returns sequence that expresses sentiment of given tweet.

        Args:
            tweet (str): Tweet text.
            sentiment (str): sentiment expressed by tweet text.

        Returns:
            str: Sentiment (positive, negative or neutral).
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

        prediction_start = np.zeros((1,MAX_LEN))
        prediction_end = np.zeros((1,MAX_LEN))

        prediction_start += predictions[0]
        prediction_end += predictions[1]

        start_idx = np.argmax(prediction_start)
        end_idx = np.argmax(prediction_end)

        if start_idx > end_idx:
            return tweet
        
        text = " "+" ".join(tweet.split())
        encoded_txt = self.tokenizer.encode(text)
        selected_text = self.tokenizer.decode(encoded_txt.ids[start_idx-1:end_idx])

        return selected_text






