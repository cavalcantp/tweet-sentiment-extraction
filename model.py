import tensorflow as tf
from transformers import *
import numpy as np


# Hard Code some parameters of model
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
        Builds the roBERTa based model to extract sentiment from tweets.
        It uses a pre-trained roBERTa model and several convolutional and dropout layers
        to extract sentiment from the text.

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
        Runs inference. It recieves the tokenized inputs and returns information about the start
        and end indices of the sequence that represents the sentiment.

        Args:
            input_ids (np.ndarray): 
            attention_mask (np.ndarray): 
            token_type_ids (np.ndarray):

        Returns:
            str: Extracted sentiment of the tweet.
        """
        preds = self.model.predict([input_ids, attention_mask, token_type_ids], verbose=DISPLAY)

        return preds
        
