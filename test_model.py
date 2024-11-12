import pytest
from unittest.mock import MagicMock #, patch
import numpy as np
import tensorflow as tf
from transformers import RobertaConfig, TFRobertaModel
from tweet_sentiment_service.model import SentimentModel

MOCK_WEIGHTS_PATH = "/path/to/mock_weights.h5"
MOCK_CONFIG_PATH = "/path/to/mock_config/"


def test_initialization(mocker):
    # Mock the build_model method to return a tf.keras.Model
    build_model_mock = mocker.patch.object(SentimentModel, "build_model", return_value=tf.keras.Model())
    
    # Mock the load_weights method of the model instance
    load_weights_mock = mocker.patch.object(tf.keras.Model, "load_weights", MagicMock())
    model = SentimentModel(weights_path=MOCK_WEIGHTS_PATH, config_path=MOCK_CONFIG_PATH)
    
    # Check that SentimentModel instance was created successfully
    assert isinstance(model, SentimentModel)
    assert model.max_len == 96
    
    # Check that build_model and load_weights methods were called once
    build_model_mock.assert_called_once()
    load_weights_mock.assert_called_once()

def test_build_model(mocker):
    # Mock the load_weights method of the model instance
    load_weights_mock = mocker.patch.object(tf.keras.Model, "load_weights", MagicMock())
    # Create the sentiment model with mocked methods
    sentiment_model = SentimentModel(weights_path=MOCK_WEIGHTS_PATH, config_path="./config/")
    
    built_model = sentiment_model.build_model()

    assert len(built_model.inputs) == 3
    assert len(built_model.outputs) == 2

def test_predict_shape(mocker):
    # Mock the build_model method to return a tf.keras.Model
    build_model_mock = mocker.patch.object(SentimentModel, "build_model", return_value=tf.keras.Model())
    
    # Mock the load_weights method of the model instance
    load_weights_mock = mocker.patch.object(tf.keras.Model, "load_weights", MagicMock())
    sentiment_model = SentimentModel(weights_path=MOCK_WEIGHTS_PATH, config_path=MOCK_CONFIG_PATH)

    # Mock model.predict to avoid running actual inference
    mocker.patch.object(sentiment_model.model, 'predict', return_value=[np.random.rand(1, 96), np.random.rand(1, 96)])

    # Create dummy input data with appropriate shapes
    input_ids = np.random.randint(0, 100, (1, 96))
    attention_mask = np.random.randint(0, 2, (1, 96))
    token_type_ids = np.zeros((1, 96))

    # Run prediction and check output
    preds = sentiment_model.predict(input_ids, attention_mask, token_type_ids)
    assert isinstance(preds, list)
    assert len(preds) == 2
    assert preds[0].shape == (1, 96)
    assert preds[1].shape == (1, 96)

    
