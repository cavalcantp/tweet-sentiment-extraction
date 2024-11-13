import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import MagicMock
from tokenizers import Encoding
from tweet_sentiment_service.inference import SentimentExtractor, SentimentID
from tweet_sentiment_service.model import SentimentModel
from pytest_mock import MockerFixture


MOCK_WEIGHTS_PATH = "/path/to/mock_weights.h5"
MOCK_CONFIG_PATH = "/path/to/mock_config/"
MAX_LEN = 96


def test_initialization(mocker: MockerFixture) -> None:
    # Given

    # Mock the build_model method to return a tf.keras.Model
    build_model_mock = mocker.patch.object(SentimentModel, "build_model", return_value=MagicMock())
    # Mock the load_weights method of the model instance
    load_weights_mock = mocker.patch.object(tf.keras.Model, "load_weights", MagicMock())
    # Mock tokenizer
    tokenizer_mock = mocker.patch("tweet_sentiment_service.inference.ByteLevelBPETokenizer", return_value=MagicMock())
    # Mock SentimentModel init
    model_init_mock = mocker.patch.object(SentimentModel, "__init__", return_value=None)
    
    # When

    sentiment_extractor = SentimentExtractor(weights_path=MOCK_WEIGHTS_PATH, config_path=MOCK_CONFIG_PATH)

    # Then

    tokenizer_mock.assert_called_with(
        vocab=f"{MOCK_CONFIG_PATH}vocab-roberta-base.json", 
        merges=f"{MOCK_CONFIG_PATH}merges-roberta-base.txt", 
        lowercase=True, 
        add_prefix_space=True
    )

    model_init_mock.assert_called_once_with(weights_path=MOCK_WEIGHTS_PATH, config_path=MOCK_CONFIG_PATH)


def test_tokenize_and_mask(mocker: MockerFixture) -> None:
    # Given
    tweet = "This is a happy test tweet."
    sentiment = "positive"
    expected_ids = [10, 11, 12, 15]
    expected_attention_mask = np.zeros((1,MAX_LEN),dtype='int32')
    expected_attention_mask[0,:9] = 1
    expected_token_type_ids = np.zeros((1,MAX_LEN),dtype='int32')
    expected_input_ids = np.ones((1,MAX_LEN),dtype='int32')
    expected_input_ids[0,:9] = [0] + expected_ids + [2,2] + [SentimentID.POSITIVE.value] + [2]

    # Mocks
    build_model_mock = mocker.patch.object(SentimentModel, "build_model", return_value=MagicMock())
    load_weights_mock = mocker.patch.object(tf.keras.Model, "load_weights", MagicMock())
    tokenizer_mock = mocker.patch("tweet_sentiment_service.inference.ByteLevelBPETokenizer", return_value=MagicMock())
    encode_mock = MagicMock()
    encode_mock.ids = expected_ids
    tokenizer_mock.return_value.encode.return_value = encode_mock
    sentiment_extractor = SentimentExtractor(MOCK_WEIGHTS_PATH, MOCK_CONFIG_PATH)

    # When
    input_ids, attention_mask, token_type_ids = sentiment_extractor.tokenize_and_mask(tweet, sentiment)

    # Then
    assert (input_ids == expected_input_ids).all()
    assert (attention_mask == expected_attention_mask).all()
    assert (token_type_ids == expected_token_type_ids).all()


@pytest.mark.parametrize(
        ("tweet", "sentiment", "start_idx", "end_idx", "expected_output"),
        [
            ("test sad", "negative", 15, 11, "test sad"), # return tweet
            ("This is a happy test tweet.", "positive", 7, 11, "happy"), # return selected text
        ]
)
def test_extract_sentiment(mocker: MockerFixture, tweet: str, sentiment: str, start_idx: int, end_idx: int, expected_output: str) -> None:
    # Given
    expected_input_ids = np.zeros((1,MAX_LEN),dtype='int32')
    expected_attention_mask = np.zeros((1,MAX_LEN),dtype='int32')
    expected_token_type_ids = np.zeros((1,MAX_LEN),dtype='int32')
    expected_preds = np.zeros((2,MAX_LEN),dtype='int32')
    expected_preds[0, start_idx] = 100
    expected_preds[1, end_idx] = 100

    # Mocks
    build_model_mock = mocker.patch.object(SentimentModel, "build_model", return_value=MagicMock())
    load_weights_mock = mocker.patch.object(tf.keras.Model, "load_weights", MagicMock())
    tokenizer_mock = mocker.patch("tweet_sentiment_service.inference.ByteLevelBPETokenizer", return_value=MagicMock())

    tokenize_and_mask_mock = mocker.patch.object(SentimentExtractor, "tokenize_and_mask")
    tokenize_and_mask_mock.return_value = (expected_input_ids, expected_attention_mask, expected_token_type_ids)

    encode_mock = MagicMock()
    encode_mock.ids = [10, 11, 12]
    tokenizer_mock.return_value.encode.return_value = encode_mock
    tokenizer_mock.return_value.decode.return_value = expected_output

    model_prediction_mock = mocker.patch.object(SentimentModel, "predict", return_value=expected_preds)

    # When
    sentiment_extractor = SentimentExtractor(MOCK_WEIGHTS_PATH, MOCK_CONFIG_PATH)
    selected_text = sentiment_extractor.extract_sentiment(tweet, sentiment)

    # Then
    tokenize_and_mask_mock.assert_called_with(tweet=tweet, sentiment=sentiment)
    model_prediction_mock.assert_called_once()
    
    assert selected_text == expected_output


