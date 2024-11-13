import pytest
from fastapi.testclient import TestClient
from tweet_sentiment_service.inference import SentimentExtractor
from tweet_sentiment_service.app import app, get_sentiment_extractor
from pytest_mock import MockerFixture

client = TestClient(app=app)

def test_extract_sentiment_sucess(mocker: MockerFixture) -> None:
    # Given
    tweet = "This is a test happy tweet."
    sentiment = "positive"
    expected_status = 200
    expected_output = "happy"
    # Patch the dependency to use a mock instead of the real extractor
    sentiment_extractor_mock = mocker.Mock()
    sentiment_extractor_mock.extract_sentiment.return_value = expected_output

    app.dependency_overrides[get_sentiment_extractor] = lambda: sentiment_extractor_mock


    # When
    response = client.post("/sentiment", json={"tweet": tweet, "sentiment": sentiment})

    # Then
    assert response.status_code == expected_status
    assert response.json() == {"status": "success", "selected_text": expected_output}

    # Remove override after test
    app.dependency_overrides.clear()

def test_extract_sentiment_ill_conditioned_request() -> None:
    # Given
    tweet = "This is a test happy tweet."
    sentiment = "ill conditioned sentiment"

    # When
    response = client.post("/sentiment", json={"tweet": tweet, "sentiment": sentiment})

    # Then
    assert response.status_code == 400
    assert response.json()["detail"] == "Sentiment must be negative, neutral or positive."

def test_extract_sentiment_inference_error(mocker: MockerFixture) -> None:
    # Given 
    tweet = "This is a test happy tweet."
    sentiment = "positive"
    # Patch the sentiment extractor to raise an exception
    def mock_get_sentiment_extractor():
        extractor_mock = mocker.Mock()
        extractor_mock.extract_sentiment.side_effect = Exception("test interval error")
        return extractor_mock
    
    app.dependency_overrides[get_sentiment_extractor] = mock_get_sentiment_extractor
    #mocker.patch("tweet_sentiment_service.inference.SentimentExtractor.extract_sentiment", side_effect=Exception("test interval error"))

    # When
    response = client.post("/sentiment", json={"tweet": tweet, "sentiment": sentiment})

    # Then
    assert response.status_code == 500
    assert response.json()["detail"] == f"Error while extracting sentiment: test interval error"
    app.dependency_overrides.clear()



