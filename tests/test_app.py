import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from tweet_sentiment_service.inference import SentimentExtractor
from tweet_sentiment_service.app import app

client = TestClient(app=app)

def test_extract_sentiment_sucess(mocker):
    # Given
    tweet = "This is a test happy tweet"
    sentiment = "positive"
    expected_status = 200
    expected_output = "happy"
    sentiment_extractor_mock = mocker.patch("tweet_sentiment_service.app.SentimentExtractor")
    sentiment_extractor_mock.return_value.extract_sentiment.return_value = expected_output


    # When
    response = client.post("/sentiment", json={"tweet": tweet, "sentiment": sentiment})

    # Then
    assert response.status_code == expected_status
    assert response.json() == {"status": "success", "selected_text": expected_output}

def test_extract_sentiment_ill_conditioned_request():
    # Given
    tweet = "This is a test happy tweet."
    sentiment = "ill conditioned sentiment"

    # When
    response = client.post("/sentiment", json={"tweet": tweet, "sentiment": sentiment})

    # Then
    assert response.status_code == 400
    assert response.json()["detail"] == "Sentiment must be negative, neutral or positive."

def test_extract_sentiment_inference_error(mocker):
    # Given 
    tweet = "This is a test happy tweet."
    sentiment = "positive"
    mocker.patch("tweet_sentiment_service.inference.SentimentExtractor.extract_sentiment", side_effect=Exception("test interval error"))

    # When
    response = client.post("/sentiment", json={"tweet": tweet, "sentiment": sentiment})

    # Then
    assert response.status_code == 500
    assert response.json()["detail"] == f"Error while extracting sentiment: test interval error"



