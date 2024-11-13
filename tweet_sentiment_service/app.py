from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from tweet_sentiment_service.inference import SentimentExtractor
from pydantic import BaseModel, Field

WEIGHTS_PATH = "weights_final.h5"
CONFIG_PATH = "./config/"
# Load model once during startup, to preserve model between requests
sentiment_extractor = SentimentExtractor(weights_path=WEIGHTS_PATH, config_path=CONFIG_PATH)

class SentimentRequest(BaseModel):
    tweet: str = Field(description="The text of the tweet.")
    sentiment: str = Field(description="Sentiment expressed by tweet.")

app = FastAPI()

# Dependency function to provide the sentiment extractor
def get_sentiment_extractor():
    return sentiment_extractor

@app.post("/sentiment")
def extract_sentiment(
    request: SentimentRequest,
    sentiment_extractor: SentimentExtractor = Depends(get_sentiment_extractor)
    ) -> JSONResponse:

    if request.sentiment not in ["negative", "neutral", "positive"]:
        raise HTTPException(status_code=400, detail="Sentiment must be negative, neutral or positive.")

    try:
        selected_txt = sentiment_extractor.extract_sentiment(tweet=request.tweet, sentiment=request.sentiment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while extracting sentiment: {e}")


    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "selected_text": selected_txt},
        )

