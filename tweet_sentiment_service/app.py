from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from tweet_sentiment_service.inference import SentimentExtractor
from pydantic import BaseModel, Field

WEIGHTS_PATH = "weights_final.h5"
CONFIG_PATH = "./config/"

class SentimentRequest(BaseModel):
    tweet: str = Field(description="The text of the tweet.")
    sentiment: str = Field(description="Sentiment expressed by tweet.")

app = FastAPI()

@app.post("/sentiment")
def extract_sentiment(request: SentimentRequest) -> JSONResponse:

    if request.sentiment not in ["negative", "neutral", "positive"]:
        raise HTTPException(status_code=400, detail="Sentiment must be negative, neutral or positive.")

    try:
        sentiment_extractor = SentimentExtractor(weights_path=WEIGHTS_PATH, config_path=CONFIG_PATH)
        selected_txt = sentiment_extractor.extract_sentiment(tweet=request.tweet, sentiment=request.sentiment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while extracting sentiment: {e}")


    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "selected_text": selected_txt},
        )

