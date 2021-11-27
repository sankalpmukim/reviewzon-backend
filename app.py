from fastapi import FastAPI, Request
import uvicorn
import threading
from fastapi.middleware.cors import CORSMiddleware
from handler import organizer
import pyrebase
import json
import random
import time
from sentiment_analysis.LocalSentimentAnalysis import SentimentAnalysis_Local
from sentiment_analysis.LiveSentimentAnalysis import SentimentAnalysis_Live
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("config.json") as config_file:
    config = json.load(config_file)

firebase = pyrebase.initialize_app(config['firebaseConfig'])
db = firebase.database()


@app.post("/")
async def root(request: Request):
    data = await request.json()
    livedata = db.child("livedata").get()
    key = random.randint(10000000, 99999999)
    while key in livedata.val().keys():
        key = random.randint(10000000, 99999999)
    init_data = "#67FF0F:Data received"
    db.child("livedata").child(str(key)).set(
        {0: {'color': '#00FF00', "message": "Data received", "error": False, "end": False}})
    db.child("livedata").child(str(key)).update(
        {1: {'color': '#00FF00', "message": "Initializing processes..", "error": False, "end": False}})
    count = 2
    logger = [count, firebase, key]
    threadsplit = threading.Thread(
        target=organizer, args=(data, logger, data['doExperiment']))
    threadsplit.start()
    return {"unique_id": key}


@app.post("/checksentiment")
async def root(request: Request):
    data = await request.json()
    # lsa = LocalSentimentAnalysis()
    sentiment_lst = []
    strings = data['text'].replace("check-sentiment ", "").split(",")
    if data['data']['train']['mode'] == 1:
        lsa = SentimentAnalysis_Local(origin='api', key=data['uniqueKey'])
        sentiment_lst = str(lsa.check_sentiment(strings))
    else:
        lsa = SentimentAnalysis_Live(origin='api', key=data['uniqueKey'])
        sentiment_lst = str(lsa.check_sentiment(strings))
    return {"color": "#00FF00", "message": sentiment_lst}

if __name__ == '__main__':
    uvicorn.run(app='app:app')
