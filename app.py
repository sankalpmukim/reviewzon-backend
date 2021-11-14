from fastapi import FastAPI, Request
import uvicorn
import threading
from fastapi.middleware.cors import CORSMiddleware
from handler import handler
import pyrebase
import json
import random

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
    db.child("livedata").child(str(key)).set({0: init_data})
    db.child("livedata").child(str(key)).update(
        {1: "#00FF00:Initializing processes.."})
    count = 2
    threadsplit = threading.Thread(target=handler, args=(data, key, count))
    threadsplit.start()
    return {"unique_id": key}

if __name__ == '__main__':
    uvicorn.run(app='app:app', reload=True, debug=True)
