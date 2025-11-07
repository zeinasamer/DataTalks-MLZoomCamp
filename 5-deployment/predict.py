import pickle

import uvicorn
from fastapi import FastAPI

app = FastAPI(title="churn-prediction")

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(costumer):
    result = pipeline.predict_proba([costumer])[0, 1]
    return float(result)

@app.post("/predict")
def predict(costumer):
    prob = predict_single(costumer)

    return{
        "churn_probability" : prob,
        "churn" : bool(prob >= 0.5)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)




