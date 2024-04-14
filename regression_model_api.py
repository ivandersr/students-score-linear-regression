from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib

app = FastAPI()

class RequestBody(BaseModel):
    study_hours: float

score_model = joblib.load('./regression_model.pkl')

@app.post('/predict')
def predict(data: RequestBody):
    input_feature = [[data.study_hours]]
    y_pred = score_model.predict(input_feature)[0][0].astype(int)

    return {
        'test_score': y_pred.tolist()
    }

