from app.data_processing.data_cleaning.prediction_service_two import get_prediction
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class InputData(BaseModel):
    age: float
    sex: float
    cp: float
    restbp: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.get("/get-prediction")
async def firstRoute(data: InputData):
    prediction_result = get_prediction(data)
    return {"prediction": prediction_result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
