import logging

from fastapi import FastAPI, HTTPException
from request_model import RequestModel
from model import Model

app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

path_to_config = './config.yaml'
logger.info(f'{path_to_config=}')
model = Model(path_to_config)


@app.get("/predict")
async def predict(data: RequestModel):
    try:
        prediction = model.predict(data)
        logger.info(f'{prediction=}')
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during prediction: {str(e)}")
