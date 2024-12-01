from buisness_rules import get_model_by_buisness_rules, get_basic_recommendation_text
from fastapi import FastAPI, HTTPException
from model import Model
from request_model import RequestModel
import logging
import os

app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

path_to_config = os.path.join(os.path.dirname(__file__), 'config.yaml')
logger.info(f'{path_to_config=}')

model_catboost = Model(path_to_config, "catboost")
model_kmeans = Model(path_to_config, "kmeans")
model_decision_tree = Model(path_to_config, "decision_tree")
model_dict = {
    "catboost": model_catboost,
    "kmeans": model_kmeans,
    "decision_tree" : model_decision_tree
}
logger.info(f"available_models: {list(model_dict)}")


@app.post("/predict")
async def predict(data: RequestModel):
    try:
        model = get_model_by_buisness_rules(model_dict, data.context, is_multi_model_enabled=False)
        logger.info(f"model_name: {model}")
        prediction = model.predict(data)
        text = get_basic_recommendation_text(data.context, prediction)
        logger.info(f'{prediction=}')
        logger.info(f'{text=}')
        return {"prediction": prediction, "text": text}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during prediction: {str(e)}")
