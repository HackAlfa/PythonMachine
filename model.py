import logging
import pickle
import yaml
import pandas as pd
from request_model import RequestModel
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Model:
    def __init__(self, config_path: str):
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            try:
                with open(config['path_to_model'], 'rb') as f:
                    self.model = pickle.load(f)
            except:
                logger.error(f'{traceback.format_exc()}')
                raise RuntimeError(
                    f'There is no model by path: {config["path_to_model"]}')
        except:
            logger.error(f'{traceback.format_exc()}')            
            raise RuntimeError(
                f'There is no config file by path: {config_path}')

    def predict(self, data: RequestModel):
        input_data = {
            "clientId": [data.clientId],
            "organizationId": [data.organizationId],
            "segment": [data.segment],
            "role": [data.role],
            "organizations": [data.organizations],
            "currentMethod": [data.currentMethod],
            "mobileApp": [data.mobileApp],
            "signatures": [
                {
                    "common": data.signatures.common,
                    "special": data.signatures.special,
                }
            ],
            "availableMethods": [data.availableMethods],
            "claims": [data.claims],
            "context": [data.context],
        }
        df = pd.DataFrame(input_data)
        logging.info(f'df before preprocessing: {df.values}')
        df[['common_mobile', 'common_web', 'special_mobile', 'special_web']] = pd.json_normalize(df['signatures']).set_index(df.index)
        features_to_drop = ['clientId', 'organizationId', 'currentMethod', 'signatures', 'availableMethods']
        df = df.drop(features_to_drop, axis=1)
        logging.info(f'df after preprocessing: {df.values}')
        selected_cols = [
            'segment',
            'role',
            'organizations',
            'mobileApp',
            'claims',
            'common_mobile',
            'common_web',
            'special_mobile',
            'special_web',
            'context'
        ]
        df = df[selected_cols]
        return self.model.predict(df)[0][0]
