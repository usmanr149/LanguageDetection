import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from data_preprocessing import preprocess

from urllib.parse import urlparse
import mlflow

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def run():

    df = pd.read_csv('input/Language Detection.csv')

    x_train, x_test, y_train, y_test, le, cv = preprocess(df)

    with mlflow.start_run():
        model = MultinomialNB()
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        ac = accuracy_score(y_test, y_pred)

        mlflow.log_param("ac", ac)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(model, "model")

if __name__ == '__main__':
    run()