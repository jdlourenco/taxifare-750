from taxifare.data import get_data, clean_df, holdout, BUCKET_NAME
from taxifare.pipeline import get_pipeline
from taxifare.utils import compute_rmse
import joblib
from google.cloud import storage

VALID_REGRESSORS = ["random_forest" , "linear_model"]

class Trainer():
    def __init__(self, **kwargs):
        self.regressor = kwargs.get("regressor", "random_forest")
        self.nrows = kwargs.get("nrows", 100)
        print(f"I'm going to use {self.regressor}")
        if not self.regressor in VALID_REGRESSORS:
            print("invalid regressor")
            raise Exception()
    
    def upload_to_gcp(self, filename):
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/myfirstmodel/{self.regressor}.joblib")
        blob.upload_from_filename(filename)

    def save_model(self):
        model_file_name = f"{self.regressor}.joblib"
        joblib.dump(self.pipeline, model_file_name)
        self.upload_to_gcp(model_file_name)
    
    def train(self):
        # get data
        df = get_data(nrows=self.nrows)

        # clean data
        df_clean = clean_df(df)

        # # holdout
        X_train, X_test, y_train, y_test = holdout(df_clean)

        # # get_pipeline
        self.pipeline = get_pipeline(model=self.regressor)

        # # train
        self.pipeline.fit(X_train, y_train)

        # evaluate
        y_pred = self.pipeline.predict(X_test)

        metric = compute_rmse(y_pred, y_test)
        print(f"rmse={metric}")
        # save
        self.save_model()

if __name__ == "__main__":
    for model in ["random_forest" , "linear_model"]:
        trainer = Trainer(regressor=model)
        trainer.train()
