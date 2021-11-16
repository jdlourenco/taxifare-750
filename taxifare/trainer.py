from taxifare.data import get_data, clean_df, holdout
from taxifare.pipeline import get_pipeline
from taxifare.utils import compute_rmse
import joblib

VALID_REGRESSORS = ["random_forest" , "linear_model"]

class Trainer():
    def __init__(self, **kwargs):
        self.regressor = kwargs.get("regressor", "random_forest")
        self.nrows = kwargs.get("nrows", 100)
        print(f"I'm going to use {self.regressor}")
        if not self.regressor in VALID_REGRESSORS:
            print("invalid regressor")
            raise Exception()
    
    def save_model(self):
        model_file_name = f"{self.regressor}.joblib"
        joblib.dump(self.pipeline, model_file_name)
    
    def train(self):
        # get data
        df = get_data(nrows=self.nrows)
        # print(df.head())
        # print(df.shape)
        # clean data
        df_clean = clean_df(df)
        print(df_clean.shape)
        # holdout
        X_train, X_test, y_train, y_test = holdout(df_clean)
        # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        # get_pipeline
        self.pipeline = get_pipeline(model=self.regressor)
        # pipeline = get_pipeline()
        # train
        self.pipeline.fit(X_train, y_train)
        # pipeline.fit(X_train, y_train)
        # evaluate
        y_pred = self.pipeline.predict(X_test)
        # y_pred = pipeline.predict(X_test)
        metric = compute_rmse(y_pred, y_test)
        print(f"rmse={metric}")
        # save
        # self.save_model()
        # self.save_model(pipeline)
        self.save_model()


if __name__ == "__main__":
    for model in ["random_forest" , "linear_model"]:
        trainer = Trainer(regressor=model)
        trainer.train()
