import joblib 

class PriceRegressor:
    def __init__(self):
        self.model = joblib.load("../models/trained_models/xgb_regressor_model.joblib")
    
    def predict(self, X):
        return self.model.predict(X)[0]