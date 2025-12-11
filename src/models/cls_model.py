import joblib

class InvestmentClassifier:
    def __init__(self):
        self.model = joblib.load("../models/trained_models/xgb_classifier_model.joblib")
    
    def predict(self, X):
        return self.model.predict(X)[0]

    def predict_proba(self, X):
        return float(self.model.predict_proba(X)[0][1])