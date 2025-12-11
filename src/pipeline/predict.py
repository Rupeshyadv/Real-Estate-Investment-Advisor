from preprocess import Preprocessor
from models.cls_model import InvestmentClassifier
from models.reg_model import PriceRegressor

class PredictionPipeline:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.cls_model = InvestmentClassifier()
        self.reg_model = PriceRegressor()
    
    def predict(self, user_input):
        # transform the user input first 
        X = self.preprocessor.transform(user_input)
        
        # make predictions 
        investment_pred = self.cls_model.predict(X) # predict whether the property is a good investment or not 
        price_pred = self.reg_model.predict(X)  # predict the price of the property for 5 years from now 
        
        return investment_pred, price_pred