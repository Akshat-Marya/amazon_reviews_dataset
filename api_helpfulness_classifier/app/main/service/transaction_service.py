import uuid
import datetime
import pandas as pd

from ..config import key


from app.main import db
from app.main.model.review_model import Reviews
from app.main.model.transaction_model import Transactions

import os
from joblib import load
#from model_xgboost import load_model, TextSelector, NumberSelector, WordCounter, Tokenizer
from model_bert import predict_review


def save_new_transaction(data):
    
    try: 
        review = Reviews(
                text = data['X']
        )
        transaction = Transactions(for_review=review)

        db.session.add(review)
        db.session.add(transaction)
        db.session.commit()


        """
        Model:XgBoost
        estimator = load_model(os.path.abspath(f"../rate_helpfulness/published_models/model_xgboost_v1.gz"))
        pred = estimator.predict(pd.DataFrame([data['X']],columns=['Text']))
        pred = 'not helpful' if pred[0]==0 else 'helpful' 
        """

        # Model: BERT
        pred = predict_review(data['X'], model_file=os.path.abspath('../rate_helpfulness/published_models/bert_model_state_v1.bin'))
        pred='not helpful' if pred[0]==0 else 'helpful'
        response_object = {
            'status':'success',
            'classification': f'{pred}',
        }
    except Exception as e:
        response_object = {
            'status':'fail',
            'message': 'Input is invalid, formatting issues with input',
        }
        return response_object, 502
    return response_object, 200

