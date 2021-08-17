from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet') 
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
from utils import parse
import pandas as pd
from matplotlib import pyplot as plt
import re,os
import numpy as np
from joblib import dump,load
from sklearn.model_selection import GridSearchCV
from utils import get_data



def Tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    porter_stemmer=nltk.PorterStemmer()
    words = [porter_stemmer.stem(word) for word in words]
    lemmatizer=WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.field]

class NumberSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[[self.field]]

class WordCounter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.field=None
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.DataFrame(X.apply(lambda x:len(x)))





def model_train(pipeline,X_train, X_test, y_train, y_test):
    """
    HYPER PARAMETER TUNING SECTION


    #hyper parameter gridsearch
    parameters = {
                'features__text__tfidf__ngram_range': [(1, 1), (1, 2)],
                'features__text__tfidf__use_idf': (True, False),
                'features__text__tfidf__max_df': [0.25, 0.5, 0.75, 1.0],
                'features__text__tfidf__max_features': [250, 500, 1000, None],
                'features__text__tfidf__stop_words': ('english', None),
                'features__text__tfidf__smooth_idf': (True, False),
                'features__text__tfidf__norm': ('l1', 'l2', None),
                'features__text__svd__n_components':[200,300,400]
                }


        #hyper param tuning
        grid = GridSearchCV(classifier, parameters, cv=2, verbose=1)
        grid.fit(X_train, y_train)
        preds = grid.predict(X_test)

        joblib.dump(grid.best_estimator_, 'model_xgboost.pkl')
    """

    # predict
    estimator=pipeline.fit(X_train, y_train)
    return estimator


def load_model(filename):
    estimator=load(filename)
    print('Model Loaded')
    return estimator

def get_pipeline(type=None):
    
    classifier = Pipeline([
        ('features', FeatureUnion([
            ('text', Pipeline([
                ('colext', TextSelector('Text')),
                #('tfidf', TfidfVectorizer(tokenizer=Tokenizer, stop_words=set(stopwords.words('english')),min_df=.0025, max_df=0.25, ngram_range=(1, 3))),
                ('tfidf', TfidfVectorizer(tokenizer=Tokenizer)),
                ('svd', TruncatedSVD(algorithm='randomized', n_components=300)),  # for XGB
                ])),
            ('wordcount', Pipeline([
                ('colext', TextSelector('Text')),
                ('wrcount', WordCounter()),
            ])),
        ])),
        ('clf', XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1, nthread=4, verbosity=1)),
        #    ('clf', RandomForestClassifier()),
    ],verbose=True)

    return classifier


if __name__=='__main__':


    filename='rate_helpfulness\dataset\All_Amazon_Review.json.gz'
    #filename='rate_helpfulness\dataset\All_Beauty.json.gz'
    helpful,sentences = get_data(filename,count=2000000)

    # load, and create train test sets
    #X = pd.DataFrame(list(zip(sentences,ratings)), columns=['Text','Rating'])
    X = pd.DataFrame(sentences, columns=['Text'])
    y = pd.DataFrame([x if x<1 else 1 for x in helpful],columns=['Pred'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print('Loaded Dataset')

    """#binary classification task
    pipeline=get_pipeline()
    estimator=model_train(pipeline,X_train, X_test, y_train, y_test)
    # publish
    if not os.path.exists('./published_models'):
        os.mkdir('./published_models')
    dump(estimator, './published_models/model_xgboost_v1.gz')
    print('Model Saved and Exited')"""

    # load model
    estimator=load_model('rate_helpfulness\published_models\model_xgboost_v1.gz')
    preds = estimator.predict(X_test)
    
    """# write predictions to file
    with open('./predicted_results.csv','w') as f:
        for x,y,y_p in zip(X_test.values,y_test.values,preds):
            f.write(f"{x}\t{y}\t{y_p}\n")"""
    
    print ("Accuracy:", accuracy_score(y_test, preds))
    print ("Precision:", precision_score(y_test, preds))
    print (classification_report(y_test, preds))
    print (confusion_matrix(y_test, preds))

    with open('./xgboost_models_stat.txt','w',newline='') as f:
        f.write('accuracy:'+str(accuracy_score(y_test, preds))+'\n')
        f.write('precision:'+str(precision_score(y_test, preds))+'\n')
        f.write('classification_report:\n'+str(classification_report(y_test, preds))+'\n')
        f.write('confusion matrix:\n'+str(confusion_matrix(y_test, preds))+'\n')

