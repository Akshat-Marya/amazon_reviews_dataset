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
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
from utils import parse
import pandas as pd
from matplotlib import pyplot as plt
import re,os
from joblib import dump,load
from sklearn.model_selection import GridSearchCV

def Tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    porter_stemmer=nltk.PorterStemmer()
    words = [porter_stemmer.stem(word) for word in words]
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



def get_data(filename, count):
    ratings=[]
    helpful=[]
    sentences=[]
    for i,item in enumerate(parse(filename)):
        if i>=count:
            break
        ratings.append(item['overall'])
        try:
            sentences.append(item['reviewText'])
        except:
            sentences.append('')
        try:
            helpful.append(int(item['vote'].replace(',','')))
        except:
            helpful.append(0)

    print(f"helpful||count:{len(helpful)},total:{sum(helpful)},mean:{sum(helpful)/len(helpful)}")
    print(f"ratings||count:{len(ratings)},total:{sum(ratings)},mean:{sum(ratings)/len(ratings)}")

    return ratings,helpful,sentences


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
    # publish
    if not os.path.exists('./published_models'):
        os.mkdir('./published_models')
    dump(estimator, './published_models/model_xgboost.gz')
    print('Model Saved and Exited')

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
            ('Ratings', Pipeline([
                ('wordext', NumberSelector('Rating')),
                ('wscaler', StandardScaler()),
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


    filename='rate_helpfulness\dataset\All_Beauty.json.gz'
    ratings,helpful,sentences = get_data(filename, count=500)

    # load, and create train test sets
    #X = pd.DataFrame(list(zip(sentences,ratings)), columns=['Text','Rating'])
    X = pd.DataFrame(sentences, columns=['Text'])
    y=pd.DataFrame([x if x<1 else 1 for x in helpful],columns=['Pred'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print('Loaded Dataset')

    #binary classification task
    pipeline=get_pipeline()
    estimator=model_train(pipeline,X_train, X_test, y_train, y_test)
    
    # load model
    estimator=load_model('./published_models/model_xgboost.gz')
    preds = estimator.predict(X_test)
    
    print ("Accuracy:", accuracy_score(y_test, preds))
    print ("Precision:", precision_score(y_test, preds))
    print (classification_report(y_test, preds))
    print (confusion_matrix(y_test, preds))

