import os
import unittest

from app.main import create_app, db
from app import blueprint

from logging.config import fileConfig
#from model_xgboost import load_model, TextSelector, NumberSelector, WordCounter, load_model, Tokenizer

# get config from env for 'prod', 'dev' or 'test', if none then defaults to 'dev'
app = create_app(os.getenv('APP_ENV') or 'dev')
fileConfig("logging.cfg")
app.register_blueprint(blueprint)
app.app_context().push()

if __name__=='__main__':
    app.run()