# Requirements
- Anaconda
- PyTorch
- BERT
- XgBoost

# Installation and QuickRun
- Set-up a conda environment and run app using following commands
```
cd /amazon_review_dataset_challenge/api_helpfulness_classifier
$ conda env create -f environment.yml
$ conda activate amazon_review_dataset_test
$ python create_db.py
$ python run.py
```
Note: Since the log level is debug it will not diplay the host and port its running on. it is `localhost:5000/api`

## API
Deployed at `localhost:5000/api`, don't forget to use the endpoint '/api'


# Run tests
```
cd /amazon_review_dataset_challenge
python manage_app.py test
```

# Database:SqlAlchemy
```
cd /amazon_review_dataset_challenge/api_helpfulness_classifier
python create_db.py
```

# Database:SqlAlchemy - Migrate and update
```
python manage_app.py db migrate
python manage_app.py db upgrade
```

# Logging
Saved as `api/helpfulness_classifier demo_alpha.log`  
Level: Debug

# Models
Two Models: XgBoost, BERT  
Located at: `rate_helpfulness/published_models`
