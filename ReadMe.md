# Modules
 - Flask-restx API: api_helpfulness_classifier
 - Classification Module: ratehelpfulness
# System Specs
 - Windows 10

# Requirements, Installation and QuickRun 
Please follow the steps mentioned in the `api_helpfullness_classifier/ReadMe.md`. The below are a copy of the same"

# Installation and QuickRun
- Set-up a conda environment and run app using following commands
```
cd /medchart_challenge/api_helpfulness_classifier
$ conda env create -f environment.yml
$ conda activate medchart_test
$ python create_db.py
$ python run.py
```
Note: Since the log level is debug it will not diplay the host and port its running on.

## API
Deployed at `localhost:5000/api`, don't forget to use the endpoint '/api'

# Directory Structure
Please do not change the directory structure from initial:

- medchart-challenge
    - api_helpfulness_classifier
    - rate_helpfulness

