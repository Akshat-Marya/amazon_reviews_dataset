import os
import unittest

from flask_migrate import Migrate, MigrateCommand
from flask_script import Manager

from app.main import create_app, db
from app.main.model import transaction_model,review_model
from app import blueprint

from logging.config import fileConfig

fileConfig("logging.cfg")
# get config from env for 'prod', 'dev' or 'test', if none then defaults to 'dev'
app = create_app(os.getenv('APP_ENV') or 'dev')
app.register_blueprint(blueprint)
app.app_context().push()

manager = Manager(app)

migrate = Migrate(app, db)

manager.add_command('db', MigrateCommand)
#expose commands to be accuess via terminal
@manager.command
def run():
    app.run()

@manager.command
def test():
    """Runs the unit tests."""
    tests = unittest.TestLoader().discover('app/test', pattern='test*.py')
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    if result.wasSuccessful():
        return 0
    return 1

if __name__ == '__main__':
    manager.run()