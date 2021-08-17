from .. import db
import datetime
from ..config import key


class Reviews(db.Model):
    """ House Model for storing house features """
    __tablename__ = "reviews"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    text = db.Column(db.String,nullable=False)
    
    transaction = db.relationship("Transactions",  backref="for_review")
    
    def __repr__(self):
        return f"<Reviews '{self.id}','{self.text}'>"