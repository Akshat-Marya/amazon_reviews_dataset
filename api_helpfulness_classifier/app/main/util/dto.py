from flask_restx import Namespace, fields, SchemaModel
from werkzeug.datastructures import FileStorage

class TransactionDto:
    api = Namespace('transaction', description='user related operations')
    transaction = api.model('transaction', {
        'X':fields.String(required=True, description='review', cls_or_instance=fields.String)
    })