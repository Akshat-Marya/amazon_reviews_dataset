  
import unittest
import json
import sys
sys.path.append('./')
from app.test.base import BaseTestCase

class TestDataUpload(BaseTestCase):

    def test_house_price_prediction(self):
            """ Test for free text review prediction """
            with self.client:
                response = self.client.post(
                    '/api/predict-helpfulness',
                    data=json.dumps({'X':'this is a review for testing'}),
                    content_type='application/json'
                )
                self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()