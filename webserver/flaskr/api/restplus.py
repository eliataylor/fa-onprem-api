import logging

from flask_cors import CORS
from flask_restx import Api

log = logging.getLogger(__name__)

api = Api(version='1.0', title='Flexible Vision Capture and Detection API',
          description='A Image capture and object detection api')
cors = CORS(api, resources={r"/api/*": {"origins": "*"}})

@api.errorhandler
def default_error_handler(e):
    message = 'An unhandled exception occurred.'
    log.exception(message)
    return {'message': message}, 500
