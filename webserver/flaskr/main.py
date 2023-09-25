import os
import numpy as np
import uuid
import json

from pathlib import Path
import datetime

import logging.config
import threading

from flask import Flask, Blueprint
from flask import render_template, send_from_directory
from flask import session
from flask import jsonify
from flask import make_response
from flask import g
from flask import request
from flask import Response
from flask_restx import Api
from flask_cors import CORS
import sys
import time

import settings

logging_conf_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../logging.config'))
print(logging_conf_path)
logging.config.fileConfig(logging_conf_path)
log = logging.getLogger(__name__)

app = Flask(__name__,instance_relative_config=True)
#CORS(app)

app.config['SWAGGER_UI_DOC_EXPANSION'] = settings.RESTPLUS_SWAGGER_UI_DOC_EXPANSION
app.config['RESTPLUS_VALIDATE'] = settings.RESTPLUS_VALIDATE
app.config['RESTPLUS_MASK_SWAGGER'] = settings.RESTPLUS_MASK_SWAGGER
app.config['ERROR_404_HELP'] = settings.RESTPLUS_ERROR_404_HELP
if 'JWT_SECRET_KEY' in os.environ:
    app.config['JWT_SECRET_KEY'] = os.environ['JWT_SECRET_KEY']
else:
    app.config['JWT_SECRET_KEY'] = 'imagerie'

app.config['PROPAGATE_EXCEPTIONS'] = True
app.config['JWT_TOKEN_LOCATION'] = ['headers','query_string']
app.config['JWT_BLACKLIST_ENABLED'] = True
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = False
app.config['JWT_BLACKLIST_TOKEN_CHECKS'] = ['access', 'refresh']
#settings.jwt = JWTManager(app)
app.secret_key = 'any random string'
blueprint = Blueprint('api', __name__, url_prefix='/api/dev')
authorizations = {
    'Bearer Auth': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'Authorization'
    },
}
settings.api = Api(blueprint,version='1.0', title='Flexible Vision Camera Server',
            description='A service to handle lite model predictions',security='Bearer Auth', authorizations=authorizations)
#settings.jwt._set_error_handler_callbacks(settings.api)


from api.endpoints.test import ns as test_ns
settings.api.add_namespace(test_ns)

from api.endpoints.motion_detection import ns as motion_ns
settings.api.add_namespace(motion_ns)


CORS(app, resources=r'/api/*', allow_headers='*')
app.register_blueprint(blueprint)

# @settings.jwt.token_in_blacklist_loader
# def check_if_token_in_blacklist(decrypted_token):
#     jti = decrypted_token['jti']
#     return jti in settings.blacklist

config_file = Path('config.json')
if config_file.is_file():
    with open('config.json') as f:
        settings.data = json.load(f)

    settings.data['image_type'] = '.jpg'

@app.route('/')
def root():
    return render_template('index.html')

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == '__main__':
    PORT = int(os.getenv('PORT')) if os.getenv('PORT') else 5123

    # This is used when running locally. Gunicorn is used to run the
    # application on Cloud Run. See entrypoint in Dockerfile.
    app.run(host='0.0.0.0', port=PORT, debug=True)