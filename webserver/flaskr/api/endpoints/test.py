import datetime
import logging

from flask_restx import Resource, Namespace
from utils.helper import *

# from flask_jwt_extended import jwt_required, get_jwt_identity

log = logging.getLogger(__name__)

ns = Namespace('test', description='Skeleton flask app')

def ms_timestamp():
    return int(datetime.datetime.now().timestamp()*1000)


@ns.route('/server_status')
class CheckServer(Resource):
    def get(self):
        return 'Up and running', 200


@ns.route('/mock_frame_grab')
class MockFrameGrab(Resource):
    def get(self):
        frame = mock_frame_grab()
        return frame