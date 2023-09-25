import base64
import datetime

import cv2
import numpy as np
from flask import jsonify
from flask import request
from flask_restx import Resource, Namespace, fields
from utils.helper import *
from main import test_ns

# from skimage.metrics import structural_similarity as ssim

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

# Define the expected request model (JSON body)
request_model = test_ns.model('MotionDetectionModel', {
    'camId': fields.String(description='Camera ID', required=True),
    'mask': fields.Integer(description='Mask Coordinate Array', required=True),
    'threshold': fields.Integer(description='Threshold value', required=True)
})

@ns.route('/detect_motion')
class MotionDetection(Resource):
    @ns.expect(request_model)  # Use the expect decorator with the request model
    def post(self):
        try:
            data = request.get_json()

            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            # tempImage

            camId = data.get('camId', None)
            mask = data.get('mask', [0,0,100,100])
            threshold = data.get('threshold', 0)

            if camId is None:
                return jsonify({"error": "Missing camera"}), 400

            last_seen = cv2.imread(f'/tmp/lastseen-{camId}.jpg')
            frame_height, frame_width, _ = last_seen.shape
            x1, y1, x2, y2 = int(mask[0] * frame_width), int(mask[1] * frame_height), int(
                mask[2] * frame_width), int(mask[3] * frame_height)

            visible = mock_frame_grab(camId)
            frame_bytes = base64.b64decode(visible)
            visible = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)

            total_diff = 0
            frame_count = 0

            # Crop frames using the mask coordinates
            prev_frame_masked = last_seen[y1:y2, x1:x2]
            frame_masked = visible[y1:y2, x1:x2]

            # Calculate the absolute difference between frames
            diff = cv2.absdiff(prev_frame_masked, frame_masked)

            # Sum the pixel-wise differences
            total_diff += diff.sum()
            frame_count += 1

            # Update the previous frame
            cv2.imwrite(f'/tmp/lastseen-{camId}.jpg', visible)

            # Calculate the average difference
            average_diff = total_diff / frame_count

            response_data = {
                "frame": "",
                "diff": average_diff
            }
            return jsonify(response_data), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500
