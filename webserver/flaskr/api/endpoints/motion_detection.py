import base64

import cv2
import numpy as np
from flask import jsonify
from flask import request
from flask_restx import Resource, fields, Namespace
from utils.helper import *

ns = Namespace('test', description='Skeleton flask app')

# from skimage.metrics import structural_similarity as ssim

# Define the expected request model (JSON body)
request_model = ns.model('MotionDetectionModel', {
    'camId': fields.String(description='Camera ID', required=True),
    'mask': fields.List(fields.Float(min=0, max=100), min_items=4, max_items=4, required=True, description='Mask Coordinate Array', default=[0, 0, 100, 100]),
    'threshold': fields.Integer(description='Threshold value', required=True)
})

@ns.route('/detect_motion')
class MotionDetection(Resource):
    @ns.expect(request_model) # Use the expect decorator with the request model
    def post(self):
        try:
            data = request.get_json()

            if not data:
                return {"error": "No JSON data provided"}, 400

            camId = data.get('camId', None)
            mask = data.get('mask', [0,0,100,100])
            threshold = data.get('threshold', 0)

            if camId is None:
                return {"error": "Missing camera"}, 400

            if len(mask) < 4 or mask[0] >= mask[2] or mask[1] >= mask[3]:
                return {"error": "Invalid mask"}, 400

            # tempImage
            visible = mock_frame_grab(camId)
            frame_bytes = base64.b64decode(visible)
            visible = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR) # or IMREAD_COLOR?

            tmpname = f'/tmp/lastseen-{camId}.jpg' # maybe prefix the name with a client id?
            last_seen = cv2.imread(tmpname)
            if last_seen is None:
                response_data = {
                    "frame": visible,
                    "diff": 100
                }
            else:

                frame_height, frame_width, _ = last_seen.shape
                x1, y1, x2, y2 = int(mask[0] * frame_width), int(mask[1] * frame_height), int(
                    mask[2] * frame_width), int(mask[3] * frame_height)

                # Crop frames using the mask coordinates
                prev_frame_masked = last_seen[y1:y2, x1:x2]
                frame_masked = visible[y1:y2, x1:x2]

                # Calculate the absolute difference between frames
                diff = cv2.absdiff(prev_frame_masked, frame_masked)

                # pixel-wise diff
                total_diff = diff.sum()
                pixels = frame_width * frame_height
                average_diff = total_diff / pixels

                response_data = {
                    "frame": "",
                    "diff": average_diff
                }

                if average_diff > threshold: # the front-end can do this off diff too
                    response_data["changed"] = True

            # Update the previous frame
            update = cv2.imwrite(tmpname, visible)
            return response_data, 200

        except Exception as e:
            return {"error": str(e)}, 500
