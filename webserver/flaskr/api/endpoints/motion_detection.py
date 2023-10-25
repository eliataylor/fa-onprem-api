from flask import jsonify
from flask import request
from flask_restx import Resource, fields, Namespace
from utils.helper import *
import requests
import cv2
import numpy as np
import base64
import re
from PIL import Image
from io import BytesIO

ns = Namespace('test', description='Skeleton flask app')

# from skimage.metrics import structural_similarity as ssim

# Define the expected request model (JSON body)
request_model = ns.model('MotionDetectionModel', {
    'camId': fields.String(description='Camera ID', required=True),
    'camhost': fields.String(description='Camera Server Host', required=True),
    'fvhost': fields.String(description='Mask Server Host', required=True),
    'maskId': fields.String(description='ID of Mask to ignore areas of change'),
    'threshold': fields.Integer(description='Threshold value', required=True)
})

@ns.route('/detect_motion')
class MotionDetection(Resource):

    def mergeBase64(self, bg, mask):
        image1 = Image.open(BytesIO(base64.b64decode(bg)))
        image2 = Image.open(BytesIO(base64.b64decode(mask)))
        if image2.mode != 'RGBA':
            image2 = image2.convert('RGBA')
        image2 = image2.resize(image1.size)
        result_image = image1.copy()

        result_image.paste(image2, (0, 0), image2)
        return result_image

    def mergeImages(self, bg, mask):
        if bg.size != mask.size:
            mask_image = mask.resize(bg.size)

        # Convert images to NumPy arrays
        background_np = np.array(bg)
        mask_np = np.array(mask)

        alpha = 1  # Adjust opacity as needed (0 = fully transparent, 1 = fully opaque)
        result = cv2.addWeighted(background_np, 1 - alpha, mask_np, 1, 0)
        # Convert the result back to an RGBA image
        result_image = Image.fromarray(result, "RGBA")

        return result_image

    @ns.expect(request_model)  # Use the expect decorator with the request model
    def post(self):
        try:
            data = request.get_json()

            if not data:
                return {"error": "No JSON data provided"}, 400

            camId = data.get('camId', None)
            mask = data.get('mask', None)
            camHost = data.get('camhost', None)
            detectHost = data.get('fvhost', None)
            threshold = int(data.get('threshold', 0))

            if camHost is None:
                return {"error": "Camera Host missing"}, 400

            if detectHost is None:
                return {"error": "Detect Host missing"}, 400

            if camId is None:
                return {"error": "Missing camera"}, 400

            # tmpname = f'/tmp/lastseen-merged-{camId}.png'
            tmpname = f'/tmp/lastseen-{camId}.png'  # maybe prefix the name with a client id?

            try:
                url = f"{camHost}/api/vision/vision/b64Frame/{camId}"
                response = requests.get(url)
                if response.status_code == 200:
                    json_data = response.json()
                    frameImage = json_data["b64"]
                else:
                    return {"error": f"image request failed: {response.status_code}"}, 400
            except requests.exceptions.RequestException as e:
                return {"error": f"image request exception thrown {e}"}, 500

            if frameImage is None:
                return {"error": f"no image found for camera: {camId}"}, 400

            mergedImage = None
            if mask is not None:
                try:
                    response = requests.get(f"{detectHost}/api/capture/mask/get_masks")
                    if response.status_code == 200:
                        json_data = response.json()
                        mask = next((person for person in json_data if person["maskId"] == mask), None)
                        if mask is not None:
                            base64_data = mask['b64Mask'].split(",")[1]
                            mergedImage = self.mergeBase64(frameImage, base64_data)
                            mergedImage.save(tmpname, format="PNG")
                    else:
                        return {"error": f"mask request failed: {response.status_code}"}, 400
                except requests.exceptions.RequestException as e:
                    return {"error": f"mask request exception thrown: {e}"}, 500

            buffered = BytesIO()
            mergedImage.save(buffered)
            visible64 = base64.b64encode(buffered.getvalue()).decode()

            visible = cv2.imdecode(np.frombuffer(visible64, np.uint8), cv2.IMREAD_COLOR)  # or IMREAD_COLOR?

            last_seen = cv2.imread(tmpname)
            if last_seen is None:
                response_data = {"base64": visible64, "diff": 100}
            else:

                frame_height, frame_width, _ = last_seen.shape

                # Calculate the absolute difference between frames
                diff = cv2.absdiff(last_seen, visible)

                total_diff = diff.sum()
                pixels = frame_width * frame_height
                average_diff = total_diff / pixels

                response_data = { "base64": visible64, "diff": round(average_diff) }

                if average_diff > threshold:  # the front-end can do this off diff too
                    response_data["changed"] = True

            return response_data, 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500
