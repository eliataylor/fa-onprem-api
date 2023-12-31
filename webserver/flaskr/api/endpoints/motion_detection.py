from flask import jsonify
from flask import request
from flask_restx import Resource, fields, Namespace
from utils.helper import *
import requests
import cv2
import numpy as np
import time
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from io import BytesIO
import base64

ns = Namespace('test', description='Skeleton flask app')


# Define the expected request model (JSON body)
request_model = ns.model('MotionDetectionModel', {
    'ip': fields.String(description='FV Config IP Address', required=True),
    'camId': fields.String(description='Camera ID', required=True),
    'maskId': fields.String(description='ID of Mask to ignore areas of change', required=False),
    'threshold': fields.Float(description='Threshold value -1 (no similarity) to 1 (identical)', required=False),
    'debounce': fields.Integer(description='Milliseconds to wait for retry. Use a negative value to return anyway and allow front-end to poll this endpoint', required=False)
})

@ns.route('/detect_motion', methods=['POST'])
class MotionDetection(Resource):

    def mergeBase64(self, image1, mask, camId):
        # image1 = Image.open(BytesIO(base64.b64decode(bg)))
        image2 = Image.open(BytesIO(base64.b64decode(mask)))
        if image2.mode != 'RGBA':
            image2 = image2.convert('RGBA')
        image2 = image2.resize(image1.size)
        result_image = image1.copy()

        image1.save(f"/tmp/lastseen-{camId}-bg.png", format="JPEG")
        image2.save(f"/tmp/lastseen-{camId}-mask.png", format="PNG")

        result_image.paste(image2, (0, 0), image2)
        result_image = result_image.convert("RGB")
        return result_image

    def getMaskById(self, ip, mask):
        try:
            response = requests.get(f"http://{ip}:5000/api/capture/mask/get_masks")
            if response.status_code == 200:
                json_data = response.json()
                mask = next((person for person in json_data if person["maskId"] == mask), None)
                if mask is not None:
                    return mask['b64Mask'].split(",")[1]
                else:
                    return {"error": f"mask id not found"}, 404
            else:
                return {"error": f"mask request failed: {response.status_code}"}, 400
        except requests.exceptions.RequestException as e:
            return {"error": f"mask request exception thrown: {e}"}, 500


    def getCameraImage(self, ip, camId):
        try:
            response = requests.get(f"http://{ip}:5555/api/vision/vision/b64Frame/{camId}")
            if response.status_code == 200:
                json_data = response.json()
                if "b64" in json_data:
                    image1 = Image.open(BytesIO(base64.b64decode(json_data["b64"])))
                    return image1
                else:
                    return {"error": f"camera id not found"}, 404
            else:
                return {"error": f"image request failed: {response.status_code}"}, 400
        except requests.exceptions.RequestException as e:
            return {"error": f"image request exception thrown {e}"}, 500


    def compareImages(self, data) :
        camId = data.get('camId', None)
        mask = data.get('mask', None)
        ip = data.get('ip', None)
        threshold = data.get('threshold', 50)
        if threshold is None:
            threshold = 0
        else:
            threshold = float(data.get("threshold", 50))
        debounce = int(data.get('debounce', 250))

        tmpname = f'/tmp/lastseen-{camId}-flat.jpg'  # maybe prefix the name with a client id?

        frameImage = self.getCameraImage(ip, camId)
        if not isinstance(frameImage, Image.Image):
            return frameImage, 201 # is an error dictionary


        baseImage = frameImage
        if mask is not None and mask != '':
            mask_str = self.getMaskById(ip, mask)
            if not isinstance(mask_str, str):
                return mask_str, 201 # is an error dictionary
            baseImage = self.mergeBase64(baseImage, mask_str, camId)

        # reduce image to 640 wide / downsize only
        if baseImage.width > 640:
            new_height = int(baseImage.height * (640 / baseImage.width))
            baseImage = baseImage.resize((640, new_height), Image.ANTIALIAS)

        last_seen_np = cv2.imread(tmpname, cv2.IMREAD_UNCHANGED)
        baseimage_np = np.array(baseImage)

        if last_seen_np is None:
            response_data = {"score": 100}
        else:
            if last_seen_np.shape != baseimage_np.shape:
                similarity_score = 100 # just update the file and wait until next round
            else:
                baseimage_np = baseimage_np.astype(np.uint8)
                last_seen_np = last_seen_np.astype(np.uint8)

                win_size = min(min(baseimage_np.shape), min(last_seen_np.shape))
                if win_size % 2 == 0:  # win_size must be odd
                    win_size -= 1  #
                similarity_score = ssim(baseimage_np, last_seen_np, win_size=win_size)
                # similarity_score = 0 if similarity_score > 0 else (similarity_score + 1) * 50  # as percent > 0
                similarity_score = (similarity_score + 1) * 50  # as percentage of (-1 to 1)

            response_data = {"score": round(similarity_score)}

        baseImage.save(tmpname, format="JPEG")
        buffered = BytesIO()
        baseImage.save(buffered, format="JPEG")
        response_data["b64"] = base64.b64encode(buffered.getvalue()).decode()

        if threshold < response_data['score']:
            response_data["changed"] = True
            return response_data, 200
        else:
            if debounce > 0:
                time.sleep(debounce / 1000)
                return self.compareImages(data)
            else:
                return response_data, 200


    @ns.expect(request_model)
    def post(self):
        try:
            data = request.get_json()

            if not data:
                return {"error": "No JSON data provided"}, 400

            ip = data.get('ip', None)

            if ip is None:
                return {"error": "API IP Address is missing"}, 400

            camId = data.get('camId', None)
            if camId is None:
                return {"error": "Missing camera"}, 400

            result, status_code = self.compareImages(data)
            return jsonify(result)

        except Exception as e:
            return jsonify({"error": str(e)}), 500
