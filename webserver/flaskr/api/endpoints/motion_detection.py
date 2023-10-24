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
    'mask': fields.List(fields.Float(min=0, max=100), min_items=4, max_items=4, required=True,
                        description='Mask Coordinate Array', default=[0, 0, 100, 100]),
    'threshold': fields.Integer(description='Threshold value', required=True)
})


@ns.route('/detect_motion')
class MotionDetection(Resource):

    def mergeBase64(self, bg, mask):
        image1 = Image.open(BytesIO(base64.b64decode(bg)))
        image2 = Image.open(BytesIO(base64.b64decode(mask)))
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

    def mergeMask(self, jpeg_base64, mask_base64):

        jpeg_bytes = base64.b64decode(jpeg_base64)
        mask_bytes = base64.b64decode(mask_base64)

        # image_np = np.frombuffer(jpeg_bytes, np.uint8)
        # mask_np = np.frombuffer(mask_bytes, np.uint8)

        # jpeg_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        # mask_image = cv2.imdecode(mask_np, cv2.IMREAD_COLOR)

        # if jpeg_image.shape[:2] != mask_image.shape[:2]:
        #    mask_image = cv2.resize(mask_image, (jpeg_image.shape[1], jpeg_image.shape[0]))

        # Decode the base64 data for both images
        background_image = Image.open(BytesIO(jpeg_bytes))
        background_image = background_image.convert("RGBA")
        mask_image = Image.open(BytesIO(mask_bytes))
        mask_image = mask_image.convert("RGBA")

        if background_image.size != mask_image.size:
            mask_image = mask_image.resize(background_image.size)

        # Convert images to NumPy arrays
        background_np = np.array(background_image)
        mask_np = np.array(mask_image)

        # Perform overlay with opacity
        alpha = 1  # Adjust opacity as needed (0 = fully transparent, 1 = fully opaque)
        result = cv2.addWeighted(background_np, 1 - alpha, mask_np, 1, 0)
        # Convert the result back to an RGBA image
        result_image = Image.fromarray(result, "RGBA")

        return result_image

    @ns.expect(request_model)  # Use the expect decorator with the request model
    def post(self):
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

        frameImage = None
        try:
            url = f"{camHost}/api/vision/vision/b64Frame/{camId}"
            # url = f"{camHost}/api/vision/vision/getJpg/{camId}"
            response = requests.get(url)
            if response.status_code == 200:
                json_data = response.json()
                frameImage = json_data["b64"]

                # with BytesIO(response.content) as image_data:
                #    frameImage = Image.open(image_data)

            else:
                return {"error": f"image request failed: {response.status_code}"}, 400
        except requests.exceptions.RequestException as e:
            return {"error": f"image request exception thrown {e}"}, 500

        if frameImage is None:
            return {"error": f"no image found for camera: {camId}"}, 400

        frame_bytes = None
        if mask is not None:
            try:
                response = requests.get(f"{detectHost}/api/capture/mask/get_masks")
                if response.status_code == 200:
                    json_data = response.json()
                    mask = next((person for person in json_data if person["maskId"] == mask), None)
                    if mask is not None:
                        # base64_data = re.search(r'base64,(.*)', mask['b64Mask']).group(1)
                        base64_data = mask['b64Mask'].split(",")[1]
                        # frame_bytes = self.mergeMask(frameImage, base64_data)
                        frame_bytes = self.mergeBase64(frameImage, base64_data)

                        # frame_bytes = self.mergeImages(frameImage, mask_image)
                        frame_bytes.save(tmpname, format="PNG")
                        # cv2.imwrite(tmpname, frame_bytes)
                else:
                    return {"error": f"mask request failed: {response.status_code}"}, 400
            except requests.exceptions.RequestException as e:
                return {"error": f"mask request exception thrown: {e}"}, 500

        buffered = BytesIO()
        frame_bytes.save(buffered, format="PNG")  # You can change the format to match your image type
        visible64 = base64.b64encode(buffered.getvalue()).decode()

        visible = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)  # or IMREAD_COLOR?

        last_seen = cv2.imread(tmpname)
        if last_seen is None:
            response_data = {
                "base64": visible64,
                "diff": 100
            }
        else:

            frame_height, frame_width, _ = last_seen.shape

            # Calculate the absolute difference between frames
            diff = cv2.absdiff(last_seen, visible)

            # pixel-wise diff
            total_diff = diff.sum()
            pixels = frame_width * frame_height
            average_diff = total_diff / pixels

            response_data = {
                "base64": visible64,
                "diff": round(average_diff)
            }

            if average_diff > threshold:  # the front-end can do this off diff too
                response_data["changed"] = True

        # Update the previous frame
        # update = cv2.imwrite(tmpname, visible)
        return response_data, 200
