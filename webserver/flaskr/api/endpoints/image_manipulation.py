from flask import jsonify
from flask import request
from flask_restx import Resource, fields, Namespace
from utils.helper import *
import cv2
import numpy as np
import base64

ns = Namespace('test', description='Skeleton flask app')

# Define the expected request model (JSON body)
request_model = ns.model('ImageManipulationModel', {
    'base64': fields.String(description='base64 string', required=True),
    'filter': fields.String(description='Filter name', required=True),
    'value': fields.String(description='Filter value', required=True)
})

FILTERS = {
    "grayscale": cv2.COLOR_BGR2GRAY,
    "GaussianBlur": cv2.GaussianBlur,
    "medianBlur": cv2.medianBlur,
    "bilateralFilter": cv2.bilateralFilter,
    "Canny": cv2.Canny,
    "Sobel": cv2.Sobel,
    "Laplacian": cv2.Laplacian,
    "threshold": cv2.threshold,
    "adaptiveThreshold": cv2.adaptiveThreshold,
    "fastNlMeansDenoising": cv2.fastNlMeansDenoising,
    "erode": cv2.erode,
    "dilate": cv2.dilate,
    "morphologyEx": cv2.morphologyEx,
#    "color_space_conversions": {
        "BGR2HSV": cv2.COLOR_BGR2HSV,
        "BGR2YUV": cv2.COLOR_BGR2YUV,
        # Add more color space conversions as needed
#    },
    "equalizeHist": cv2.equalizeHist,
    "filter2D": cv2.filter2D,
    "resize": cv2.resize,
 #   "rotation_and_affine_transformations": {
        "getRotationMatrix2D": cv2.getRotationMatrix2D,
        "warpAffine": cv2.warpAffine,
#    },
    "warpPerspective": cv2.warpPerspective,
    "addWeighted": cv2.addWeighted
}


@ns.route('/image_manipulation', methods=['POST'])
class ImageManipulation(Resource):

    @ns.expect(request_model)  # Use the expect decorator with the request model
    def post(self):
        try:
            data = request.get_json()

            if not data:
                return {"error": "No JSON data provided"}, 400

            efx_name = data.get('efx_name', None)
            efx_value = data.get('efx_value', None)
            b64 = data.get('base64', None)

            if efx_value is None:
                return {"error": "filter value missing"}, 400

            if efx_name is None:
                return {"error": "filter missing"}, 400

            if b64 is None:
                return {"error": "Missing image"}, 400

            if efx_name not in FILTERS:
                return jsonify({"error": "Invalid filter name"}), 400

            # Decode the base64 string into a numpy array
            image_data = base64.b64decode(b64)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Apply the selected filter
            if efx_name == "grayscale":
                image = cv2.cvtColor(image, FILTERS[efx_name])
            elif efx_name == "GaussianBlur":
                efx_value = int(data.get('efx_value', 5))
                image = cv2.GaussianBlur(image, (efx_value, efx_value), 0)
            else:
                # THIS WON'T WORK FOR ALL FILTERS OBVIOUSLY!
                image = FILTERS[efx_name](image, efx_value)

            _, encoded_image = cv2.imencode(".jpg", image)
            base64_filtered = base64.b64encode(encoded_image).decode('utf-8')

            return jsonify({"b64": base64_filtered})

        except Exception as e:
            return jsonify({"error": str(e)}), 500


