import inspect

from flask import jsonify
from flask import request
from flask_restx import Resource, fields, Namespace
from utils.helper import *
import cv2
import numpy as np
import base64
import json
from collections import namedtuple
import re

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

# Specify the path to your JSON file
with open("./utils/opencv-filters.json", "r") as json_file:
    FILTERPARAMS = json.load(json_file)

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

            if efx_name not in FILTERS or efx_name not in FILTERPARAMS:
                return jsonify({"error": "Invalid filter name"}), 400

            binary_data = base64.b64decode(b64)
            nparr = np.frombuffer(binary_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            method_function = FILTERS[efx_name]

            # inspected = inspect.getfullargspec(method_function).args
            inspected = FILTERPARAMS[efx_name]
            topass = []
            for param in inspected:

                type, index = inspected[param]["type"], inspected[param]["index"]

                if param == 'src':
                    topass.insert(index, nparr)
                elif param == 'dst':
                    height, width, channels = image.shape
                    dst = np.zeros((height, width, channels), dtype=np.uint8)
                    topass.insert(index, dst)
                else:
                    if param not in efx_value:
                        print('invalid param passed ' + param)
                    elif efx_value[param] is None or efx_value[param] == '':
                        continue
                    elif type == 'Point' or type == 'Point2f':
                        point = self.parse_point(efx_value[param])
                        topass.insert(index, point)
                    else:
                        if type == 'int':
                            cast = int(efx_value[param])
                        elif type == 'double':
                            cast = float(efx_value[param])
                        elif type == 'Scalar':
                            cast = getattr(cv2, efx_value[param])
                        elif type == 'bool':
                            cast = bool(efx_value[param])
                        elif type == 'Size':
                            cast = self.parse_size(efx_value[param])
                        elif type == 'Mat': # src2
                            cast = cv2.imdecode(np.frombuffer(base64.b64decode(efx_value[param]), np.uint8), cv2.IMREAD_COLOR)
                        else:
                            cast = efx_value[param]

                        topass.insert(index, cast)

            image = method_function(*topass)

            _, encoded_image = cv2.imencode(".jpg", image)
            base64_filtered = base64.b64encode(encoded_image).decode('utf-8')

            return jsonify({"b64": base64_filtered})

        except Exception as e:
            return jsonify({"error": str(e)}), 500


    def parse_point(self, point_str):
        # Define the Point namedtuple
        Point = namedtuple('Point', ['x', 'y'])

        # Use regular expression to extract coordinates
        match = re.match(r"\((\d+), (\d+)\)", point_str)

        if match:
            x = int(match.group(1))
            y = int(match.group(2))

            # Create a Point object
            point = Point(x, y)
            return point
        else:
            raise ValueError("Invalid point format")

    def parse_size(self, size_str):
        Size = namedtuple('Size', ['width', 'height'])

        # Use regular expression to extract width and height
        match = re.match(r"(\d+)\s*,?\s*(\d+)", size_str)

        if match:
            width = int(match.group(1))
            height = int(match.group(2))

            # Create a Size object
            size = Size(width, height)
            return size
        else:
            raise ValueError("Invalid size format")
