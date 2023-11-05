import inspect

from flask import jsonify
from flask import request
from flask_restx import Resource, fields, Namespace
from utils.helper import *
import cv2
import numpy as np
import imghdr
import base64
import json
from collections import namedtuple
import re
from PIL import Image
from io import BytesIO
import codecs

ns = Namespace('test', description='Skeleton flask app')

# Define the expected request model (JSON body)
request_model = ns.model('ImageManipulationModel', {
    'base64': fields.String(description='base64 string', required=True),
    'efx_name': fields.String(description='Filter name', required=True),
    'efx_value': fields.String(description='Filter value', required=True)
})

FILTERS = {
    "cvtColor": cv2.cvtColor,
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
    "equalizeHist": cv2.equalizeHist,
    "filter2D": cv2.filter2D,
    "resize": cv2.resize,
    "getRotationMatrix2D": cv2.getRotationMatrix2D,
    "warpAffine": cv2.warpAffine,
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
                return jsonify({"error": "No JSON data provided"})

            efx_name = data.get('efx_name', None)
            efx_value = data.get('efx_value', None)
            b64 = data.get('base64', None)

            if efx_name is None:
                return jsonify({"error": "filter missing"})

            if b64 is None:
                return jsonify({"error": "Missing base64 image"})

            if efx_value is None or efx_name not in FILTERS or efx_name not in FILTERPARAMS:
                return jsonify({"error": "Invalid filter name"})

            binary_data = base64.b64decode(b64)
            mime_type = imghdr.what(None, h=binary_data)
            src_np = np.array(Image.open(BytesIO(binary_data)))
            if efx_name == 'adaptiveThreshold':
                src_np = cv2.cvtColor(src_np, cv2.COLOR_BGR2GRAY)
            else:
                src_np = cv2.cvtColor(src_np, cv2.COLOR_BGR2RGB)

            parameters = FILTERPARAMS[efx_name]
            topass = []
            for param in parameters:

                type, index = parameters[param]["type"], parameters[param]["index"]

                if param == 'src':
                    topass.insert(index, src_np)
                elif param == 'dst':
                    dst = np.zeros_like(src_np)
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
                        elif type == 'Matrix':
                            cast = self.parse_matrix(efx_value[param])
                        elif type == 'base64':
                            cast = self.convert_to_img(efx_value[param])
                        elif type == 'anchor':
                            cast = self.parse_anchor(efx_value[param])
                        elif type == 'kernel':
                            cast = self.parse_kernel(efx_value[param])
                        else:
                            cast = efx_value[param]

                        topass.insert(index, cast)

            method_function = FILTERS[efx_name]

            if "src2" in parameters:
                src2 = topass[parameters["src2"]["index"]]
                topass[parameters["src2"]["index"]] = cv2.resize(src2, (topass[0].shape[1], topass[0].shape[0]))

            resp_np = method_function(*topass)

            if resp_np is not None and len(resp_np.shape) == 2:
                if resp_np.dtype != np.uint8:
                    resp_np = resp_np.astype(np.uint8)

            __, im_arr = cv2.imencode('.' + mime_type, resp_np)
            im_bytes = im_arr.tobytes()
            resp_b64 = base64.b64encode(im_bytes).decode()

            return jsonify({"b64": resp_b64})

        except Exception as e:
            return jsonify({"error": str(e)})

    def parse_matrix(self, matrix_str):
        try:
            # Load the JSON-formatted matrix string
            matrix_data = json.loads(matrix_str)

            # Convert the nested list to a NumPy array
            matrix = np.array(matrix_data)

            return matrix
        except (ValueError, json.JSONDecodeError):
            # Handle any potential parsing errors
            print("Error parsing the JSON-formatted matrix string.")
            return None

    def convert_to_img(self, b64_img):
        img = np.array(Image.open(
            BytesIO(base64.b64decode(b64_img))
        ))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def parse_anchor(self, point_str):
        values = point_str.strip("()").split(",")
        anchor = (int(values[0]), int(values[1]))
        return anchor

    def parse_point(self, point_str):
        # Define the Point namedtuple
        Point = namedtuple('Point', ['x', 'y'])

        # Use regular expression to extract coordinates
        match = re.match(r"\(([-\d]+), ([-\d]+)\)", point_str)

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

    def parse_kernel(self, size_str):
        values = size_str.split("x")
        size = (int(values[0]), int(values[1]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
        return kernel
