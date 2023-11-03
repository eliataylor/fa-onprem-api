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

            if efx_name is None:
                return {"error": "filter missing"}, 400

            if b64 is None:
                return {"error": "Missing base64 image"}, 400

            if efx_value is None or efx_name not in FILTERS or efx_name not in FILTERPARAMS:
                return jsonify({"error": "Invalid filter name"}), 400

            binary_data = base64.b64decode(b64)

            """
            with open("/tmp/b64-src.txt", "w") as file:
                file.write(b64)
            with open("/tmp/binary-b64decodes.png", "wb") as file:
                file.write(binary_data)
            binary_data = base64.decodebytes(b64.encode("utf-8"))
            with open("/tmp/binary-decodebytes.png", "wb") as file:
                file.write(binary_data)
            binary_data = base64.standard_b64decode(b64)
            with open("/tmp/binary-standard_b64decode.png", "wb") as file:
                file.write(binary_data)
            binary_data = codecs.decode(b64.encode("utf-8"), "base64")
            with open("/tmp/binary-codecs.png", "wb") as file:
                file.write(binary_data)
            """

            mime_type = imghdr.what(None, h=binary_data)
            src_np = np.frombuffer(binary_data, np.uint8)

            # image = Image.fromarray(src_np)

            image = Image.open(BytesIO(binary_data))

            parameters = FILTERPARAMS[efx_name]
            topass = []
            for param in parameters:

                type, index = parameters[param]["type"], parameters[param]["index"]

                if param == 'src':
                    topass.insert(index, src_np)
                elif param == 'dst':
                    height, width, channels = src_np.shape
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

            method_function = FILTERS[efx_name]
            resp_np = method_function(*topass)
            resp_image = Image.fromarray(resp_np)


            buffer = BytesIO()
            resp_image.save(buffer, format=mime_type)  # You can specify the desired image format (e.g., PNG)
            # image = Image.open(buffer)
            resp_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return jsonify({"b64": resp_b64})

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
