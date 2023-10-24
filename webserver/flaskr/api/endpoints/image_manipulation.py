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

# Define the expected request model (JSON body)
request_model = ns.model('ImageManipulationModel', {
    'base64': fields.String(description='base64 string', required=True),
    'filter': fields.String(description='Filter name', required=True),
    'value': fields.String(description='Filter value', required=True)
})

FILTERS = {
    "grayscale": cv2.COLOR_BGR2GRAY,
    "gaussian_blur": cv2.GaussianBlur,
    "median_blur": cv2.medianBlur,
    "bilateral_filter": cv2.bilateralFilter,
    "canny_edge_detection": cv2.Canny,
    "sobel_filter": cv2.Sobel,
    "laplacian_filter": cv2.Laplacian,
    "thresholding": cv2.threshold,
    "adaptive_thresholding": cv2.adaptiveThreshold,
    "denoising": cv2.fastNlMeansDenoising,
    "morphological_erosion": cv2.erode,
    "morphological_dilation": cv2.dilate,
    "morphological_operations": cv2.morphologyEx,
#    "color_space_conversions": {
        "BGR2HSV": cv2.COLOR_BGR2HSV,
        "BGR2YUV": cv2.COLOR_BGR2YUV,
        # Add more color space conversions as needed
#    },
    "histogram_equalization": cv2.equalizeHist,
    "custom_kernel_convolution": cv2.filter2D,
    "resize_and_scaling": cv2.resize,
 #   "rotation_and_affine_transformations": {
        "get_rotation_matrix": cv2.getRotationMatrix2D,
        "warp_affine": cv2.warpAffine,
#    },
    "image_warping": cv2.warpPerspective,
    "image_blending": cv2.addWeighted
}


@app.route('/image_manipulation', methods=['POST'])
class ImageManipulation(Resource):

    @ns.expect(request_model)  # Use the expect decorator with the request model
    def post(self):
        try:
            # Get filter name, filter value, and base64 string from the request
            filter_name = request.json.get('filter_name')
            filter_value = request.json.get('filter_value')
            base64_string = request.json.get('base64_string')

            if filter_name not in FILTERS:
                return jsonify({"error": "Invalid filter name"}), 400

            # Decode the base64 string into a numpy array
            image_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Apply the selected filter
            if filter_name == "grayscale":
                image = cv2.cvtColor(image, FILTERS[filter_name])
            else:
                image = FILTERS[filter_name](image, filter_value)

            # Encode the processed image as base64
            _, encoded_image = cv2.imencode(".jpg", image)
            base64_filtered = base64.b64encode(encoded_image).decode('utf-8')

            return jsonify({"filtered_base64": base64_filtered})

        except Exception as e:
            return jsonify({"error": str(e)}), 500


