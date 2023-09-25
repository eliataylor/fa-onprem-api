#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np
import time
import datetime
import tarfile
import urllib.request
import settings
import base64
import threading
import re
import sys
import ctypes
import time
import json
import random


def ipAddr_from_string(s):
    "Convert dotted IPv4 address to integer."
    return reduce(lambda a,b: a<<8 | b, map(int, s.split(".")))

def ipAddr_to_string(ip):
    "Convert 32-bit integer to dotted IPv4 address."
    return ".".join(map(lambda n: str(ip>>n & 0xFF), [24,16,8,0]))

def macIdtoInteger(macId):
    res = re.match('^((?:(?:[0-9a-f]{2}):){5}[0-9a-f]{2})$', macId.lower())
    if res is None:
        print('INVALID MAC ID')
        return None    
    return int(res.group(0).replace(':', ''), 16)

def mock_frame_grab(camId=None):
    f = open (f'{os.path.dirname(__file__)}/mocked_b64_frames.json', "r")
    data = json.loads(f.read())

    idx   = random.randint(0, len(data['frames'])-1)
    frame = data['frames'][idx]

    return frame