import settings
import os
import cv2
import numpy as np
import time
import datetime
import tarfile
import urllib.request
import settings
import base64
from pathlib import Path
import json
import threading
import requests
import gc
import ipaddress
from multiprocessing import Process

CAPDEV_HOST             = '172.17.0.1'
CAPDEV_PORT             = '5000'

if 'CAPDEV_HOST' in os.environ:
    CAPDEV_HOST = os.environ['CAPDEV_HOST']

if 'CAPDEV_PORT' in os.environ:
    CAPDEV_PORT = os.environ['CAPDEV_PORT']

def ms_timestamp():
    return int(datetime.datetime.now().timestamp()*1000)

class Test:
    def __init__(self, idx=None, info=None, info_ptr=None):
        self.idx = idx
        self.streaming = False
    
    def startWatcherThread(self):
        self.grabWatcher = threading.Thread(target=self.grab_watcher_loop, args=())
        self.grabWatcher.start()

    def grab_watcher_loop(self):
        while True:
            if self.lastGrab != None:
                cam_is_inactive = (self.lastGrab + CAM_INACTIVITY_INTERVAL) < ms_timestamp()
                if cam_is_inactive and self.thread != None:
                    print(self.info['serial_number'], ' is inactive... RELEASING')
                    self.release()
            time.sleep(5)
