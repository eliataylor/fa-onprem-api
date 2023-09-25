# Flask-Restplus settings
RESTPLUS_SWAGGER_UI_DOC_EXPANSION = 'list'
RESTPLUS_VALIDATE = True
RESTPLUS_MASK_SWAGGER = False
RESTPLUS_ERROR_404_HELP = False
global data
data = {}
global api
api =None
global jwt
jwt = None
global blacklist
blacklist = set()
global domain
domain = None
global camera_info_list
camera_info_list = None 
global cameras
cameras = []
global selected_camera 
selected_camera = -1
global acquirers
acquirers = {}
global pygigev
pygigev = None
