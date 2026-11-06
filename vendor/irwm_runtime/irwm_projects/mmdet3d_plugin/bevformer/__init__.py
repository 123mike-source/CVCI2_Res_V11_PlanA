
from .dense_heads import *
from .detectors import *
from .modules import *

# CVCI3 uses the ResNet/FPN registered by MMDetection. Importing all optional
# upstream modules also imports InternImage and requires its DCNv3 extension,
# even though that backbone is absent from the IR-WM configuration.
