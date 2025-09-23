from .utils import *
from .transforms import *
try:
    from .face_fusion import FaceFusion
except Exception as e:
    print(e, 'FaceFusion 不可用')
    FaceFusion = None
    pass