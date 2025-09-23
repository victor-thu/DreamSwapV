import numpy as np
import thriftpy2
import cv2
import backoff
import euler
import os
current_directory = os.path.dirname(os.path.abspath(__file__))

from PIL import Image
from euler.base_compat_middleware import client_middleware
import time, random, thriftpy2, euler, hashlib, cv2, json, io
from euler import base_compat_middleware
# from laplace import Client

def gen_sign(nonce, app_secret, timestamp):
    keys = [str(nonce), str(app_secret), str(timestamp)]
    keys.sort()
    keystr = ''.join(keys)
    keystr = keystr.encode('utf-8')
    signature = hashlib.sha1(keystr).hexdigest()
    return signature.lower()
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return image_file.read()
def initiliaze_zc_model():
    service_thrift = thriftpy2.load(f"{current_directory}/idls/vproxy.thrift", module_name="service_thrift")
    base_thrift = thriftpy2.load(f"{current_directory}/idls/base.thrift", module_name="base_thrift")
    common_thrift = thriftpy2.load(f"{current_directory}/idls/common.thrift", module_name="common_thrift")
    consul = "sd://toutiao.labcv.algo_vproxy?cluster=default"
    client = euler.Client(service_thrift.VisionService, "{}".format(consul), timeout=10)
    client.use(base_compat_middleware.client_middleware)
    req = common_thrift.AlgoReq()
    req.Base = base_thrift.Base()
    req.auth_info = common_thrift.AuthInfo()
    req.auth_info.app_key = "6be35b36c56711eea01000163e10869c"
    req.auth_info.timestamp = str(int(time.time()))
    req.auth_info.nonce = str(random.randint(0, (1<<31)-1))
    req.auth_info.sign = gen_sign(req.auth_info.nonce, '6be35b51c56711eea01000163e10869c', req.auth_info.timestamp ) # 参考2.2.3 服务鉴权
    req.req_key = "saliency_seg"
    req_json = {
        "only_mask": 1,
        "rgb": [-1, -1, -1],
        "refine_mask": 2
    }
    req.req_json = json.dumps(req_json)
    return req, client


def segment(image, req, client):
    # 4. seg
    buffered = io.BytesIO()
    image.convert('RGB').save(buffered, format="JPEG")
    req.binary_data = [buffered.getvalue()]
    resp = client.Process(req)
    image_mask = cv2.imdecode(np.fromstring(resp.binary_data[0], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    image_mask = Image.fromarray(image_mask)
    return image_mask


def check_img(img):
    """图片转numpy格式"""
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    elif isinstance(img, np.ndarray):
        img_np = img
    else:
        raise NotImplementedError
    return img_np

def postprocess_img_mask(img, mask, return_type):
    img_np = check_img(img)
    mask = check_img(mask)
    
    assert img.shape[:2] == mask.shape[:2], f"{img.shape[:2]} !=  {mask.shape[:2]}"

    if return_type=='rgba':
        return Image.fromarray(np.concatenate([img_np, mask[..., np.newaxis]], axis=2))

    if return_type=='mask':
        return Image.fromarray(mask)

    if return_type=='white_bg':
        mask = mask[..., np.newaxis] / 255.
        img_np = img_np * mask + np.ones((img_np.shape)) * 255 * (1 - mask)
        return Image.fromarray(img_np.astype(np.uint8))
        
    if return_type=='white_bg&mask':
        mask = mask[..., np.newaxis] / 255.
        img_np = img_np * mask + np.ones((img_np.shape)) * 255 * (1 - mask)
        return Image.fromarray(img_np.astype(np.uint8)), Image.fromarray((mask[...,0] * 255).astype(np.uint8))
    
    raise NotImplementedError


class ICCV_Segmentor:
    def __init__(self):
        self.base_thrift = thriftpy2.load(f'{current_directory}/run_seg_inference_library/idls/base.thrift', module_name='base_thrift')
        self.segment_inference_thrift = thriftpy2.load(f'{current_directory}/run_seg_inference_library/idls/iccv/segment_inference.thrift', module_name='segment_inference_thrift')
        self.client = euler.Client(self.segment_inference_thrift.SegmentInferenceService, 'sd://ic.cv.segment_inference?idc=sg1&cluster=default', timeout=10)
        self.client.use(client_middleware)
    
    @backoff.on_predicate(backoff.expo,
                        predicate=lambda x: x is None,
                        jitter=backoff.full_jitter,
                        max_tries=3,
                        max_time=10)
    @backoff.on_exception(backoff.expo,
                        Exception,
                        jitter=backoff.full_jitter,
                        max_tries=3,
                        max_time=10)
    def __call__(self, img, req_type='clothes_seg', return_type='mask'):
        img_np = check_img(img)
        req = self.segment_inference_thrift.SegmentRequest()
        req.image = cv2.imencode('.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))[1].tobytes()
        req.req_type_list = [req_type]
        rsp = self.client.Process(req)
        mask = cv2.imdecode(np.fromstring(rsp.results[req_type], dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        return postprocess_img_mask(img_np, mask, return_type)

# class SaliencySegmentor:
#     def __init__(self):
#         self.pixel_means = (123.675, 116.28, 103.53)
#         self.pixel_stds = (58.395, 57.12, 57.375)
#         self.size = (1024, 1024)
#         self.input_dtypes = { 
#             "img": "float32"
#         }   

#         self.seg_model_name = 'sd_saliency_seg'
#         psm = 'sd://lyl.sd.seg?cluster=default'
#         self.seg_client = Client(psm, timeout=30)

#     def _pre_process(self, img):

#         if not isinstance(img, np.ndarray):
#             img = np.array(img)

#         img = cv2.resize(img, self.size, cv2.INTER_CUBIC)
#         img = np.asarray(img, dtype=np.float32)

#         if img.ndim < 3:
#             img = np.expand_dims(img, axis=-1)

#         img -= np.array(self.pixel_means)
#         img /= np.array(self.pixel_stds)
#         img = np.rollaxis(img, 2)
#         img = np.expand_dims(img, axis=0)
        
#         return img
    
#     @backoff.on_predicate(backoff.expo,
#                         predicate=lambda x: x is None,
#                         jitter=backoff.full_jitter,
#                         max_tries=3,
#                         max_time=10)
#     @backoff.on_exception(backoff.expo,
#                         Exception,
#                         jitter=backoff.full_jitter,
#                         max_tries=3,
#                         max_time=10)
#     def __call__(self, img, return_type='mask'):
#         img_np = check_img(img)
#         h, w = img_np.shape[:2]
#         img = self._pre_process(img_np)
#         inputs = {"img": img}
#         mask = self.seg_client.predict(self.seg_model_name, inputs, input_dtypes=self.input_dtypes)["pred"]
#         mask = (mask * 255).astype(np.uint8)
#         mask = Image.fromarray(mask.squeeze()).resize((w, h))
#         return postprocess_img_mask(img_np, mask, return_type)
#
# saliency_seg = SaliencySegmentor()

iccv_seg = ICCV_Segmentor()
req, zc_saliency_seg = initiliaze_zc_model()

def call_segmentation(img, seg_type='saliency', return_type='mask'):
    if seg_type=='saliency':
        # return saliency_seg(img, return_type=return_type)
        mask = segment(img, req, zc_saliency_seg)
        return postprocess_img_mask(np.array(img), np.array(mask), return_type)
    elif seg_type=='cloth':
        return iccv_seg(img, return_type=return_type)
    else:
        raise NotImplementedError('seg_type {} is not implemented, {} and {} are supported'.format(seg_type, 'saliency', 'cloth'))

if __name__=='__main__':
    call_segmentation(Image.open('/mlx/users/luyulei.233/playground/sd_video_pose/cardibs/image - 2024-04-16T113119.794.png'), ).save('test2.png')
