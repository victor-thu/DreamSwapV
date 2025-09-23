import euler
import thriftpy2
import json
from euler import base_compat_middleware
import hashlib
import time
import random
from io import BytesIO
from PIL import Image
import backoff

import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)

def gen_sign(nonce, app_secret, timestamp):
    keys = [str(nonce), str(app_secret), str(timestamp)]
    keys.sort()
    keystr = ''.join(keys)
    keystr = keystr.encode('utf-8')
    signature = hashlib.sha1(keystr).hexdigest()
    return signature.lower()

def pil_2_bin(pil_image):
    # 将图像文件转换为二进制数据
    with BytesIO() as buffer:
        pil_image.save(buffer, format='JPEG')
        binary_data = buffer.getvalue()
    return binary_data

def backoff_hdlr(details):
    print ("Backing off {wait:0.1f} seconds after {tries} tries "
           "calling function {target} with args {args} and kwargs "
           "{kwargs}".format(**details))

class NoFaceException(Exception):
    pass

class ReqOverlimitException(Exception):
    pass

class FaceFusion:
    def __init__(self):
        service_thrift = thriftpy2.load(f"{script_dir}/idls/vproxy.thrift",module_name="service_thrift")
        self.base_thrift = thriftpy2.load(f"{script_dir}/idls/base.thrift",module_name="base_thrift")
        self.common_thrift = thriftpy2.load(f"{script_dir}/idls/common.thrift",module_name="common_thrift")
        consul = "sd://toutiao.labcv.algo_vproxy?cluster=default"

        self.client = euler.Client(service_thrift.VisionService, "{}".format(consul), timeout=10)
        self.client.use(base_compat_middleware.client_middleware)

    @backoff.on_exception(backoff.expo, ReqOverlimitException, max_tries=3)
    def __call__(self, source_img, target_img):

        req = self.common_thrift.AlgoReq()
        req.Base = self.base_thrift.Base()

        req.auth_info = self.common_thrift.AuthInfo()
        req.auth_info.app_key = "911a91f3faf411ee8bcc00163e31b08c"
        req.auth_info.timestamp = str(int(time.time()))
        req.auth_info.nonce = str(random.randint(0, (1<<31)-1))
        req.auth_info.sign = gen_sign(req.auth_info.nonce, '911a920afaf411ee8bcc00163e31b08c', req.auth_info.timestamp ) # 参考2.2.3 服务鉴权

        req.req_key = "faceswap_ai_v3_fast"

        req_json = {
            "gpen": 0.5,
            "skin": 0.1,
            "skin_unifi":0.2
        }
        req.req_json = json.dumps(req_json)

        req.binary_data = [pil_2_bin(source_img), pil_2_bin(target_img)]
        resp = self.client.Process(req)
        if len(resp.binary_data) != 0:
            pil_image = Image.open(BytesIO(resp.binary_data[0]))
            return pil_image
        elif resp.BaseResp.StatusCode == 11000:
            raise ReqOverlimitException(resp.BaseResp.StatusMessage)
        elif resp.BaseResp.StatusCode == 22000:
            return target_img
        else:
            raise Exception(resp.BaseResp.StatusMessage)

if __name__ == '__main__':
    from glob import glob
    source_img = Image.open('/mlx/users/luyulei.233/playground/sd_video_pose/20240416-120047.jpeg')
    server = FaceFusion()
    for target_img_f in glob(f'/mlx/users/luyulei.233/playground/sd_video_pose/cardibs/*.png'):
        target_img = Image.open(target_img_f)
        res = server(source_img, target_img)
        print(res)
        res.save(target_img_f.replace('kemusan_lyl','kemusan_lyl_facefusion'))
