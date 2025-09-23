# coding=utf-8
# Author: hankun.01@bytedance.com

import euler
from euler.base_compat_middleware import client_middleware
import thriftpy2
import cv2
import numpy as np

base_thrift = thriftpy2.load('idls/base.thrift', module_name='base_thrift')
segment_inference_thrift = thriftpy2.load(
    'idls/iccv/segment_inference.thrift', module_name='segment_inference_thrift')


# client = euler.Client(service_cls=segment_inference_thrift.SegmentInferenceService, target='tcp://127.0.0.1:10011') # 'tcp://127.0.0.1:10011'
client = euler.Client(segment_inference_thrift.SegmentInferenceService, 'sd://ic.cv.segment_inference?idc=lf&cluster=default', timeout=10)

# ic.cv.segment_inference?idc=lf&cluster=default
# ic.cv.segment_inference.service.lf?cluster=default

client.use(client_middleware)

req = segment_inference_thrift.SegmentRequest()

image_path = '/mlx_devbox/users/luyulei.233/playground/U-2-Net/test_data/test_images/s_10.jpeg'
image_data = cv2.imread(image_path, cv2.IMREAD_COLOR)
with open(image_path, 'rb') as f:
    req.image = f.read()

# 支持的能力请查询 https://bytedance.feishu.cn/docs/doccnsBi5fhFiZL8GWHzR0JvUxh#FSPn53 中 req_type_list 部分
# req.req_type_list = ['human_seg', 'human_seg_with_object', 'hair_seg', 'head_seg', 'skin_seg', 'nose_seg', 'mask_seg', 'sky_seg', 'ground_seg', 'saliency_seg', 'human_parsing', 'clothes_seg', 'face_parsing', 'animated_face_parsing', 'human_instance_seg', 'accessory_seg']
req.req_type_list = ['saliency_seg']
rsp = client.Process(req)
print(rsp.BaseResp)

for algo_type in req.req_type_list:
    try:
        mask = cv2.imdecode(np.fromstring(rsp.results[algo_type], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except:
        import pdb
        pdb.set_trace()
    # print('algo_type:{}, max value:{}'.format(algo_type, mask.max()))
    if mask.max() != 0:
        mask = 255. / mask.max() * mask

    cv2.imwrite('{}.png'.format(algo_type), np.hstack([image_data, np.tile(mask[:,:,np.newaxis], (1,1,3))]))


