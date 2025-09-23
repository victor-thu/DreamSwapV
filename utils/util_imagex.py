import os
import io
from PIL import Image
cur_dir = os.getcwd() # os.path.dirname(os.path.abspath(__file__))
from bytedance.image_service_sdk import image_service_sdk
from bytedance.image_service_sdk.image_service_sdk import ImageServiceSDK
os.chdir(cur_dir)
sdk = ImageServiceSDK(caller_name="hynis")
import backoff
import socket
import thriftpy2
from thriftpy2.transport import TTransportException

ImageService_thrift = image_service_sdk.ImageService_thrift
ImageService = image_service_sdk.ImageService

# hack_sdk
def upload_pic_by_bytes(self, image_byte_list):
    if not isinstance(image_byte_list, list):
        image_byte_list = [image_byte_list]
    req = ImageService_thrift.ImageServiceRequest()
    req.Caller = self.caller_name
    
    Image_data = []
    for image_byte in image_byte_list:
        try:
            Image_data.append(image_byte)
        except (IOError, SyntaxError):
            continue

    if not Image_data or len(Image_data) == 0:
        return "dir has no image"
    try:
        req.Image_data = Image_data
        resp = self.client.uploadImage(req)
        return resp
    except TTransportException as ex:
        return "Thrift transport exception" + str(ex)
sdk.upload_pic_by_bytes = upload_pic_by_bytes.__get__(sdk)

def _image_to_bytes(image):
    if isinstance(image, Image.Image):
        img = image
    else:
        img = Image.open(image)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    image_byte_stream = img_bytes.getvalue()
    img_bytes = bytes(image_byte_stream)
    return img_bytes

@backoff.on_exception(backoff.expo,
                    (socket.timeout, Exception),
                    jitter=backoff.full_jitter,
                    max_tries=3,
                    giveup=lambda x:'')
def upload_sdk(image):
    global sdk
    image_bytes = _image_to_bytes(image)
    resp = sdk.upload_pic_by_bytes(image_bytes)
    return resp.Image_info_list[0].External_uri