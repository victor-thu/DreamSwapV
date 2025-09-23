import logging
import os
import re
import pdb
import traceback
import pandas as pd
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO
try:
    import bytedtos
except Exception as e:
    os.system('pip install bytedtos --index-url=https://bytedpypi.byted.org/simple/')
    import bytedtos
try:
    import cv2
except Exception as e:
    os.system('pip install opencv-python-headless --index-url=https://bytedpypi.byted.org/simple/')
    import cv2
import time
import io, numpy as np
import json


# test image download path is https://cdn-tos-va.byteintl.net/obj/magellan-tiktok-trending-us/tmp/test_clothes.jpeg
# pip3 install opencv-python-headless

class TosClient(object):
    def __init__(self, bucket, access_key, timeout=10):
        self.bucket = bucket
        self.access_key = access_key
        self.tos_cli = bytedtos.Client(
            bucket=bucket,
            access_key=access_key,
            cluster="default",
            timeout=timeout)

    def get(self, key):
        try:
            return self.tos_cli.get_object(key).data
        except Exception as e:
            logging.exception('tos {}, get:{} failed, err:{}, trace:{}'.format(
                self.bucket, key, e, traceback.print_exc()))
            return None

    def put(self, key, data):
        """
        Args:
            key: string
            data: byte[]
        """
        try:
            self.tos_cli.put_object(key, data)
            return True
        except Exception as e:
            logging.exception('tos {} send:{} failed, err:{}, track:{}'.format(
                self.bucket, key, e, traceback.print_exc()))
            return False

    def exists(self, key):
        try:
            resp = self.tos_cli.head_object(key)
            return resp and resp.size > 0
        except Exception as e:
            logging.warn('tos {} can not find {}, err:{}, track:{}'.format(
                self.bucket, key, e, traceback.print_exc()))
            return False


bucket_name = 'gecom-scl-cv-nlp-public-us'
AK = 'BMI2QORLOVMPO39ZZFWX'
tos_cli = TosClient(bucket_name, AK)
Template_url = 'https://cdn-tos-va.byteintl.net/obj/{}/'.format(bucket_name) + '{}'





















def upload(local_file_path, save_path=None):
    if save_path is None:
        save_path =  'tmp/{}'.format(local_file_path.split('/')[-1])
    with open(str(local_file_path),'rb') as f:
        tos_cli.put(save_path, f.read())
    return Template_url.format(save_path)


def download(remote_path, save_path=None):
    if tos_cli.exists(remote_path):
        if save_path is None:
            save_path = 'tmp'
        os.makedirs(save_path, exist_ok=True)
        with open('{}/{}'.format(save_path, remote_path.split('/')[-1]),'wb') as f:
            f.write(tos_cli.get(remote_path))
        return True
    else:
        return False

def download_image(img_url, filename=None):
    # 发送 GET 请求
    response = requests.get(img_url)

    # 检查响应状态码
    if response.status_code == 200:
        # # 使用响应内容创建一个 BytesIO 对象
        if filename is None:
            img_data = BytesIO(response.content) 
            img = Image.open(img_data)
            # 保存图片到本地文件
            img.save(filename)
        return True
    else:
        # print(f"img_url: {img_url} Unable to download image. HTTP Response Code: {response.status_code}", )
        return False


def load_img(img_path):
    if os.path.isfile(img_path):
        img = cv2.imread(img_path)
    else:
        if tos_cli.exists(img_path):
            img_stream = io.BytesIO(tos_cli.get(img_path))
            img = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
    return img

def upload_img(img, img_path):
    is_success, im_buf_arr = cv2.imencode(".jpg", img)
    byte_im = io.BytesIO(im_buf_arr).getvalue()
    tos_cli.put(img_path, byte_im)
    return Template_url.format(img_path)


def load_json(json_path):
    data = None
    if os.path.isfile(json_path):
        img = json.load(open(json_path))
    else:
        if tos_cli.exists(json_path):
            data = json.loads(tos_cli.get(json_path))
        else:
            raise json_path + ' not exists'
    return data

def save_json(data, save_path=None):
    data = tos_cli.put(save_path,json.dumps(data))
    return Template_url.format(save_path)

def sort_key(s):
    try:
        return int(re.findall(r'\d+', s)[0])
    except:
        pdb.set_trace()
        print(9)

def save_csv(dict_data, save_path, columns=None):
    df = pd.DataFrame(dict_data)
    if columns is not None:
        df = df.reindex(columns=columns)
    df.to_csv(save_path, index=False)
    print(f"csv保存于: {save_path}")

def util_dict_list(json_path, mode='r', dict_list=None):
    '''[{},{}]'''
    if mode=='w':
        with open(json_path, 'w') as f:
            for dic in dict_list:
                json.dump(dic, f)
                f.write('\n')
        print(f"保存于: {json_path}")
    elif mode=='r':
        loaded_dict_list = []
        with open(json_path, 'r') as f:
            for line in f:
                loaded_dict_list.append(json.loads(line))
        return loaded_dict_list

if __name__=="__main__":
    # AB_0221
    folder_date = 'model_eval_0221'             # 子文件夹名，含日期
    folder_last = ('OutfitAnyone', 'Tryon')     # 最后一级的子文件夹名

    # TCS deploy
    folder_date = 'TCS'             # 子文件夹名，含日期
    folder_last = ('AB', 'score')     # 最后一级的子文件夹名
    folder_AB    = ['id', 'cloth_url', 'model_url',  'try_on_url_1', 'try_on_url_2']        # 与folder_last中一一对应, 一版不改,让folder_last与之对齐
    folder_score = ['id', 'cloth_url', 'model_url', 'try_on_url']

    # AB_0223
    folder_date = 'model_eval_0226'                                           # 子文件夹名，含日期
    another_project = 'ootd'                                                  # 与tryon对比的名称
    json_path = "/mnt/bn/gecom-scl-cvnlp-public/dingshunyi/prob/TryonOnline/data/de_data/test_sop_B/json/pairs.json"
    root_dir = os.path.dirname(json_path)
    pair_list = util_dict_list(json_path=json_path, mode='r')

    pair_score_list, pair_score_another_list, pair_AB_list = [], [], []
    for pair in tqdm(pair_list):
        cloth_path, model_path, try_on_path = pair['input_garment'], pair['input_person'], pair['out_path']
        name = cloth_path.split('/')[-1]
        if another_project=='ootd':
            try_on_path_2 = try_on_path.replace('out_img', 'out_img_ootd')

        cloth_url_online = os.path.join(f'dreamcloth_tryon/{folder_date}', 'cloth_url', name)
        model_url_online = os.path.join(f'dreamcloth_tryon/{folder_date}', 'model_url', name)
        try_on_url_1_online = os.path.join(f'dreamcloth_tryon/{folder_date}', 'try_on_url_1', name)
        try_on_url_2_online = os.path.join(f'dreamcloth_tryon/{folder_date}', 'try_on_url_2', name)

        if os.path.isfile(cloth_path) and os.path.isfile(model_path):
            cloth_url = upload(cloth_path, cloth_url_online)
            model_url = upload(model_path, model_url_online)

            if  os.path.isfile(try_on_path):
                try_on_url_1 = upload(try_on_path, try_on_url_1_online)
                pair_score_list.append({'cloth_url': cloth_url, 'model_url': model_url, 'tryon_url': try_on_url_1})
        
            if os.path.isfile(try_on_path_2):
                try_on_url_2 = upload(try_on_path_2, try_on_url_2_online)
                pair_score_another_list.append({'cloth_url': cloth_url, 'model_url': model_url, 'tryon_url': try_on_url_2})

            if os.path.isfile(try_on_path) and os.path.isfile(try_on_path_2):
                try_on_url_1 = upload(try_on_path, try_on_url_1_online)
                try_on_url_2 = upload(try_on_path_2, try_on_url_2_online)
                pair_AB_list.append({'cloth_url': cloth_url, 'model_url': model_url, 'try_on_url_1': try_on_url_1, 'try_on_url_2': try_on_url_2})
        

    print(f"打分样本数: {len(pair_score_list)} {len(pair_score_another_list)}\t AB样本数: {len(pair_AB_list)}")

    img_url_score_dict = {
        'cloth_url': [i['cloth_url'] for i in pair_score_list],
        'model_url': [i['model_url'] for i in pair_score_list],
        'tryon_url': [i['tryon_url'] for i in pair_score_list],
    }
    img_url_score_another_dict = {
        'cloth_url': [i['cloth_url'] for i in pair_score_another_list],
        'model_url': [i['model_url'] for i in pair_score_another_list],
        'tryon_url': [i['tryon_url'] for i in pair_score_another_list],
    }
    img_url_AB_dict = {
        'cloth_url': [i['cloth_url'] for i in pair_AB_list],
        'model_url': [i['model_url'] for i in pair_AB_list],
        'try_on_url_1': [i['try_on_url_1'] for i in pair_AB_list],
        'try_on_url_2': [i['try_on_url_2'] for i in pair_AB_list],
    }
    pdb.set_trace()

    save_dict = {'id':list(range(len(pair_AB_list)))}
    save_dict.update(img_url_AB_dict)
    save_csv(save_dict, save_path=f'{root_dir}/tryon_AB_{folder_date}.csv')

    save_dict = {'id':list(range(len(pair_score_list)))}
    save_dict.update(img_url_score_dict)
    save_csv(save_dict, save_path=f'{root_dir}/tryon_score_{folder_date}.csv')

    save_dict = {'id':list(range(len(pair_score_another_list)))}
    save_dict.update(img_url_score_another_dict)
    save_csv(save_dict, save_path=f'{root_dir}/tryon_score_{another_project}_{folder_date}.csv')