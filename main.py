#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：CVAT_prepare 
@File    ：main.py
@IDE     ：PyCharm 
@Author  ：mabotan
@Date    ：2022/4/6 上午9:59 
@Copyright: Copyright: Copyright(C) 2017 MXNavi Co.,Ltd. ALL RIGHTS RESERVED
'''
import json
from model_handler import ModelHandler
import io
from PIL import Image
import numpy as np
import base64


def init_context(context):
    context.logger.info("INit context... 0%")
    model = ModelHandler()
    context.user_data.model = model
    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info()
    data = event.body
    pos_points = data["pos_points"]
    neg_points = data["neg_points"]
    obj_bbox = data.get("obj_bbox", None)
    threshold = data.get("threshold", 0.8)
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)

    polygon = context.user_data.model.handle(image)
    return context.Response(body=json.dumps(polygon),
                            headers={},
                            content_type='application/json',
                            status_code=200)
