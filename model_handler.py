import os
from torchvision import transforms
import torch
import cv2
from PIL import Image
import numpy as np


def convert_mask_to_polygon(mask):
    mask = np.array(mask, dtype=np.uint8)
    cv2.normalize(mask, mask, 0, 255, cv2.NORM_MINMAX)
    contours = None
    if int(cv2.__version__.split('.')[0]) > 3:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[0]
    else:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[1]

    contours = max(contours, key=lambda arr: arr.size)
    if contours.shape.count(1):
        contours = np.squeeze(contours)
    if contours.size < 3 * 2:
        raise Exception('Less then three point have been detected. Can not build a polygon.')

    polygon = []
    for point in contours:
        polygon.append([int(point[0]), int(point[1])])

    return polygon


class ModelHandler:
    def __init__(self):
        base_dir = os.environ.get("MODEL_PATH", "/opt/nuclio/segformer")
        model_path = os.path.join(base_dir, "segformer_b0.pt")
        self.device = torch.device("cpu")
        print(torch.load(model_path))
        self.model = torch.load(model_path).to(self.device).eval()
        self.classes = ['background','void','dividing','guiding','stopping','chevron', 'parking','zebra','turn','reduction','attention','no parking' ]

    def handle(self, image, bbox = None, pos_points = None, neg_points = None, threshold = None):
        org_height = image.height
        org_width = image.width

        with torch.no_grad():
            # # extract a crop with padding from the image
            # width = image.width
            # height = image.height
            # crop_padding = 30
            # crop_bbox = [
            #     max(bbox[0][0] - crop_padding, 0),
            #     max(bbox[0][1] - crop_padding, 0),
            #     min(bbox[1][0] + crop_padding, width - 1),
            #     min(bbox[1][1] + crop_padding, height - 1)
            # ]
            # crop_shape = (
            #     int(crop_bbox[2] - crop_bbox[0] + 1),  # width
            #     int(crop_bbox[3] - crop_bbox[1] + 1),  # height
            # )
            # # try to use crop_from_bbox(img, bbox, zero_pad) here
            # input_crop = np.array(image.crop(crop_bbox)).astype(np.float32)
            #
            # # resize the crop
            # input_crop = cv2.resize(input_crop, (width, height), interpolation=cv2.INTER_NEAREST)
            # crop_scale = (width / crop_shape[0], height / crop_shape[1])
            #
            # def translate_points_to_crop(points):
            #     point = [((p[0] - crop_bbox[0]) * crop_shape[0], # x
            #               (p[1] - crop_bbox[1]) * crop_shape[1]) # y
            #              for p in points]
            #     return point
            #
            # pos_points = translate_points_to_crop(pos_points)
            # neg_points = translate_points_to_crop(neg_points)

            input_transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Resize(1024),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.22399999999999998, 0.225])
            ])

            input = input_transform(image).unsqueeze(0).to(self.device)
            output = self.model(input).squeeze(0)

            # make mask
            mask = output.argmax(axis = 0).numpy()

            return convert_mask_to_polygon(mask)


# img = Image.open("702.jpeg")
# hl = ModelHandler()
# out = hl.handle(img)
# print(out)


