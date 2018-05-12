import utils
import cv2
import numpy as np
from keras.models import load_model
import argparse
import os

def run_images():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', default="./test_images", type=str)
    parser.add_argument('--out_path', default='./out_images', type=str)
    parser.add_argument('--model_file', default="./model/yolov2-tiny-voc.h5", type=str)
    args = parser.parse_args()

    paths = utils.get_image_path(args.dir_path)
    images = []
    print('reading image from %s'%args.dir_path)
    for path in paths:
        image = cv2.imread(path)
        resized = cv2.resize(image, (416, 416))
        images.append(resized)

    image_processed = []
    for image in images:
        image_processed.append(utils.preprocess_image(image))

    print('loading model from %s'%args.model_file)
    model = load_model(args.model_file)
    predictions = model.predict(np.array(image_processed))

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    print('writing image to %s'% args.out_path)
    for i in range(predictions.shape[0]):
        boxes = utils.process_predictions(predictions[i],probs_threshold=0.3,iou_threshold=0.2)
        out_image = utils.draw_boxes(images[i],boxes)
        cv2.imwrite('%s/out%s.jpg'%(args.out_path,i), out_image)


run_images()
