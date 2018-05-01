from YOLO_net import YOLO_net
import utils
import cv2
import numpy as np
from keras.models import load_model
import argparse
from moviepy.editor import VideoFileClip
from math import ceil

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

    print('writing image to %s'% args.out_path)
    for i in range(predictions.shape[0]):
        boxes = utils.process_predictions(predictions[i],probs_threshold=0.2,iou_threshold=0.5)
        out_image = utils.draw_boxes(images[i],boxes)
        cv2.imwrite('%s/out%s.jpg'%(args.out_path,i), out_image)


def run_video(src_path,out_path,batch_size=32):
    video_frames, num_frames, fps, fourcc = utils.preprocess_video(src_path)
    gen = utils.video_batch_gen(video_frames,batch_size=batch_size)

    model = load_model("./model/yolov2-tiny-voc.h5")

    print("predicting......")
    predictions = model.predict_generator(gen,steps=ceil(len(video_frames)/batch_size))

    # vedio_writer = cv2.VideoWriter(out_path,fourcc=fourcc,fps=fps,frameSize=(416,416))
    for i in range(len(predictions)):
        boxes = utils.process_predictions(predictions[i], probs_threshold=0.3, iou_threshold=0.1)
        out_frame = utils.draw_boxes(video_frames[i], boxes)
        cv2.imshow('frame', out_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # vedio_writer.write(out_frame)










# run_images()
run_video("./city_ride_cut.mp4","./out_video/city_ride_out.mp4")