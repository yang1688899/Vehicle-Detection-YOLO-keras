from YOLO_net import YOLO_net
import utils
import cv2
import numpy as np
from keras.models import load_model
from moviepy.editor import VideoFileClip

def run_images(dir_path):
    paths = utils.get_image_path(dir_path)
    images = []
    for path in paths:
        image = cv2.imread(path)
        resized = cv2.resize(image, (416, 416))
        images.append(resized)

    image_processed = []
    for image in images:
        image_processed.append(utils.preprocess_image(image))

    model = load_model("./model/yolov2-tiny-voc.h5")
    predictions = model.predict(np.array(image_processed))

    for i in range(predictions.shape[0]):
        boxes = utils.process_predictions(predictions[i],probs_threshold=0.3,iou_threshold=0.1)
        out_image = utils.draw_boxes(images[i],boxes)
        cv2.imwrite('./out_images/out%s.jpg'%i, out_image)

def process_frame(frame):
    frame = cv2.resize(frame, (416, 416))
    frame = utils.preprocess_image(frame)
    frame = np.expand_dims(frame,axis=0)
    model = load_model("./model/yolov2-tiny-voc.h5")
    predictions = model.predict(np.array(frame))

    boxes = utils.process_predictions(predictions[0], probs_threshold=0.3, iou_threshold=0.1)
    out_frame = utils.draw_boxes(frame, boxes)
    return out_frame

def run_video(src_path,out_path):
    project_video_clip = VideoFileClip(src_path)
    project_video_out_clip = project_video_clip.fl_image(process_frame)
    project_video_out_clip.write_videofile(out_path, audio=False)

#run_images("./test_images")
run_video("./project_video.mp4","./out_video/out.mp4")