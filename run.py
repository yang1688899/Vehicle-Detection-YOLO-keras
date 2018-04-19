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

def run_video(src_path,out_path,batch_size=32):
    cap = cv2.VideoCapture(src_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    video_frames = []
    for i in range(num_frames-1):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (416, 416))
            frame = utils.preprocess_image(frame)
            video_frames.append(frame)
    cap.release()
    video_frames = np.array(video_frames)

    model = load_model("./model/yolov2-tiny-voc.h5")
    all_predictions = None
    print("predicting......")
    for offset in range(0,64,batch_size):
        end = offset + batch_size
        predictions = model.predict(video_frames[offset:end])

        if offset==0:
            all_predictions = predictions
        else:
            all_predictions = np.concatenate((all_predictions,predictions),axis=0)

    vedio_writer = cv2.VideoWriter(out_path,fourcc=fourcc,fps=fps,frameSize=(416,416))
    for i in range(len(all_predictions)):
        boxes = utils.process_predictions(all_predictions[0], probs_threshold=0.3, iou_threshold=0.1)
        out_frame = utils.draw_boxes(video_frames[i], boxes)
        vedio_writer.write(out_frame)








#run_images("./test_images")
run_video("./project_video.mp4","./out_video/out.mp4")