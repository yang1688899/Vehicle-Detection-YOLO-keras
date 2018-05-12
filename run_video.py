import utils
import cv2
from keras.models import load_model
from math import ceil
import os
import time

def run_video(src_path,out_path,batch_size=32):
    video_frames, num_frames, fps, fourcc = utils.preprocess_video(src_path)
    gen = utils.video_batch_gen(video_frames,batch_size=batch_size)

    model = load_model("./model/yolov2-tiny-voc.h5")

    print("predicting......")
    predictions = model.predict_generator(gen,steps=ceil(len(video_frames)/batch_size))

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    # vedio_writer = cv2.VideoWriter(out_path,fourcc=fourcc,fps=fps,frameSize=(416,416))
    for i in range(len(predictions)):
        boxes = utils.process_predictions(predictions[i], probs_threshold=0.3, iou_threshold=0.3)
        out_frame = utils.draw_boxes(video_frames[i], boxes)
        cv2.imshow('frame', out_frame)
        write_path = '%s/%05d.jpg'%(out_path,i)
        cv2.imwrite(write_path,out_frame*127)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # vedio_writer.write(out_frame)

run_video("./project_video.mp4","./out_video_img_1")