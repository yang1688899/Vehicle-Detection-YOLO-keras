import numpy as np
import cv2
import os

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
colors = [(254.0, 254.0, 254), (239.88888888888889, 211.66666666666669, 127),
          (225.77777777777777, 169.33333333333334, 0), (211.66666666666669, 127.0, 254),
          (197.55555555555557, 84.66666666666667, 127), (183.44444444444443, 42.33333333333332, 0),
          (169.33333333333334, 0.0, 254), (155.22222222222223, -42.33333333333335, 127),
          (141.11111111111111, -84.66666666666664, 0), (127.0, 254.0, 254),
          (112.88888888888889, 211.66666666666669, 127), (98.77777777777777, 169.33333333333334, 0),
          (84.66666666666667, 127.0, 254), (70.55555555555556, 84.66666666666667, 127),
          (56.44444444444444, 42.33333333333332, 0), (42.33333333333332, 0.0, 254),
          (28.222222222222236, -42.33333333333335, 127), (14.111111111111118, -84.66666666666664, 0),
          (0.0, 254.0, 254), (-14.111111111111118, 211.66666666666669, 127)]

anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

class Box:
    def __init__(self):
        self.w = float()
        self.h = float()
        self.p_max = float()
        self.clas = int()
        self.x1 = int()
        self.y1 = int()
        self.x2 = int()
        self.y2 = int()

def sigmoid(x):
  return 1. / (1. + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def preprocess_image(resized):
    out_image = resized/127.
    return out_image

def load_weights(model, yolo_weight_file):
    data = np.fromfile(yolo_weight_file, np.float32)
    data = data[4:]
    index = 0
    for layer in model.layers:
        shape = [w.shape for w in layer.get_weights()]
        print(shape)
        if shape != []:
            kshape, bshape = shape
            bia = data[index:index + np.prod(bshape)].reshape(bshape)
            index += np.prod(bshape)
            ker = data[index:index + np.prod(kshape)].reshape(kshape)
            index += np.prod(kshape)
            layer.set_weights([ker, bia])

def iou(box1, box2):
    # Determine the coordinates of the intersection rectangle
    xA = max(box1.x1, box2.x1)
    yA = max(box1.y1, box2.y1)
    xB = min(box1.x2, box2.x2)
    yB = min(box1.y2, box2.y2)

    # Compute the area of intersection
    intersection_area = (xB - xA + 1) * (yB - yA + 1)

    # Compute the area of both rectangles
    box1_area = box1.w * box2.h
    box2_area = box2.w * box2.h

    # Compute the IOU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou


def non_maximal_suppression(thresholded_boxes, iou_threshold=0.3):
    nms_boxes = []
    if len(thresholded_boxes) > 0:
        # Add the best box because it will never be deleted
        nms_boxes.append(thresholded_boxes[0])

        # For each box (starting from the 2nd) check its iou with the higher score B-Boxes
        i = 1
        while i < len(thresholded_boxes):
            n_boxes_to_check = len(nms_boxes)
            # print('N boxes to check = {}'.format(n_boxes_to_check))
            to_delete = False

            j = 0
            while j < n_boxes_to_check:
                curr_iou = iou(thresholded_boxes[i], nms_boxes[j])
                if (curr_iou > iou_threshold):
                    to_delete = True
                # print('Checking box {} vs {}: IOU = {} , To delete = {}'.format(thresholded_predictions[i][0],nms_predictions[j][0],curr_iou,to_delete))
                j = j + 1

            if to_delete == False:
                nms_boxes.append(thresholded_boxes[i])
            i = i + 1

    return nms_boxes

#read anchors file
def get_anchors(filepath):
    file_object = open(filepath)
    try:
        contents = file_object.read()
    finally:
        file_object.close()

    anchors = [float(s) for s in contents.strip().replace(' ', '').split(',')]
    return anchors

def process_predictions(prediction, n_grid=13, n_class=20, n_box=5, probs_threshold=0.3, iou_threshold=0.3):
    prediction = np.reshape(prediction, (n_grid, n_grid, n_box, 5+n_class))
    boxes = []
    for row in range(n_grid):
        for col in range(n_grid):
            for b in range(n_box):
                tx, ty, tw, th, tc = prediction[row, col, b, :5]
                box = Box()

                box.w = np.exp(tw) * anchors[2 * b + 0] * 32.0
                box.h = np.exp(th) * anchors[2 * b + 1] * 32.0

                c_probs = softmax(prediction[row, col, b, 5:])
                box.clas = np.argmax(c_probs)
                box.p_max = np.max(c_probs) * sigmoid(tc)

                center_x = (float(col) + sigmoid(tx)) * 32.0
                center_y = (float(row) + sigmoid(ty)) * 32.0

                box.x1 = int(center_x - (box.w / 2.))
                box.x2 = int(center_x + (box.w / 2.))
                box.y1 = int(center_y - (box.h / 2.))
                box.y2 = int(center_y + (box.h / 2.))

                if box.p_max > probs_threshold:
                    boxes.append(box)

    boxes.sort(key=lambda b: b.p_max, reverse=True)

    filtered_boxes = non_maximal_suppression(boxes, iou_threshold)

    return filtered_boxes


def draw_boxes(image,boxes):
    for i in range(len(boxes)):
        color = colors[boxes[i].clas]
        best_class_name = classes[boxes[i].clas]

        # Put a class rectangle with box coordinates and a class label on the image
        image = cv2.rectangle(image, (boxes[i].x1, boxes[i].y1),
                                    (boxes[i].x2, boxes[i].y2),color)
        # cv2.putText(image, best_class_name, (int((boxes[i].x1 + boxes[i].x2) / 2),
        #                                            int((boxes[i].y1 + boxes[i].y2) / 2)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.putText(
            image, best_class_name + ' : %.2f' % boxes[i].p_max,
            (int(boxes[i].x1 + 5), int(boxes[i].y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            color, 1)

    return image


def get_image_path(path):
    paths = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        paths.append(file_path)
    return paths