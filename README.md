[//]: # (Image References)

[image1]: ./rm_img/Grid.png
[image2]: ./rm_img/yolo-output.png
[image3]: ./rm_img/Scores.png
[image4]: ./rm_img/Prediction.png
[image5]: ./rm_img/Prediction.png
[image6]: ./rm_img/out_nonms.jpg
[image7]: ./rm_img/out_filtered.jpg


[video1]: ./vedio_out/project_video_out.mp4 "Video"
## **车辆检测（YOLOV2)**

基于YOLOV2-tiny实现车辆检测

### YOLO简介：
YOLO意为 You Only Look Once，是一种基于深度学习的端对端（end to end）物体检测方法.与R-CNN,Fast-R-CNN,Faster-R-CNN等通过region proposal产生大量的可能包含待检测物体的 potential bounding box，再用分类器去判断每个 bounding box里是否包含有物体，以及物体所属类别的方法不同，YOLO将物体检测任务当做一个regression问题来处理.

### YOLO检测思路：
(这里以输入为416x416的YOLOV2为例)

首先将图片划分为13x13个栅格(grid cell):

![alt text][image1]

每个栅格负责预测5个bounding box(bounding box包括中心点坐标x,y及其宽w,高h,共4个值)。对于每个bounding box预测其是否包含物体的confidence score（1个值）,及其所包含物体class的possibility分布(由于有20个class，这里有20个值)。最终模型的的输出为13x13x125.这里13x13对应13x13个栅格(grid cell).每个bounding box 一共有4+1+20=25个值，每个栅格检测5个bounding box，则有每个栅格对应5x25=125个值，因此13x13x125.

以下为每个栅格的对应输出：

![alt text][image2]

模型最终检测到13x13x5=845个bounding box把所有的bounding box都画到原图上可能会是这样子的：

![alt text][image3]

大多数的bounding box 的confidence score都是非常低的(也就是没有检测到物体的)，只要少数的bounding box 是高confidence score的,检测到物体的。通过confidence score与最大的class的possibility相乘可以得到该bounding box 包含某物体的置信度，对这一置信度进行阈值过滤可以把大部分无意义的bounding box过滤掉。剩下的bounding box 可能存在的多重检测问题(即一个物体被多个bounding box检测)可以用IOU,heatmap等方法进行过滤整合，得到最终检测结果。

经过过滤处理的检测结果会是这样的：

![alt text][image4]

### 实现步骤

ps:本来是打算keras构建模型结构，然后加载weights文件训练后了参数实现的，但一直没有搞清楚weights文件的参数是怎么和模型各层对应上，最后找了[YAD2K](https://github.com/allanzelener/YAD2K),里面提供了把YOLOV2 weights文件直接转换成keras model文件的方法，就直接拿来用了。

* 使用[YAD2K](https://github.com/allanzelener/YAD2K)把weights文件转化为keras的h5文件

* 使用model预测bounding box

* 阈值筛选bounding box

#### 使用[YAD2K](https://github.com/allanzelener/YAD2K)把weights文件转化为keras的h5文件

下载相应的YOLO weights和cfg文件:
[weight文件下载](https://pjreddie.com/darknet/yolov2/);
获得[YAD2K](https://github.com/allanzelener/YAD2K);

运行yad2k.py文件，参数依次为：cfg文件路径，weights文件路径，model文件输出路径.
这里使用yolov2-tiny模型的voc版本，运行如下命令：
```
python ./yad2k.py ./yolov2-tiny-voc.cfg ./yolov2-tiny-voc.weights ./model/yolov2-tiny-voc.h5
```

#### 使用model预测bounding box
这里是一个使用keras的`predict_generator`对视频进行预测的示例(在内存显存足够的情况下可以直接用`predict`):

首先使用opencv读取视频，并进行resize，normalize等预处理:
```
#把给定视频转换为图片
def preprocess_video(src_path):
    cap = cv2.VideoCapture(src_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    video_frames = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (416, 416))
            frame = preprocess_image(frame)
            video_frames.append(frame)
    video_frames = np.array(video_frames)
    cap.release()
    return video_frames,num_frames,fps,fourcc
 ```
 
利用yield写一个generator function，用于在预测时生成指定batch_size大小的图片batch：
 ```
#prediction_generator
def video_batch_gen(video_frames,batch_size=32):
    for offset in range(0,len(video_frames),batch_size):
        yield video_frames[offset:offset+batch_size]
 ```
 
最后加载model,使用`predict_generator`进行预测:

```
video_frames, num_frames, fps, fourcc = utils.preprocess_video(src_path)
gen = utils.video_batch_gen(video_frames,batch_size=batch_size)

model = load_model("./model/yolov2-tiny-voc.h5")

print("predicting......")
predictions = model.predict_generator(gen)
```

#### 阈值筛选bounding box
得到predictions后，还要对predictions进行进一步处理,目的是过滤掉置信度低的及重叠的bounding box,使得图片的每一个物体对应一个置信度最高的bounding box。
为了方便对predictions进行处理，这里定义一个box类对bounding box的参数进行存储:
```
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
```

模型预测的是bounding box的中心点坐标x,y及其宽w,高h及是否包含物体的confidence score和其所包含物体class的possibility，其中中心点坐标x,y及其宽w,高h都是相对于每个栅格（grid cell）而言的，因此这些参数都需要进行转换。使用confidence score与最大的class的possibility相乘可以得到置信度，对这个置信度进行一些阈值过滤可以过滤掉置信度不高的bounding box:

```
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

```

经过初步处理后的bounding box仍会有大量重叠的情况:

![alt text][image6]

这里使用非极大值抑制（NMS）对bounding box进行过滤(ps:也可以使用[车辆识别（特征提取+svm分类器）](https://zhuanlan.zhihu.com/p/35607432)中介绍的heatmap进行过滤，只要达到使每个物体对应一个合适的bounding box的目的)：

```
#使用non_maxsuppression 筛选box
def non_maximal_suppression(thresholded_boxes, iou_threshold=0.3):
    nms_boxes = []
    if len(thresholded_boxes) > 0:
        # 添加置信度最高的box
        nms_boxes.append(thresholded_boxes[0])

        i = 1
        while i < len(thresholded_boxes):
            n_boxes_to_check = len(nms_boxes)
            to_delete = False

            j = 0
            while j < n_boxes_to_check:
                curr_iou = iou(thresholded_boxes[i], nms_boxes[j])
                if (curr_iou > iou_threshold):
                    to_delete = True
                j = j + 1

            if to_delete == False:
                nms_boxes.append(thresholded_boxes[i])
            i = i + 1

    return nms_boxes

```

最后把过滤后的bounding box展示在图片上,以下为示例代码:

```
#在图片上画出box
def draw_boxes(image,boxes):
    for i in range(len(boxes)):
        color = colors[boxes[i].clas]
        best_class_name = classes[boxes[i].clas]

        image = cv2.rectangle(image, (boxes[i].x1, boxes[i].y1),
                                    (boxes[i].x2, boxes[i].y2),color)

        cv2.putText(
            image, best_class_name + ' : %.2f' % boxes[i].p_max,
            (int(boxes[i].x1 + 5), int(boxes[i].y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            color, 1)

    return image
```
结果图片:

![alt text][image7]

