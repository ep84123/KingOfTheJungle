import pathlib

import cv2
import numpy as np
import tensorflow as tf

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display


from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def object_image_detection(path: str):
    image = cv2.imread(path)
    image = cv2.resize(image, (640, 480))
    h = image.shape[0]
    w = image.shape[1]
    weights = "ssd_mobilenet/frozen_inference_graph.pb"
    model = "ssd_mobilenet/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    # load the MobileNet SSD model trained  on the COCO dataset
    net = cv2.dnn.readNetFromTensorflow(weights, model)
    class_names = []
    with open("ssd_mobilenet/coco_names.txt", "r") as f:
        class_names = f.read().strip().split("\n")
    # create a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0 / 127.5, (320, 320), [127.5, 127.5, 127.5])
    # pass the blog through our network and get the output predictions
    net.setInput(blob)
    output = net.forward()  # shape: (1, 1, 100, 7)
    # loop over the number of detected objects
    for detection in output[0, 0, :, :]:  # output[0, 0, :, :] has a shape of: (100, 7)
        # the confidence of the model regarding the detected object
        probability = detection[2]
        # if the confidence of the model is lower than 50%,
        # we do nothing (continue looping)
        if probability < 0.5:
            continue
        # perform element-wise multiplication to get
        # the (x, y) coordinates of the bounding box
        box = [int(a * b) for a, b in zip(detection[3:7], [w, h, w, h])]
        box = tuple(box)
        # draw the bounding box of the object
        cv2.rectangle(image, box[:2], box[2:], (0, 255, 0), thickness=2)

        # extract the ID of the detected object to get its name
        class_id = int(detection[1])
        # draw the name of the predicted object along with the probability
        label = f"{class_names[class_id - 1].upper()} {probability * 100:.2f}%"
        cv2.putText(image, label, (box[0], box[1] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Image', image)
    cv2.waitKey()


def image_proccess(path: str):
    image = cv2.imread(path)
    x1, y1, x2, y2 = 100, 100, 200, 200  # example bounding box coordinates
    roi = image[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    # Example thresholding approach:
    _, thresh = cv2.threshold(blur_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Example post-processing steps:
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Object Detection Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def new_image_detection(path: str):
#     wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
#     wget https://pjreddie.com/media/files/yolov3.weights
#     wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names



def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name,
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))
  model = model.signatures['serving_default']
  return model

# image_proccess("C:\\Users\\TLP-300\\Downloads\\photo_tree_1.jpg")
# object_image_detection("C:\\Users\\TLP-300\\Downloads\\photo_tree_1.jpg")
PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

PATH_TO_TEST_IMAGES_DIR = pathlib.Path('object_detection/test_images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
TEST_IMAGE_PATHS

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)

print(detection_model.inputs)
detection_model.output_dtypes


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    output_dict = model(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def show_inference(model, image_path, class_id):
    image_np = np.array(Image.open(image_path))
    output_dict = run_inference_for_single_image(model, image_np)
    boxes = []
    classes = []
    scores = []
    for i, x in enumerate(output_dict['detection_classes']):
        if x == class_id and output_dict['detection_scores'][i] > 0.5:
            classes.append(x)
            boxes.append(output_dict['detection_boxes'][i])
            scores.append(output_dict['detection_scores'][i])
    boxes = np.array(boxes)
    classes = np.array(classes)
    scores = np.array(scores)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        boxes,
        classes,
        scores,
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=2)
    display(Image.fromarray(image_np))


image_path = "C:\\Users\\TLP-300\\Downloads\\photo_tree_1.jpg"
#show_inference(detection_model, image_path, class_id)
show_inference(detection_model, image_path, 1)


