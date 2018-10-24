import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import glob
import os
import cv2
import numpy as np
def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()

def detect_img1(yolo,path):
    image = Image.open(path)
    r_image = yolo.detect_image(image)
    r_image.show()
    yolo.close_session()

def detect_img_batch(yolo,image_dir):
    path_list = glob.glob(os.path.join(image_dir,"*.jpg"))
    for path in path_list:
        image = cv2.imread(path)
        if isinstance(image,np.ndarray)==False:
            continue
        image = Image.open(path)
        r_image = yolo.detect_image(image)
        r_image.show()

    yolo.close_session()
FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()
    FLAGS.model_path = r"E:\project\yolo_V3\keras-yolo3\model_data\20181024\trained_weights_final_tiny.h5"
    FLAGS.anchors_path = r"E:\project\yolo_V3\keras-yolo3\model_data\tiny_yolo_anchors.txt"
    FLAGS.classes_path = r"E:\project\yolo_V3\keras-yolo3\model_data\voc_classes_person.txt"
    FLAGS.image = True
    FLAGS.input = r"E:\project\helmet\data\gao\yellow"
    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        #detect_img1(YOLO(**vars(FLAGS)),'E:\project\yolo_V3\keras-yolo3\JPEGImages\JPEGImages\PartB_00761.jpg')#返回属性和属性值的字典对象
        detect_img_batch(YOLO(**vars(FLAGS)),FLAGS.input)
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
