import pathlib
import os,sys
import tensorflow as tf
import numpy as np
import cv2
data_dir = f'/home/veritas/src/train/'
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*')))

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') ])


list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  return parts[-2] == CLASS_NAMES


def get_img(src,width, height):
  # 압축된 문자열을 3D uint8 텐서로 변환합니다
  src = tf.image.decode_jpeg(src, channels=3)
  # `convert_image_dtype`은0~1 사이의 float 값으로 변환해줍니다.
  src = tf.image.convert_image_dtype(src, tf.float32)
  # 이미지를 원하는 크기로 조정합니다.
  src = tf.image.resize(src, [width, height])
  return src


