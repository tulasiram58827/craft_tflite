"""
Authors
 * Tulasi Ram
"""

import onnx
import os
import cv2
import imgproc
import torch

import tensorflow as tf
from onnx_tf.backend import prepare
from torch.autograd import Variable


dataset_path = '/home/ram/Projects/OCR/test_files/'

# Export model to tensorflow
onnx_model = onnx.load('../models/craft.onnx')
tf_rep = prepare(onnx_model)
tf_rep.export_graph('../models/craft.pb')

print("Model converted to tensorflow graph succesfully.")

loaded = tf.saved_model.load('../models/craft.pb')

concrete_func = loaded.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

concrete_func.inputs[0].set_shape([None, 3, 1280, 800])

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Uncomment this line for float16 quantization.
#converter.target_spec.supported_types = [tf.float16]

# For Full Integer Quantization
def representative_data_gen():
    for file in os.listdir(dataset_path)[:10]:
        file_path = dataset_path+file
        image = imgproc.loadImage(file_path)
        image = cv2.resize(image, dsize=(800, 1280), interpolation=cv2.INTER_LINEAR)
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        x = x.cpu().detach().numpy()
        yield [x]

converter.representative_dataset = representative_data_gen


tf_lite_model = converter.convert()

open('craft_int.tflite', 'wb').write(tf_lite_model)

print("Converted to tensorflow lite succesfully.")