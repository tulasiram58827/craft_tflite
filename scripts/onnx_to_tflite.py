"""
Authors
 * Tulasi Ram
"""

import onnx


import tensorflow as tf
from onnx_tf.backend import prepare


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

tf_lite_model = converter.convert()

open('craft.tflite', 'wb').write(tf_lite_model)

print("Converted to tensorflow lite succesfully.")