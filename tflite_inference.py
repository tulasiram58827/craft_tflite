import sys
import time
import cv2
import os
import imgproc
import torch
import craft_utils
import file_utils

import numpy as np

import tensorflow as tf
from torch.autograd import Variable

text_threshold = 0.7
link_threshold = 0.4
low_text = 0.4
canvas_size = 1280
mag_ratio = 1.5
poly=False
result_folder = './result/'


def run_tflite_model(input_data):
    print(input_data.shape)
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="craft.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
 
    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    y = interpreter.get_tensor(output_details[0]['index'])
    feature = interpreter.get_tensor(output_details[1]['index'])

    return y, feature

if __name__ == '__main__':
    image_path = sys.argv[1]

    start_time = time.time()
    image = imgproc.loadImage(image_path)
    image = cv2.resize(image, dsize=(800, 1280), interpolation=cv2.INTER_LINEAR)
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    # forward pass

    x = x.cpu().detach().numpy()
    y, feature = run_tflite_model(x)

    y = torch.from_numpy(y)
    feature = torch.from_numpy(feature)
    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()


    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    
    file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)
    filename, file_ext = os.path.splitext(os.path.basename(image_path))
    print("Total time taken to run CRAFT tflite model......", time.time()-start_time)