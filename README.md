# CRAFT: Character-Region Awareness For Text detection

This is the github repository for converting craft pretrained model to tflite version and to provide an inference code using the tflite model.

Please find the original paper [here](https://arxiv.org/abs/1904.01941)

# About CRAFT

PyTorch implementation for CRAFT text detector that effectively detect text area by exploring each character region and affinity between characters. The bounding box of texts are obtained by simply finding minimum bounding rectangles on binary map after thresholding character region and affinity scores.


### About the files

 - `pytorch_to_onnx.py` - Converts pretrained pytorch model to onnx.
 - `onnx_to_tflite.py` - Converts Onnx to TFLITE
 - `tflite_inference.py` - Inference with converted tflite model.
 - `craft_inference.py` - Inference with Pytorch Pretrained model.
 
 Pretrained model can be downloaded from [here](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ)
 
 Corresponding `ipynb` files are also provided.
 
 ### Results
 
### References

Some portions of the code are taken from [this repo](https://github.com/clovaai/CRAFT-pytorch)
 
 
