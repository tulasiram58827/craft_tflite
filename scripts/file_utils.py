# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import easyocr
import imgproc

# reader = easyocr.Reader(['en'], detector=False, recognizer=True)

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    print(img_dir)
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def save_polygon(img, pts, count):
    points = np.array(pts, dtype=np.int32)
    height = img.shape[0]
    width = img.shape[1]

    mask = np.zeros((height, width), dtype=np.uint8)
    print(points)
    cv2.fillPoly(mask, [points], (255))

    res = cv2.bitwise_and(img,img,mask = mask)

    rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

    cv2.imwrite("output"+"/"+str(count)+".jpg" , cropped )
    results = reader.recognize(cropped, batch_size=128)
    text = results[0][-2]
    return text

    # cv2.imshow("same size" , res)
    # cv2.waitKey(0)

def saveResult(img_file, img, boxes, dirname='./result/', verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        res_file = dirname + "res_" + filename + '.txt'
        res_img_file = dirname + "res_" + filename + '.jpg'

        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        #data = open('task3.txt', 'w')
        count = 0
        with open(res_file, 'w') as f:
            for i, box in enumerate(boxes):
                #text = save_polygon(img, box, count)
                #box_data = ""
                #for co_ord in box:
                #    box_data+=f"{co_ord[0]}, {co_ord[1]}"
                #print(box_data, text)
                #data.write(box_data+","+text+"\n")
                #count+=1
                poly = np.array(box).astype(np.int32).reshape((-1))
                #strResult = ','.join([str(p) for p in poly]) + '\r\n'
                #f.write(strResult)
                poly = poly.reshape(-1, 2)
                min_co = tuple(np.min(poly, axis=0))
                max_co = tuple(np.max(poly, axis=0))
                #x_1, x_2, y_1, y_2 = poly[0][0], poly[1][0], poly[1][1], poly[2][1]
                cv2.rectangle(img, min_co, max_co, (0, 0, 255), 2)
                #cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                ptColor = (0, 255, 255)
                if verticals is not None:
                    if verticals[i]:
                        ptColor = (255, 0, 0)

                if texts is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                    cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

        # Save result image
        cv2.imwrite(res_img_file, img)

