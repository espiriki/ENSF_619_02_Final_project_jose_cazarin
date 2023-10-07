#!/usr/bin/python
from PIL import Image
import os
import shutil
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from pathlib import Path
from cv2 import dnn_superres
import cv2
import numpy as np
import time
import re

def add_margin(height, width, pil_img, top, right, bottom, left, color):

    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


TARGET_WIDTH = 1024
TARGET_HEIGHT = 1024

PATH = "./all_together_dataset"

def remove_symbols(name):
    pattern_remove_symbols = r'[^a-zA-Z _-]+'
    text = re.sub(pattern_remove_symbols, '', name)
    return text

def keep_AR(img):

    target_aspect_ratio = TARGET_WIDTH/TARGET_HEIGHT
    original_width = img.width
    original_height = img.height
    current_aspect_ratio = original_width / original_height
    new_img = []

    if current_aspect_ratio == target_aspect_ratio:
        new_img = img
    if current_aspect_ratio < target_aspect_ratio:
        # need to increase width
        target_width = int(target_aspect_ratio * original_height)
        pad_amount_pixels = target_width - original_width
        new_img = add_margin(original_height, original_width, img, 0, int(pad_amount_pixels/2),
                             0, int(pad_amount_pixels/2), (0, 0, 0))

    if current_aspect_ratio > target_aspect_ratio:
        # need to increase height
        target_height = int(original_width/target_aspect_ratio)
        pad_amount_pixels = target_height - original_height
        new_img = add_margin(original_height, original_width, img, int(pad_amount_pixels/2),
                             0, int(pad_amount_pixels/2), 0, (0, 0, 0))

    return new_img


def resize(class_):
    
    # # Create an SR object
    # sr = dnn_superres.DnnSuperResImpl_create()
    # # Read the desired model
    # path = "./EDSR_x2.pb"
    # sr.readModel(path)
    # # Set the desired model and scale to get correct pre- and post-processing
    # sr.setModel("edsr", 2)
    # sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    # start = time.time()
    # result_with_upsample = sr.upsample(np.asarray(im))                    
    # end = time.time()
    # im = Image.fromarray(result_with_upsample)
    # print("Took {} seconds".format(end-start))
    # print("Size after upscaling: ", im.size)
                    

    
    path = os.path.join(PATH, class_)
    dirs = os.listdir(path)
    err_count = 0
    same_name_count = 0
    resized_path = os.path.join(PATH + "_resized", class_)
    if os.path.exists(resized_path):
        shutil.rmtree(resized_path)
    Path(resized_path).mkdir(parents=True, exist_ok=True)
    print("Num total files: ", len(dirs))
    for item in dirs:
        # print(path+"/"+item)
        if os.path.isfile(os.path.join(path, item)):
            try:
                im = Image.open(path+"/"+item)
                im = keep_AR(im)

                if True:
                    # print("resizing image {} with size: {}, {} ".format(
                    #     item, im.width, im.height))
                    imResize = im.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
                    item_no_extension = Path(item).stem
                    
                    print("1: ", item_no_extension)
                    
                    item_no_extension = remove_symbols(item_no_extension)
                    
                    print("2: ", item_no_extension)
                    
                    item_no_extension = item_no_extension.replace(" ", "_")
                    item_no_extension = item_no_extension.replace("-", "_")
                    
                    print("3: ", item_no_extension)
                    
                    item_no_extension = item_no_extension + "_" + str(same_name_count)
                    
                    print("4: ", item_no_extension)
                    
                    same_name_count = same_name_count + 1
                    
                    item_no_extension = item_no_extension.replace("__", "_")
                    
                    print("5: ", item_no_extension)
                    
                    path_to_save = f"./" + resized_path + "/" + item_no_extension + ".png"
                    # isExist = os.path.exists(path_to_save)
                    
                    # print(item_no_extension)
                    
                    # while isExist:
                    #     # print("file exists: ", path_to_save)
                    #     filename = Path(path_to_save).stem
                    #     new_filename = filename + "_" + str(same_name_count)
                        
                    #     new_filename = new_filename.replace("__", "_")
                        
                    #     same_name_count = same_name_count + 1
                    #     path_to_save = f"./" + resized_path + "/" + new_filename + ".png"
                    # isExist = os.path.exists(path_to_save)

                    imResize.convert("RGBA").save(path_to_save)
                else:
                    print("no need to resize, just converting image: ", item)
                    item_no_extension = Path(item).stem                    
                    item_no_extension = remove_symbols(item_no_extension)                    
                    item_no_extension = item_no_extension.replace(" ", "_")
                    path_to_save = f"./" + resized_path + "/" + item_no_extension + ".png"
                    isExist = os.path.exists(path_to_save)
                    
                    while isExist:
                        print("file exists: ", path_to_save)
                        filename = Path(path_to_save).stem
                        new_filename = filename + "_" + str(same_name_count)
                        
                        new_filename = new_filename.replace("__", "_")
                        
                        same_name_count = same_name_count + 1
                        path_to_save = f"./" + resized_path + "/" + new_filename + ".png"
                        isExist = os.path.exists(path_to_save)

                    im.convert("RGBA").save(path_to_save)
            except Exception as e:
                err_count = err_count + 1
                print("error on image {}: {}".format(item, str(e)))
        else:
            print("Item {} is not a file!".format(os.path.join(path, item)))

    print("Num of errors for class {}: {}".format(class_, err_count))


classes = ["Black", "Blue", "Green", "TTR"]
# classes = ["Black"]
# classes = ["Blue"]
# classes = ["TTR"]

pool = ThreadPool(processes=len(classes))
pool.map(resize, classes)
