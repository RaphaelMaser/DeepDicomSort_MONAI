import os
import warnings
import dicom2nifti
import subprocess
import argparse
import json
import monai
import tensorflow as tf
import yaml
from yaml import Loader
from tensorflow.keras.models import load_model
from monai.transforms import Compose, LoadImaged, Orientationd, AsChannelLastd, ScaleIntensityd, Resized, ScaleIntensity, ToNumpyd, SplitDimd
import numpy as np
from copy import deepcopy
import pickle

def convert_dicom_to_nifti(root_dir):
    new_root_dir = root_dir + '_nifti'
    if os.path.exists(new_root_dir):
        print("Data already converted.")
        return new_root_dir
    
    os.makedirs(new_root_dir, exist_ok=True)
    subprocess.run(f"dcm2niix -z n -f \"%p/%i\" -o {new_root_dir} {root_dir}", shell=True) 
    return new_root_dir

def discover_files(root_dir):
    discovered_files = []
    nifti, dicom = False, False
    
    for root, dirs, files in os.walk(root_dir):
        for i_file in files:
            if '.nii.gz' in i_file:
                nifti = True
            if '.dcm' in i_file:
                dicom = True
        if dicom & nifti:
            break
    
    if nifti & dicom:
        raise Exception("Both dicom and nifti files were found in the root directory. Please remove one of them.")
    
    if nifti:
        print("Found NIFTI images")
        new_root_dir = root_dir
        
    if dicom:
        print("Found DICOM images")
        new_root_dir = convert_dicom_to_nifti(root_dir)
    
    if dicom:
        mode = "dicom"
    else:
        mode = "bids"
    
    for root, dirs, files in os.walk(new_root_dir):
        for i_file in files:
            discovered_files.append(os.path.join(root, i_file))
            
    return discovered_files, mode

def get_dicom_label(file_path, label_indicator):
    
    root, ext = os.path.splitext(file_path)
    meta_data_file = root + ".json"
    with open(meta_data_file, "r") as f:
        meta_data = json.load(f)
        label = meta_data[label_indicator]   
         
        return label

# TODO fix: contrast is visible in sidecar file, but not in dicom header
def get_bids_label(file_path):
    root, ext = os.path.splitext(file_path)
    label = root.split("_")[-1]
    return label

def get_data(discovered_files, label_indicator, mode):
    data = []
    for file_path in discovered_files:
        if not file_path.endswith(".json"):  
            label = get_dicom_label(file_path, label_indicator) if mode == "dicom" else get_bids_label(file_path)
            data.append( {"image": file_path, "label": label})
    return data

def create_transforms():
    transforms = Compose([
        LoadImaged(keys=["image"], reader="NibabelReader", ensure_channel_first=True),
        Orientationd(keys=["image"], axcodes="RAS"),
        Resized(keys=["image"], spatial_size=[256, 256, 25], mode="trilinear"),
        SplitDimd(keys=["image"], dim=3, keepdim=False),
        ScaleIntensityd(keys=[f"image_{i}" for i in range(25)], minv=0, maxv=1, channel_wise=True),
        ToNumpyd(keys=[f"image_{i}" for i in range(25)])
    ])
    return transforms
    

def generator(monai_dataset):
    '''
    Dimensionality
    1. data : dict with 25 images of dimension [C, W, H]
    2. data_channel_last: dict with 25 images of dimension [W, H, C]
    3. slice: [W, H, C]
    '''
    for data in monai_dataset:
        #data_channel_first["image"] = data_channel_first["image"].view((image_shape[3], image_shape[1], image_shape[2], image_shape[0]))
        
        # for i in range(25):
        #     image_slice = deepcopy(data_channel_last) #TODO: more efficient way to do this?
        #     image_slice["image"] = image_slice["image"][i, :, :, :]
        #     image_slice["image"] = ScaleIntensityd(keys=["image"], minv=0, maxv=1, channel_wise=False)(image_slice)  
        #     slice_array.append(image_slice)
            
        for i in range(25):
            slice_number = f"image_{i}"
            image_slice = data[slice_number]
            
            # convert from (C, W, H) to (W, H, C)
            image_slice[slice_number] = image_slice[slice_number].moveaxis(0, 2)
            
            # add batch dimension: (W, H, C) -> (1, W, H, C)
            numpy_image = numpy_image[np.newaxis, :, :, :]
            tf_image = tf.convert_to_tensor(numpy_image)

            extra_data = np.zeros((1))
            extra_data = extra_data[np.newaxis, :]
            tf_extra_data = tf.convert_to_tensor(extra_data)
            
            label = slice["label"]
            tf_label = tf.convert_to_tensor(label)
            
            yield [tf_image, tf_extra_data], tf_label

def create_dataset(data, transforms):
    print("Create dataset")
    monai_dataset = monai.data.Dataset(data, transforms)
    
    # tf_dataset = tf.data.Dataset.from_generator(
    #     lambda: generator(monai_dataset),
    #     output_signature=(
    #         tf.TensorSpec(shape=(512, 25, 256, 256), dtype=tf.float32), 
    #         tf.TensorSpec(shape=(1, ), dtype=tf.float32)
    #     )
    # )
    tf_dataset = generator(monai_dataset)
    return monai_dataset, tf_dataset