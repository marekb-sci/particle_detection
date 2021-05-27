# -*- coding: utf-8 -*-

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from scipy import ndimage
from PIL import Image

class Config:
    main_input_dir = Path(r'')
    input_dir_alpha = main_input_dir / 'alpha_decay'
    input_dir_detector = main_input_dir / 'detector_effects'
    input_dir_bckg = main_input_dir / 'natural_bckg'

    output_dir = Path('../output')

    params = {
        'thr': 6*256,
        'size_thr': 5
        }

Config.output_dir.mkdir(exist_ok=True, parents=True)
files_alpha = list(Config.input_dir_alpha.iterdir())
files_detector = list(Config.input_dir_detector.iterdir())
files_bckg = list(Config.input_dir_bckg.iterdir())


#%%

def detect_events(img, thr=10, size_thr=0):
    s = ndimage.generate_binary_structure(2,2) #(2,1)
    x = img>= thr
    x = ndimage.binary_fill_holes(x)
    x = ndimage.binary_opening(x, structure=s)
    x = ndimage.binary_dilation(x, iterations=3)
    labels, nl = ndimage.label(x)
    objects_slices = ndimage.find_objects(labels)
    masks = [labels[obj_slice] == idx for idx, obj_slice in enumerate(objects_slices, start=1)]
    sizes = [mask.sum() for mask in masks]


    result = [{'idx': idx, 'slice': s, 'mask': m, 'size': size} for
               idx, (s, m, size) in enumerate(zip(objects_slices, masks, sizes), start=1) \
                   if size>size_thr
               ]

    return result

def get_sizes(labels):
    sizes = []
    for idx, obj_sclice in enumerate(ndimage.find_objects(labels), start=1):
        sizes.append((labels[obj_sclice] == idx).sum())
    return sizes


#%%

for file_alpha in files_alpha:
    img = np.array(Image.open(file_alpha))

    img_cropped = img[:,:470]
    fig, ax = plt.subplots()
    ax.imshow(img_cropped, vmin=0, vmax=50*256)
    # ax.colorbar()

    detections = detect_events(img_cropped, **Config.params)
    for detection in detections:
        sy, sx = detection['slice']
        ax.add_patch(Rectangle((sx.start,sy.start),sx.stop-sx.start,sy.stop-sy.start,linewidth=1,edgecolor='r',facecolor='none'))

    fig.savefig(Config.output_dir / f'detections_{file_alpha.name}.png', dpi=300)
    plt.close('all')