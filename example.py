import sys
import os
from matplotlib import pyplot as plt
import imageio
import numpy as np
from skimage.transform import resize

from stylize import render

def show_img(img, title):
    plt.clf()
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()

def save_img(path, img):
    """Convert float [0,1] image to uint8 and save."""
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    imageio.imwrite(path, img_uint8)

if __name__ == "__main__":
    try:
        path = sys.argv[1]
    except:
        path = 'resources/ekko.jpg'

    os.makedirs('example_images', exist_ok=True)

    print("Loading and resizing image...")
    img_orig = imageio.imread(path)

    max_height = 600
    if img_orig.shape[0] > max_height:
        scale_factor = max_height / img_orig.shape[0]
        new_width = int(img_orig.shape[1] * scale_factor)
        img = resize(img_orig, (max_height, new_width), anti_aliasing=True)
    else:
        img = img_orig / 255.0 if img_orig.dtype == np.uint8 else img_orig

    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.title("Original Resized Image")
    plt.show()

    fast_params = {
        "npxs": 8e4,   # Reduced from 1e5 for faster training
        "iterations": 1,
        "anti_aliasing": False,
        "verbose": True
    }

    # Render each style
    print("Rendering default stylization...")
    defaults = render(img, **fast_params)
    show_img(defaults, "Default stylization")
    save_img('example_images/defaults.png', defaults)

    print("Rendering landmark-based stylization...")
    landmarks = render(img, features='landmarks', **fast_params)
    show_img(landmarks, "Landmarks stylization")
    save_img('example_images/landmarks.png', landmarks)

    print("Rendering abstract (depth=4)...")
    abstract = render(img, depth=4, **fast_params)
    show_img(abstract, "Abstract stylization")
    save_img('example_images/abstract.png', abstract)

    print("Rendering more detailed (ratio=0.00005)...")
    more_detail = render(img, ratio=0.00005, **fast_params)
    show_img(more_detail, "More detail")
    save_img('example_images/more_detail.png', more_detail)

    print("Rendering less detailed (ratio=0.001)...")
    less_detail = render(img, ratio=0.001, **fast_params)
    show_img(less_detail, "Less detail")
    save_img('example_images/less_detail.png', less_detail)

    print("Rendering smoother (fast test)...")
    smoother = render(img, **fast_params)  
    show_img(smoother, "Smoother (fast test)")
    save_img('example_images/smoother.png', smoother)

    print("Rendering anti-aliased (fast test)...")
    aa = render(img, **fast_params)   
    show_img(aa, "Anti-aliasing (fast test)")
    save_img('example_images/aa.png', aa)

    print("All fast-test renders are saved in the 'example_images' folder!")
