import numpy as np
from skimage.transform import resize
from sklearn.ensemble import RandomForestRegressor as RFR
from skimage.color import rgb2lab, lab2rgb
from scipy.spatial.distance import cdist
from time import time
from itertools import product

def coordinate_features(coordinates, n=4):
    x, y = coordinates[:, 0], coordinates[:, 1]
    features = coordinates
    m = 1
    for _ in range(n):
        features = np.hstack((features, np.column_stack((m*x + y, m*x - y, x + m*y, x - m*y))))
        m *= 2
    return features

def gen_x(w, h, landmarks=None):
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    coordinates = np.column_stack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
    scaled_coordinates = coordinates / np.asarray([[w, h]], dtype=float)

    if landmarks is None:
        features = coordinate_features(scaled_coordinates)
    else:
        features = cdist(scaled_coordinates, landmarks)

    return np.hstack((coordinates, features))

def gen_xy(img, landmarks=None):
    w, h = img.shape[:2]
    X = gen_x(w, h, landmarks)
    x = X[:, 0].astype(int)
    y = X[:, 1].astype(int)
    Y = img[x, y]
    return X, Y

def pred_to_img(pred, X, w, h):
    pred_img = np.zeros((w, h, 3))
    x = X[:, 0].astype(int)
    y = X[:, 1].astype(int)
    pred_img[x, y] = pred
    return pred_img

def pixel_scale(img, npxs):
    # Ensure image is not empty and has a non-zero number of pixels
    if img.size == 0 or npxs <= 0:
        return img
        
    w, h = img.shape[:2]
    current_pixels = w * h
    if current_pixels == 0:
        return img

    scale_factor = np.sqrt(npxs / (current_pixels * 3.0))
    
    # Ensure target dimensions are at least 1x1
    new_w = max(1, int(round(w * scale_factor)))
    new_h = max(1, int(round(h * scale_factor)))

    return resize(img, (new_w, new_h), anti_aliasing=True)


def render(img,
           features='coordinates',
           ratio=0.00025,
           iterations=1,
           lab=True,
           depth=None,
           npxs=5e5,
           anti_aliasing=False,
           verbose=False):
    """
    Render an image using RandomForest regression based stylization.
    """
    t = time()

    # Ensure input image is float [0, 1]
    if img.dtype == np.uint8:
        img = img / 255.0

    w, h = img.shape[:2]
    wrender, hrender = (w*2, h*2) if anti_aliasing else (w, h)

    img_o = pixel_scale(img, npxs)
    wfit, hfit = img_o.shape[:2]

    img_lab = rgb2lab(img_o) if lab else img_o

    if features == 'landmarks':
        locations = list(np.linspace(0, 1, 7))
        landmarks = list(product(locations, locations))
    else:
        landmarks = None

    X, Y = gen_xy(img_lab, landmarks)
    xrender = gen_x(wrender, hrender, landmarks)

    min_samples = max(1, int(round(ratio * len(X))))
    model = RFR(
        n_estimators=iterations,
        n_jobs=-1,
        max_depth=depth,
        random_state=42,
        min_samples_leaf=min_samples
    )
    model.fit(X[:, 2:], Y)

    pred = model.predict(xrender[:, 2:])
    pred_img = pred_to_img(pred, xrender, wrender, hrender)

    if lab:
        # Clip LAB values to be within the valid range for conversion
        pred_img[:, :, 0] = np.clip(pred_img[:, :, 0], 0, 100)
        pred_img[:, :, 1] = np.clip(pred_img[:, :, 1], -128, 127)
        pred_img[:, :, 2] = np.clip(pred_img[:, :, 2], -128, 127)
        pred_img = lab2rgb(pred_img)

    # Calculate error on resized prediction to match training size
    pred_resized_for_error = resize(pred_img, (wfit, hfit), anti_aliasing=True)
    error = np.mean(np.square(pred_resized_for_error - img_o)) * 255.0

    if anti_aliasing:
        pred_img = resize(pred_img, (w, h), anti_aliasing=True)

    if verbose:
        print(f"{time() - t:08.3f} seconds to render")
        print(f"{error:08.3f} error (0-255 scaled)")
        print(f"{min_samples:d} min pixels considered") # Use %d for integer

    return np.clip(pred_img, 0, 1)