import jpeg4py
import cv2 as cv


def default_image_loader(path):
    """The default image loader, reads the image from the given path. It first tries to use the jpeg4py_loader,
    but reverts to the opencv_loader if the former is not available."""
    if default_image_loader.use_jpeg4py is None:
        # Try using jpeg4py
        im = jpeg4py_loader(path)
        if im is None:
            default_image_loader.use_jpeg4py = False
            print('Using opencv_loader instead.')
        else:
            default_image_loader.use_jpeg4py = True
            return im
    if default_image_loader.use_jpeg4py:
        return jpeg4py_loader(path)
    return opencv_loader(path)

default_image_loader.use_jpeg4py = None


def jpeg4py_loader(path):
    """ Image reading using jpeg4py (https://github.com/ajkxyz/jpeg4py)"""
    try:
        return jpeg4py.JPEG(path).decode()
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None


def opencv_loader(path):
    """ Read image using opencv's imread function and returns it in rgb format"""
    try:
        im = cv.imread(path, cv.IMREAD_COLOR)
        # convert to rgb and return
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None
