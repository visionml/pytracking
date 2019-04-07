import jpeg4py
import cv2 as cv


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