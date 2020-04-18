"""
\file vot.py

@brief Python utility functions for VOT integration

@author Luka Cehovin, Alessio Dore

@date 2016

"""

import sys
import copy
import collections
import numpy as np

try:
    import trax
except ImportError:
    raise Exception('TraX support not found. Please add trax module to Python path.')


def make_full_size(x, output_sz):
    '''
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    '''
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)


Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])
Point = collections.namedtuple('Point', ['x', 'y'])
Polygon = collections.namedtuple('Polygon', ['points'])

class VOT(object):
    """ Base class for Python VOT integration """
    def __init__(self, region_format, channels=None):
        """ Constructor

        Args:
            region_format: Region format options
        """
        assert(region_format in [trax.Region.RECTANGLE, trax.Region.POLYGON, trax.Region.MASK])

        if channels is None:
            channels = ['color']
        elif channels == 'rgbd':
            channels = ['color', 'depth']
        elif channels == 'rgbt':
            channels = ['color', 'ir']
        elif channels == 'ir':
            channels = ['ir']
        else:
            raise Exception('Illegal configuration {}.'.format(channels))

        self._trax = trax.Server([region_format], [trax.Image.PATH], channels, customMetadata=dict(vot="python"))

        request = self._trax.wait()
        assert(request.type == 'initialize')
        if isinstance(request.region, trax.Polygon):
            self._region = Polygon([Point(x[0], x[1]) for x in request.region])
        if isinstance(request.region, trax.Mask):
            self._region = request.region.array(True)
        else:
            self._region = Rectangle(*request.region.bounds())
        self._image = [x.path() for k, x in request.image.items()]
        if len(self._image) == 1:
            self._image = self._image[0]

        self._trax.status(request.region)

    def region(self):
        """
        Send configuration message to the client and receive the initialization
        region and the path of the first image

        Returns:
            initialization region
        """

        return self._region

    def report(self, region, confidence = None):
        """
        Report the tracking results to the client

        Arguments:
            region: region for the frame
        """
        assert(isinstance(region, (Rectangle, Polygon, np.ndarray)))
        if isinstance(region, Polygon):
            tregion = trax.Polygon.create([(x.x, x.y) for x in region.points])
        if isinstance(region, np.ndarray):
            tregion = trax.Mask.create(region)
        else:
            tregion = trax.Rectangle.create(region.x, region.y, region.width, region.height)
        properties = {}
        if not confidence is None:
            properties['confidence'] = confidence
        self._trax.status(tregion, properties)

    def frame(self):
        """
        Get a frame (image path) from client

        Returns:
            absolute path of the image
        """
        if hasattr(self, "_image"):
            image = self._image
            del self._image
            return image

        request = self._trax.wait()

        if request.type == 'frame':
            image = [x.path() for k, x in request.image.items()]
            if len(image) == 1:
                return image[0]
            return image
        else:
            return None


    def quit(self):
        if hasattr(self, '_trax'):
            self._trax.quit()

    def __del__(self):
        self.quit()
