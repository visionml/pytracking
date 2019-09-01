import torch.utils.data
from ltr.data.image_loader import default_image_loader


class BaseDataset(torch.utils.data.Dataset):
    """ Base class for datasets """

    def __init__(self, root, image_loader=default_image_loader):
        """
        args:
            root - The root path to the dataset
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
        """
        if root == '':
            raise Exception('The dataset path is not setup. Check your "ltr/admin/local.py".')
        self.root = root
        self.image_loader = image_loader

        self.sequence_list = []     # Contains the list of sequences.

    def __len__(self):
        """ Returns size of the dataset
        returns:
            int - number of samples in the dataset
        """
        return self.get_num_sequences()

    def __getitem__(self, index):
        """ Not to be used! Check get_frames() instead.
        """
        return None

    def is_video_sequence(self):
        """ Returns whether the dataset is a video dataset or an image dataset

        returns:
            bool - True if a video dataset
        """
        return True

    def get_name(self):
        """ Name of the dataset

        returns:
            string - Name of the dataset
        """
        raise NotImplementedError

    def get_num_sequences(self):
        """ Number of sequences in a dataset

        returns:
            int - number of sequences in the dataset."""
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        """ Returns information about a particular sequences,

        args:
            seq_id - index of the sequence

        returns:
            Dict containing following fields
                bbox -  Annotation for the sequence. A 2d tensor of shape (num_frames, 4).
                        Format [top_left_x, top_left_y, width, height]
                valid - A 1d tensor of shape (num_frames,) denoting whether the annotation is valid
                visible - A 1d tensor of shape (num_frames,) denoting whether the target is visible in the frame
            """
        raise NotImplementedError

    def get_frames(self, seq_id, frame_ids, anno=None):
        """ Get a set of frames from a particular sequence

        args:
            seq_id      - index of sequence
            frame_ids   - a list of frame numbers
            anno(None)  - The annotation for the sequence (see get_sequence_info). If None, they will be loaded.

        returns:
            list - List of frames corresponding to frame_ids
            dict - A dict containing annotations for the frames
            dict - A dict containing meta information about the sequence, e.g. class of the target object.

        """
        raise NotImplementedError

