import random
import torch.utils.data
from pytracking import TensorDict


def no_processing(data):
    return data


class TrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. Each training sample is a
    tuple consisting of i) a set of train frames, used to learn the DiMP classification model and obtain the
    modulation vector for IoU-Net, and ii) a set of test frames on which target classification loss for the predicted
    DiMP model, and the IoU prediction loss for the IoU-Net is calculated.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames, num_train_frames=1, processing=no_processing, frame_sample_mode='causal'):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_test_frames - Number of test frames to sample.
            num_train_frames - Number of train frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_test_frames = num_test_frames
        self.num_train_frames = num_train_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        is_video_dataset = dataset.is_video_sequence()

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_test_frames + self.num_train_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset

        if is_video_dataset:
            train_frame_ids = None
            test_frame_ids = None
            gap_increase = 0

            if self.frame_sample_mode == 'interval':
                # Sample frame numbers within interval defined by the first frame
                while test_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1)
                    extra_train_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                                     min_id=base_frame_id[
                                                                                0] - self.max_gap - gap_increase,
                                                                     max_id=base_frame_id[
                                                                                0] + self.max_gap + gap_increase)
                    if extra_train_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + extra_train_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_test_frames,
                                                              min_id=train_frame_ids[0] - self.max_gap - gap_increase,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase)
                    gap_increase += 5  # Increase gap until a frame is found

            elif self.frame_sample_mode == 'causal':
                # Sample test and train frames in a causal manner, i.e. test_frame_ids > train_frame_ids
                while test_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_train_frames - 1,
                                                             max_id=len(visible) - self.num_test_frames)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + prev_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, min_id=train_frame_ids[0] + 1,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_test_frames)
                    # Increase gap until a frame is found
                    gap_increase += 5
        else:
            # In case of image dataset, just repeat the image to generate synthetic video
            train_frame_ids = [1] * self.num_train_frames
            test_frame_ids = [1] * self.num_test_frames

        train_frames, train_anno, meta_obj_train = dataset.get_frames(seq_id, train_frame_ids, seq_info_dict)
        test_frames, test_anno, meta_obj_test = dataset.get_frames(seq_id, test_frame_ids, seq_info_dict)

        data = TensorDict({'train_images': train_frames,
                           'train_anno': train_anno['bbox'],
                           'test_images': test_frames,
                           'test_anno': test_anno['bbox'],
                           'dataset': dataset.get_name(),
                           'test_class': meta_obj_test.get('object_class_name')})

        return self.processing(data)


class DiMPSampler(TrackingSampler):
    """ See TrackingSampler."""

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames, num_train_frames=1, processing=no_processing, frame_sample_mode='causal'):
        super().__init__(datasets=datasets, p_datasets=p_datasets, samples_per_epoch=samples_per_epoch, max_gap=max_gap,
                         num_test_frames=num_test_frames, num_train_frames=num_train_frames, processing=processing,
                         frame_sample_mode=frame_sample_mode)


class ATOMSampler(TrackingSampler):
    """ See TrackingSampler."""

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames=1, num_train_frames=1, processing=no_processing, frame_sample_mode='interval'):
        super().__init__(datasets=datasets, p_datasets=p_datasets, samples_per_epoch=samples_per_epoch, max_gap=max_gap,
                         num_test_frames=num_test_frames, num_train_frames=num_train_frames, processing=processing,
                         frame_sample_mode=frame_sample_mode)