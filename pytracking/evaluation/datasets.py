from collections import namedtuple
import importlib
from pytracking.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "pytracking.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter

dataset_dict = dict(
    otb=DatasetInfo(module=pt % "otb", class_name="OTBDataset", kwargs=dict()),
    nfs=DatasetInfo(module=pt % "nfs", class_name="NFSDataset", kwargs=dict()),
    uav=DatasetInfo(module=pt % "uav", class_name="UAVDataset", kwargs=dict()),
    tpl=DatasetInfo(module=pt % "tpl", class_name="TPLDataset", kwargs=dict()),
    tpl_nootb=DatasetInfo(module=pt % "tpl", class_name="TPLDataset", kwargs=dict(exclude_otb=True)),
    vot=DatasetInfo(module=pt % "vot", class_name="VOTDataset", kwargs=dict()),
    trackingnet=DatasetInfo(module=pt % "trackingnet", class_name="TrackingNetDataset", kwargs=dict()),
    got10k_test=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='test')),
    got10k_val=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='val')),
    got10k_ltrval=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='ltrval')),
    lasot=DatasetInfo(module=pt % "lasot", class_name="LaSOTDataset", kwargs=dict()),
    dv2017_val=DatasetInfo(module="ltr.dataset.davis", class_name="Davis", kwargs=dict(version='2017', split='val')),
    dv2016_val=DatasetInfo(module="ltr.dataset.davis", class_name="Davis", kwargs=dict(version='2016', split='val')),
    dv2017_test_dev=DatasetInfo(module="ltr.dataset.davis", class_name="Davis",
                                kwargs=dict(version='2017', split='test-dev')),
    dv2017_test_chal=DatasetInfo(module="ltr.dataset.davis", class_name="Davis",
                                 kwargs=dict(version='2017', split='test-challenge')),
    yt2019_test=DatasetInfo(module="ltr.dataset.youtubevos", class_name="YouTubeVOS",
                            kwargs=dict(version='2019', split='test')),
    yt2019_valid=DatasetInfo(module="ltr.dataset.youtubevos", class_name="YouTubeVOS",
                             kwargs=dict(version='2019', split='valid')),
    yt2019_valid_all=DatasetInfo(module="ltr.dataset.youtubevos", class_name="YouTubeVOS",
                                 kwargs=dict(version='2019', split='valid', all_frames=True)),
    yt2018_valid_all=DatasetInfo(module="ltr.dataset.youtubevos", class_name="YouTubeVOS",
                                 kwargs=dict(version='2018', split='valid', all_frames=True)),
    yt2018_jjval=DatasetInfo(module="ltr.dataset.youtubevos", class_name="YouTubeVOS",
                             kwargs=dict(version='2018', split='jjvalid')),
    yt2019_jjval=DatasetInfo(module="ltr.dataset.youtubevos", class_name="YouTubeVOS",
                             kwargs=dict(version='2019', split='jjvalid', cleanup=['starts'])),
    yt2019_jjval_all=DatasetInfo(module="ltr.dataset.youtubevos", class_name="YouTubeVOS",
                                 kwargs=dict(version='2019', split='jjvalid', all_frames=True, cleanup=['starts'])),
)


def load_dataset(name: str):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)  # Call the constructor
    return dataset.get_sequence_list()


def get_dataset(*args):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset