from pytracking.evaluation import Tracker, OTBDataset, NFSDataset, UAVDataset, TPLDataset, VOTDataset, TrackingNetDataset, LaSOTDataset


def atom_nfs_uav():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = [Tracker('atom', 'default', i) for i in range(3)]

    dataset = NFSDataset() + UAVDataset()
    return trackers, dataset


def uav_test():
    # Run ATOM and ECO on the UAV dataset
    trackers = [Tracker('atom', 'default', i) for i in range(1)] + \
               [Tracker('eco', 'default', i) for i in range(1)]

    dataset = UAVDataset()
    return trackers, dataset
