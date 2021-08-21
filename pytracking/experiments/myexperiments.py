from pytracking.evaluation import Tracker, get_dataset, trackerlist


def atom_nfs_uav():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = trackerlist('atom', 'default', range(3))

    dataset = get_dataset('nfs', 'uav')
    return trackers, dataset


def uav_test():
    # Run DiMP18, ATOM and ECO on the UAV dataset
    trackers = trackerlist('dimp', 'dimp18', range(1)) + \
               trackerlist('atom', 'default', range(1)) + \
               trackerlist('eco', 'default', range(1))

    dataset = get_dataset('uav')
    return trackers, dataset

def sta_ytvos():
    trackers = []
    trackers.extend(trackerlist('sta', 'sta_ytvos', range(0, 1)))

    dataset = get_dataset('yt2019_jjval')

    return trackers, dataset

def sta_davis():
    trackers = []
    trackers.extend(trackerlist('sta', 'sta_davis', range(0, 1)))

    dataset = get_dataset('dv2017_val')

    return trackers, dataset
