from pytracking.evaluation import Tracker, MobifaceDatasetAll, MobifaceDatasetTest


def all():
    # Run ATOM and ECO on the MobiFace dataset
    trackers = [Tracker('atom', 'default', i) for i in range(1)] + \
               [Tracker('eco', 'default', i) for i in range(1)]

    dataset = MobifaceDatasetAll()
    return trackers, dataset

def test():
    # Run ATOM and ECO on the MobiFace testset
    trackers = [Tracker('atom', 'default', i) for i in range(1)] + \
               [Tracker('eco', 'default', i) for i in range(1)]

    dataset = MobifaceDatasetTest()
    return trackers, dataset

