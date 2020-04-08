import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

# sys.path.append('/home/goutam/python_ws/pytracking')
# sys.path.append('/home/goutam/random_repos/vot-toolkit/native/trax/support/python')

from pytracking.evaluation import Tracker


def run_vot(tracker_name, tracker_param, run_id=None):
    tracker = Tracker(tracker_name, tracker_param, run_id)
    tracker.run_vot()


def main():
    parser = argparse.ArgumentParser(description='Run VOT.')
    parser.add_argument('tracker_name', type=str)
    parser.add_argument('tracker_param', type=str)
    parser.add_argument('--run_id', type=int, default=None)

    args = parser.parse_args()

    run_vot(args.tracker_name, args.tracker_param, args.run_id)


if __name__ == '__main__':
    main()
