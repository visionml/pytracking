import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict

sys.path.append('../..')
from pytracking.evaluation import get_dataset, Tracker
import ltr.data.processing_utils as prutils
from pytracking import dcf


def load_dump_seq_data_from_disk(path):
    d = {}

    if os.path.exists(path):
        with open(path, 'r') as f:
            d = json.load(f)

    return d


def dump_seq_data_to_disk(save_path, seq_name, seq_data):
    d = load_dump_seq_data_from_disk(save_path)

    d[seq_name] = seq_data

    with open(save_path, 'w') as f:
        json.dump(d, f)


def update_seq_data(seq_candidate_data, frame_candidate_data, frame_state, subseq_state):
    if 'frame_states' not in seq_candidate_data and frame_state is not None:
        seq_candidate_data['frame_states'] = defaultdict(list)
    if 'subseq_states' not in seq_candidate_data and subseq_state is not None:
        seq_candidate_data['subseq_states'] = defaultdict(list)

    index = frame_candidate_data['index']

    for key, val in frame_candidate_data.items():
        val = val.float().tolist() if torch.is_tensor(val) else val
        seq_candidate_data[key].append(val)

    seq_candidate_data['frame_states'][frame_state].append(index)

    if subseq_state in ['HH', 'HK', 'HG']:
        seq_candidate_data['subseq_states'][subseq_state].append(index - 1)


def determine_frame_state(tracking_data, tracker, seq, th=0.25):
    visible = seq.target_visible[tracker.frame_num - 1]
    num_candidates = tracking_data['target_candidate_scores'].shape[0]

    state = None
    if num_candidates >= 2:
        max_candidate_score = tracking_data['target_candidate_scores'].max()

        anno_and_target_candidate_score_dists = torch.sqrt(
            torch.sum((tracking_data['target_anno_coord'] - tracking_data['target_candidate_coords']) ** 2, dim=1).float())

        ids = torch.argsort(anno_and_target_candidate_score_dists)

        score_dist_pred_anno = anno_and_target_candidate_score_dists[ids[0]]
        sortindex_correct_candidate = ids[0]
        score_dist_anno_2nd_highest_score_candidate = anno_and_target_candidate_score_dists[ids[1]] if num_candidates > 1 else 10000

        if (num_candidates > 1 and score_dist_pred_anno <= 2 and score_dist_anno_2nd_highest_score_candidate > 4 and
                sortindex_correct_candidate == 0 and max_candidate_score < th and visible != 0):
            state = 'G'
        elif (num_candidates > 1 and score_dist_pred_anno <= 2 and score_dist_anno_2nd_highest_score_candidate > 4 and
              sortindex_correct_candidate == 0 and max_candidate_score >= th and visible != 0):
            state = 'H'
        elif (num_candidates > 1 and score_dist_pred_anno > 4 and max_candidate_score >= th and visible != 0):
            state = 'J'
        elif (num_candidates > 1 and score_dist_pred_anno <= 2 and score_dist_anno_2nd_highest_score_candidate > 4 and
              sortindex_correct_candidate > 0 and max_candidate_score >= th and visible != 0):
            state = 'K'

    return state


def determine_subseq_state(frame_state, frame_state_previous):
    if frame_state is not None and frame_state_previous is not None:
        return '{}{}'.format(frame_state_previous, frame_state)
    else:
        return None


def extract_candidate_data(tracker, seq):
    tracker_data = tracker.distractor_dataset_data
    search_area_box = tracker_data['search_area_box']

    gth_box = seq.get_bbox(tracker.frame_num - 1, None)
    gth_center = torch.tensor(gth_box[:2] + (gth_box[2:] - 1) / 2)[[1, 0]]
    anno_label = tracker.get_label_function(gth_center, tracker_data['sample_pos'], tracker_data['sample_scale'])[0]
    _, target_anno_coord = dcf.max2d(anno_label.squeeze())

    score_map = tracker_data['score_map'].cpu()

    target_candidate_coords, target_candidate_scores = prutils.find_local_maxima(score_map.squeeze(), th=0.05, ks=5)

    return dict(search_area_box=search_area_box, index=tracker.frame_num - 1, target_anno_coord=target_anno_coord,
                target_candidate_scores=target_candidate_scores, target_candidate_coords=target_candidate_coords)


def run_sequence(seq, tracker, save_dir):
    params = tracker.get_parameters()

    # Get init information
    init_info = seq.init_info()

    tracker_inst = tracker.create_tracker(params)

    image = tracker._read_image(seq.frames[0])

    _ = tracker_inst.initialize(image, init_info)

    seq_candidate_data = defaultdict(list)
    frame_state_of_previous_frame = None

    for frame_num, frame_path in enumerate(tqdm(seq.frames[1:], leave=False), start=1):
        image = tracker._read_image(frame_path)
        info = seq.frame_info(frame_num)

        _ = tracker_inst.track(image, info)
        frame_candidate_data = extract_candidate_data(tracker_inst, seq)

        frame_state = determine_frame_state(frame_candidate_data, tracker_inst, seq)
        subseq_state = determine_subseq_state(frame_state, frame_state_of_previous_frame)

        if frame_state is not None:
            update_seq_data(seq_candidate_data, frame_candidate_data, frame_state, subseq_state)

        frame_state_of_previous_frame = frame_state

    dump_seq_data_to_disk(save_dir, seq.name, seq_candidate_data)


def run_tracker(tracker_name, parameter_file_name, dataset_name, save_dir):
    save_path = os.path.join(save_dir, 'target_candidates_dataset_{}_{}.json'.format(tracker_name, parameter_file_name))

    dumped_data = load_dump_seq_data_from_disk(save_path)

    tracker = Tracker(tracker_name, parameter_file_name)
    dataset = get_dataset(dataset_name)

    for seq in tqdm(dataset):
        if seq.name not in dumped_data:
            run_sequence(seq, tracker, save_path)


def main():
    parser = argparse.ArgumentParser(description='Run tracker and dump tracker states to form distractor dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracker.')
    parser.add_argument('parameter_file_name', type=str, help='Name of parameter file in the tracker folder.')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset.')
    parser.add_argument('save_dir', type=str, help='Path to storage folder')

    args = parser.parse_args()

    run_tracker(args.tracker_name, args.parameter_file_name, args.dataset_name, args.save_dir)


if __name__ == '__main__':
    main()