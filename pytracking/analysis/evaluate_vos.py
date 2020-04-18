import os
import numpy as np
import torch
import pandas as pd
from collections import OrderedDict
from ltr.data.image_loader import imread_indexed
from pytracking.evaluation import get_dataset
from pathlib import Path
from pytracking.analysis.plot_results import generate_formatted_report

import pytracking.analysis.vos_utils as utils

# Originally db_eval_sequence() in the davis challenge toolkit:
def evaluate_sequence(seq_name, segmentations, annotations, object_info, measure='J'):
    """
    Evaluate video sequence results.

      Arguments:
          segmentations (dict of ndarray): segmentation labels.
          annotations   (dict of ndarray): ground-truth labels.
          object_info   dict: {object_id: first_frame_index}

      measure       evaluation metric (J,F)
    """

    results = dict(raw=OrderedDict())

    _measures = {'J': utils.davis_jaccard_measure, 'F': utils.davis_f_measure}
    _statistics = {'decay': utils.decay, 'mean': utils.mean, 'recall': utils.recall, 'std': utils.std}

    for obj_id, first_frame in object_info.items():

        r = np.ones((len(annotations))) * np.nan

        for i, (an, sg) in enumerate(zip(annotations, segmentations)):
            if list(annotations.keys()).index(first_frame) < i < len(annotations) - 1:
                r[i] = _measures[measure](annotations[an] == obj_id, segmentations[sg] == obj_id)

        results['raw'][obj_id] = r

    for stat, stat_fn in _statistics.items():
        results[stat] = [float(stat_fn(r)) for r in results['raw'].values()]

    return results


def evaluate_dataset(results_path, dset_name, measure='J', to_file=True, scores=False, sequences=None, quiet=False):
    dset = get_dataset(dset_name)
    results = OrderedDict()
    dset_scores = []
    dset_decay = []
    dset_recall = []

    if to_file:
        f = open(results_path / ("evaluation-%s.txt" % measure), "w")

    def _print(msg):
        if not quiet:
            print(msg)
        if to_file:
            print(msg, file=f)

    if sequences is not None:
        sequences = [sequences] if not isinstance(sequences, (list, tuple)) else sequences

    target_names = []
    for j, sequence in enumerate(dset):
        if (sequences is not None) and (sequence.name not in sequences):
            continue

        # Load all frames
        frames = sequence.ground_truth_seg

        annotations = OrderedDict()
        segmentations = OrderedDict()

        for f in frames:
            if f is None:
                continue

            file = Path(f)
            annotations[file.name] = imread_indexed(file)
            if not scores:
                segmentations[file.name] = imread_indexed(os.path.join(results_path, sequence.name, file.name))
            else:
                raise NotImplementedError
        # Find object ids and starting frames

        object_info = dict()

        for f_id, d in sequence.init_data.items():
            for obj_id in d['object_ids']:
                object_info[int(obj_id)] = Path(d['mask']).name

        if 0 in object_info:  # Remove background
            object_info.pop(0)

        # Evaluate
        n_seqs = len(dset)
        n_objs = len(object_info)
        seq_name = sequence.name

        _print("%d/%d: %s: %d object%s" % (j + 1, n_seqs, seq_name, n_objs, "s" if n_objs > 1 else ""))
        r = evaluate_sequence(seq_name, segmentations, annotations, object_info, measure=measure)
        results[seq_name] = r

        # Print scores, per frame and object, ignoring NaNs

        per_obj_score = []  # Per-object accuracies, averaged over the sequence
        per_frame_score = []  # Per-frame accuracies, averaged over the objects

        for obj_id, score in r['raw'].items():
            target_names.append('{}_{}'.format(seq_name, obj_id))
            per_frame_score.append(score)
            s = utils.mean(score)  # Sequence average for one object
            per_obj_score.append(s)
            if n_objs > 1:
                _print("joint {obj}: acc {score:.3f} ┊{apf}┊".format(obj=obj_id, score=s, apf=utils.text_bargraph(score)))

        # Print mean object score per frame and final score
        dset_decay.extend(r['decay'])
        dset_recall.extend(r['recall'])
        dset_scores.extend(per_obj_score)

        seq_score = utils.mean(per_obj_score)  # Final score
        seq_mean_score = utils.nanmean(np.array(per_frame_score), axis=0)  # Mean object score per frame

        # Print sequence results
        _print("final  : acc {seq:.3f} ({dset:.3f}) ┊{apf}┊".format(
            seq=seq_score, dset=np.mean(dset_scores), apf=utils.text_bargraph(seq_mean_score)))

    _print("%s: %.3f, recall: %.3f, decay: %.3f" % (measure, utils.mean(dset_scores), utils.mean(dset_recall), utils.mean(dset_decay)))

    if to_file:
        f.close()

    return target_names, dset_scores, dset_recall, dset_decay


def evaluate_vos(trackers, dataset='yt2019_jjval', force=False):
    """ evaluate a list of trackers on a vos dataset.

    args:
        trackers - list of trackers to evaluate
        dataset - name of the dataset
        force - Force re-evaluation. If False, the pre-computed results are loaded if available
    """
    csv_name_global = f'{dataset}_global_results.csv'
    csv_name_per_sequence = f'{dataset}_per-sequence_results.csv'

    table_g_all = []
    table_seq_all = []
    scores = {'J-Mean': [], 'J-Recall': [], 'J-Decay': []}
    display_names = []
    for t in trackers:
        if t.display_name is not None:
            disp_name = t.display_name
        elif t.run_id is not None:
            disp_name = '{} {}_{:03d}'.format(t.name, t.parameter_name, t.run_id)
        else:
            disp_name = '{} {}'.format(t.name, t.parameter_name)

        display_names.append(disp_name)
        results_path = t.segmentation_dir

        csv_name_global_path = os.path.join(results_path, csv_name_global)
        csv_name_per_sequence_path = os.path.join(results_path, csv_name_per_sequence)
        if os.path.exists(csv_name_global_path) and os.path.exists(csv_name_per_sequence_path) and not force:
            table_g = pd.read_csv(csv_name_global_path)
            table_seq = pd.read_csv(csv_name_per_sequence_path)
        else:
            seq_names, dset_scores, dset_recall, dset_decay = evaluate_dataset(results_path, dataset, measure='J',
                                                                               to_file=False, scores=False,
                                                                               sequences=None)
            g_measures = ['J-Mean', 'J-Recall', 'J-Decay']
            g_res = np.array([utils.mean(dset_scores), utils.mean(dset_recall), utils.mean(dset_decay)])
            g_res = np.reshape(g_res, [1, len(g_res)])

            table_g = pd.DataFrame(data=g_res, columns=g_measures)
            with open(csv_name_global_path, 'w') as f:
                table_g.to_csv(f, index=False, float_format="%.3f")

            seq_measures = ['Sequence', 'J-Mean', 'J-Recall', 'J-Decay']

            table_seq = pd.DataFrame(data=list(zip(seq_names, dset_scores, dset_recall, dset_decay)), columns=seq_measures)
            with open(csv_name_per_sequence_path, 'w') as f:
                table_seq.to_csv(f, index=False, float_format="%.3f")

        scores['J-Mean'].append(table_g['J-Mean'].values[0]*100)
        scores['J-Recall'].append(table_g['J-Recall'].values[0]*100)
        scores['J-Decay'].append(table_g['J-Decay'].values[0]*100)

        table_g_all.append(table_g)
        table_seq_all.append(table_seq)

    report = generate_formatted_report(display_names, scores)
    print(report)

    return table_g_all, table_seq_all
