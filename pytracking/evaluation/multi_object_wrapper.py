import numpy as np
from collections import OrderedDict
import time
import copy


class MultiObjectWrapper:
    def __init__(self, base_tracker_class, params, visdom=None, fast_load=False):
        self.base_tracker_class = base_tracker_class
        self.params = params
        self.visdom = visdom

        self.initialized_ids = []
        self.trackers = OrderedDict()

        self.fast_load = fast_load
        if self.fast_load:
            self.tracker_copy = self.base_tracker_class(self.params)
            if hasattr(self.tracker_copy, 'initialize_features'):
                self.tracker_copy.initialize_features()

    def create_tracker(self):
        tracker = None
        if self.fast_load:
            try:
                tracker = copy.deepcopy(self.tracker_copy)
            except:
                pass
        if tracker is None:
            tracker = self.base_tracker_class(self.params)
        tracker.visdom = self.visdom
        return tracker

    def _split_info(self, info):
        info_split = OrderedDict()
        init_other = OrderedDict()              # Init other contains init info for all other objects
        for obj_id in info['init_object_ids']:
            info_split[obj_id] = dict()
            init_other[obj_id] = dict()
            info_split[obj_id]['object_ids'] = [obj_id]
            info_split[obj_id]['sequence_object_ids'] = info['sequence_object_ids']
            if 'init_bbox' in info:
                info_split[obj_id]['init_bbox'] = info['init_bbox'][obj_id]
                init_other[obj_id]['init_bbox'] = info['init_bbox'][obj_id]
            if 'init_mask' in info:
                info_split[obj_id]['init_mask'] = (info['init_mask'] == int(obj_id)).astype(np.uint8)
                init_other[obj_id]['init_mask'] = info_split[obj_id]['init_mask']
        for obj_info in info_split.values():
            obj_info['init_other'] = init_other
        return info_split

    def _set_defaults(self, tracker_out: dict, defaults=None):
        defaults = {} if defaults is None else defaults

        for key, val in defaults.items():
            if tracker_out.get(key) is None:
                tracker_out[key] = val

        return tracker_out

    def default_merge(self, out_all):
        out_merged = OrderedDict()

        out_first = list(out_all.values())[0]
        out_types = out_first.keys()

        # Merge segmentation mask
        if 'segmentation' in out_types and out_first['segmentation'] is not None:
            # Stack all masks
            # If a tracker outputs soft segmentation mask, use that. Else use the binary segmentation
            segmentation_maps = [out.get('segmentation_soft', out['segmentation']) for out in out_all.values()]
            segmentation_maps = np.stack(segmentation_maps)

            obj_ids = np.array([0, *map(int, out_all.keys())], dtype=np.uint8)
            segm_threshold = getattr(self.params, 'segmentation_threshold', 0.5)
            merged_segmentation = obj_ids[np.where(segmentation_maps.max(axis=0) > segm_threshold,
                                                   segmentation_maps.argmax(axis=0) + 1, 0)]

            out_merged['segmentation'] = merged_segmentation

        # Merge other fields
        for key in out_types:
            if key == 'segmentation':
                pass
            else:
                out_merged[key] = {obj_id: out[key] for obj_id, out in out_all.items()}

        return out_merged

    def merge_outputs(self, out_all):
        if hasattr(self.base_tracker_class, 'merge_results'):
            out_merged = self.base_tracker_class.merge_results(out_all)
        else:
            out_merged = self.default_merge(out_all)

        return out_merged

    def initialize(self, image, info: dict) -> dict:
        self.initialized_ids = []
        self.trackers = OrderedDict()

        if len(info['init_object_ids']) == 0:
            return None

        object_ids = info['object_ids']

        init_info_split = self._split_info(info)
        self.trackers = OrderedDict({obj_id: self.create_tracker() for obj_id in object_ids})

        out_all = OrderedDict()
        # Run individual trackers for each object
        for obj_id in info['init_object_ids']:
            start_time = time.time()
            out = self.trackers[obj_id].initialize(image, init_info_split[obj_id])
            if out is None:
                out = {}

            init_default = {'target_bbox': init_info_split[obj_id].get('init_bbox'),
                            'time': time.time() - start_time,
                            'segmentation': init_info_split[obj_id].get('init_mask')}

            out = self._set_defaults(out, init_default)
            out_all[obj_id] = out

        # Merge results
        out_merged = self.merge_outputs(out_all)

        self.initialized_ids = info['init_object_ids'].copy()
        return out_merged

    def track(self, image, info: dict = None) -> dict:
        if info is None:
            info = {}

        prev_output = info.get('previous_output', OrderedDict())

        if info.get('init_object_ids', False):
            init_info_split = self._split_info(info)
            for obj_init_info in init_info_split.values():
                obj_init_info['previous_output'] = prev_output

            info['init_other'] = list(init_info_split.values())[0]['init_other']

        out_all = OrderedDict()
        for obj_id in self.initialized_ids:
            start_time = time.time()

            out = self.trackers[obj_id].track(image, info)

            default = {'time': time.time() - start_time}
            out = self._set_defaults(out, default)
            out_all[obj_id] = out

        # Initialize new
        if info.get('init_object_ids', False):
            for obj_id in info['init_object_ids']:
                if not obj_id in self.trackers:
                    self.trackers[obj_id] = self.create_tracker()

                start_time = time.time()
                out = self.trackers[obj_id].initialize(image, init_info_split[obj_id])
                if out is None:
                    out = {}

                init_default = {'target_bbox': init_info_split[obj_id].get('init_bbox'),
                                'time': time.time() - start_time,
                                'segmentation': init_info_split[obj_id].get('init_mask')}

                out = self._set_defaults(out, init_default)
                out_all[obj_id] = out

            self.initialized_ids.extend(info['init_object_ids'])

        # Merge results
        out_merged = self.merge_outputs(out_all)

        return out_merged

    def visdom_draw_tracking(self, image, box, segmentation):
        if isinstance(box, (OrderedDict, dict)):
            box = [v for k, v in box.items()]
        else:
            box = (box,)
        if segmentation is None:
            self.visdom.register((image, *box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, *box, segmentation), 'Tracking', 1, 'Tracking')
