import numpy as np
import os
import json
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text


class AVisTDataset(BaseDataset):
    """
    AVisT evaluation dataset consisting of 120 videos

    Publication:
        AVisT: A Benchmark for Visual Object Tracking in Adverse Visibility
        Mubashir Noman, Wafa Al Ghallabi, Daniya Najiha, Christoph Mayer, Akshay Dudhane, Martin Danelljan, Hisham Cholakkal, Salman Khan, Luc Van Gool, Fahad Shahbaz Khan
        arXiv, 2022
        https://arxiv.org/pdf/2208.06888.pdf

    Download the dataset from https://sites.google.com/view/avist-benchmark
    """
    def __init__(self, attribute=None):
        super().__init__()
        self.base_path = self.env_settings.avist_path
        self.sequence_list = self._get_sequence_list()

        self.att_dict = None

        if attribute is not None:
            self.sequence_list = self._filter_sequence_list_by_attribute(attribute, self.sequence_list)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/anno/{}.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=' ', dtype=np.float64)

        occlusion_label_path = '{}/full_occlusion/{}_full_occlusion.txt'.format(self.base_path, sequence_name)

        # NOTE: pandas backed seems super super slow for loading occlusion/oov masks
        full_occlusion = load_text(str(occlusion_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        out_of_view_label_path = '{}/out_of_view/{}_out_of_view.txt'.format(self.base_path, sequence_name)
        out_of_view = load_text(str(out_of_view_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        frames_path = '{}/sequences/{}'.format(self.base_path, sequence_name)

        frames_list = ['{}/img_{:05d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        return Sequence(sequence_name, frames_list, 'avist', ground_truth_rect.reshape(-1, 4), target_visible=target_visible)

    def get_attribute_names(self, mode='short'):
        if self.att_dict is None:
            self.att_dict = self._load_attributes()

        names = self.att_dict['att_name_short'] if mode == 'short' else self.att_dict['att_name_long']
        return names

    def _load_attributes(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               'dataset_attribute_specs', 'avist_attributes.json'), 'r') as f:
            att_dict = json.load(f)
        return att_dict

    def _filter_sequence_list_by_attribute(self, att, seq_list):
        if self.att_dict is None:
            self.att_dict = self._load_attributes()

        if att not in self.att_dict['att_name_short']:
            if att in self.att_dict['att_name_long']:
                att = self.att_dict['att_name_short'][self.att_dict['att_name_long'].index(att)]
            else:
                raise ValueError('\'{}\' attribute invalid.')

        return [s for s in seq_list if att in self.att_dict[s]]

    def _get_anno_frame_path(self, seq_path, frame_name):
        return os.path.join(seq_path, frame_name)  # frames start from 1

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = ['Phasmid_insect_camouflage',
                         'air_show',
                         'airplane_in_rain',
                         'airplane_in_sandstorm',
                         'airplane_in_smoke',
                         'ambulance_in_night_1',
                         'ambulance_in_night_2',
                         'animal_in_desert',
                         'animal_in_wildfire',
                         'archival_video_man',
                         'archival_video_rollercoster',
                         'armyman_camouflage',
                         'badminton_net_in_heavy_storm',
                         'ball_fast_shot',
                         'balloon_fight',
                         'balloon_man',
                         'bear_in_snow',
                         'bench_in_hailstorm',
                         'beyblade_competition',
                         'bicycle_in_glaring_sun',
                         'black_car_in_sand_storm',
                         'black_cat_1',
                         'blue_bus_in_heavy_rain',
                         'boat_in_sea_storm',
                         'boats_race',
                         'boy_in_swimming_pool',
                         'bulldozer_in_snow',
                         'burj_khalifa_show',
                         'butterfly_dance',
                         'caemoflaging_octopus',
                         'car_haze_storm',
                         'car_in_desert',
                         'car_in_fire_smoke',
                         'car_in_fog_shaking_cam',
                         'car_in_hurricane',
                         'car_in_hurricane_2',
                         'car_in_hurricane_3',
                         'car_in_smoke',
                         'car_in_smoke_night',
                         'cattle_heavy_rain',
                         'copperhead_snake',
                         'crazy_skiing',
                         'cycling_snow_low_visibility',
                         'deer_in_fog',
                         'ducklings_stairs',
                         'elephant_in_sand',
                         'fast_football_in_haze',
                         'firefighters_in_heavy_smoke',
                         'fish_in_sea',
                         'fish_in_sea_2',
                         'flag_heavy_fog_snow',
                         'flatfish_0',
                         'flatfish_1',
                         'flatfish_4',
                         'flight_black_and_white_footage',
                         'flounder_6',
                         'flounder_9',
                         'flying_bees_1',
                         'flying_bees_2',
                         'girl_in_hula_hoop',
                         'grasshopper_0',
                         'helicopter_in_dust_storm',
                         'helicopter_in_hurricane',
                         'helicoptor_in_firesmoke',
                         'helicoptor_in_tornado',
                         'jaguars_fighting',
                         'kids_on_swings',
                         'kite_flying_fog',
                         'lady_dancing_in_fog',
                         'lichen_katydid',
                         'man_in_heavy_blizzard',
                         'man_in_hula_hoop_night',
                         'man_in_sandstorm',
                         'man_in_whirlpool',
                         'monkey_fight',
                         'monkey_on_tree',
                         'motorcycle2_in_heavy_rain',
                         'motorcycle_in_sandstorm',
                         'peacock_dance',
                         'penguin_in_fog',
                         'person_in_sea_rage',
                         'person_walking_in_snow_1',
                         'plaice',
                         'plane_landing_heavy_fog',
                         'polar_bear',
                         'polar_bear_3',
                         'pygmy_seahorse_1',
                         'pygmy_seahorse_2',
                         'rally_in_smoke_night',
                         'rusty_spotted_cat_1',
                         'sailing_thunderstorm',
                         'scorpion_camouflage',
                         'seagulls_flying',
                         'ship_heavy_fog',
                         'ship_in_sea_storm',
                         'ship_in_thunderstorm_sea',
                         'skateboard_in_rain',
                         'skydiving_1',
                         'small_cricket_ball_fast',
                         'smallfish',
                         'snow_leopard_2',
                         'spider_camouflage',
                         'spider_camouflage_2',
                         'stick_insect_0',
                         'stick_insect_1',
                         'stonefish',
                         'surfer_in_bioluminescent_sea',
                         'surfer_in_storm_1',
                         'surfer_in_storm_2',
                         'surfing_in_fog_3',
                         'swimming_competition_old_footage',
                         'train_in_dense_fog',
                         'umbrella_in_heavy_storm',
                         'underwater_nuclear_burst',
                         'vehichle_in_sun_glare',
                         'vehicle_in_fire',
                         'water_splashing',
                         'white_signboard_in_storm',
                         'windmill_in_tornado',
                         'zebra_in_water']
        return sequence_list
