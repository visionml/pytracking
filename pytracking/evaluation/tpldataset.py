import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList


def TPLDataset():
    return TPLDatasetClass().get_sequence_list()


def TPLDatasetNoOtb():
    return TPLDatasetClass(exclude_otb=True).get_sequence_list()


class TPLDatasetClass(BaseDataset):
    '''Temple.'''
    def __init__(self, exclude_otb=False):
        super().__init__()
        self.base_path = self.env_settings.tpl_path
        self.sequence_info_list = self._get_sequence_info_list(exclude_otb)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])


    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        return Sequence(sequence_info['name'], frames, ground_truth_rect[init_omit:,:])

    def __len__(self):
        '''Overload this function in your evaluation. This should return number of sequences in the evaluation '''
        return len(self.sequence_info_list)


    def _get_sequence_info_list(self, exclude_otb=False):
        sequence_info_list = [
            {"name": "tpl_Skating2", "path": "tpl_Skating2/img", "startFrame": 1, "endFrame": 707, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Skating2/Skating2_gt.txt"},
            {"name": "tpl_Pool_ce3", "path": "tpl_Pool_ce3/img", "startFrame": 1, "endFrame": 124, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Pool_ce3/Pool_ce3_gt.txt"},
            {"name": "tpl_Microphone_ce1", "path": "tpl_Microphone_ce1/img", "startFrame": 1, "endFrame": 204, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Microphone_ce1/Microphone_ce1_gt.txt"},
            {"name": "tpl_Torus", "path": "tpl_Torus/img", "startFrame": 1, "endFrame": 264, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Torus/Torus_gt.txt"},
            {"name": "tpl_Lemming", "path": "tpl_Lemming/img", "startFrame": 1, "endFrame": 1336, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Lemming/Lemming_gt.txt"},
            {"name": "tpl_Eagle_ce", "path": "tpl_Eagle_ce/img", "startFrame": 1, "endFrame": 112, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Eagle_ce/Eagle_ce_gt.txt"},
            {"name": "tpl_Skating_ce2", "path": "tpl_Skating_ce2/img", "startFrame": 1, "endFrame": 497, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Skating_ce2/Skating_ce2_gt.txt"},
            {"name": "tpl_Yo_yos_ce3", "path": "tpl_Yo_yos_ce3/img", "startFrame": 1, "endFrame": 201, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Yo_yos_ce3/Yo-yos_ce3_gt.txt"},
            {"name": "tpl_Board", "path": "tpl_Board/img", "startFrame": 1, "endFrame": 598, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Board/Board_gt.txt"},
            {"name": "tpl_Tennis_ce3", "path": "tpl_Tennis_ce3/img", "startFrame": 1, "endFrame": 204, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Tennis_ce3/Tennis_ce3_gt.txt"},
            {"name": "tpl_SuperMario_ce", "path": "tpl_SuperMario_ce/img", "startFrame": 1, "endFrame": 146, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_SuperMario_ce/SuperMario_ce_gt.txt"},
            {"name": "tpl_Yo_yos_ce1", "path": "tpl_Yo_yos_ce1/img", "startFrame": 1, "endFrame": 235, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Yo_yos_ce1/Yo-yos_ce1_gt.txt"},
            {"name": "tpl_Soccer", "path": "tpl_Soccer/img", "startFrame": 1, "endFrame": 392, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Soccer/Soccer_gt.txt"},
            {"name": "tpl_Fish_ce2", "path": "tpl_Fish_ce2/img", "startFrame": 1, "endFrame": 573, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Fish_ce2/Fish_ce2_gt.txt"},
            {"name": "tpl_Liquor", "path": "tpl_Liquor/img", "startFrame": 1, "endFrame": 1741, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Liquor/Liquor_gt.txt"},
            {"name": "tpl_Plane_ce2", "path": "tpl_Plane_ce2/img", "startFrame": 1, "endFrame": 653, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Plane_ce2/Plane_ce2_gt.txt"},
            {"name": "tpl_Couple", "path": "tpl_Couple/img", "startFrame": 1, "endFrame": 140, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Couple/Couple_gt.txt"},
            {"name": "tpl_Logo_ce", "path": "tpl_Logo_ce/img", "startFrame": 1, "endFrame": 610, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Logo_ce/Logo_ce_gt.txt"},
            {"name": "tpl_Hand_ce2", "path": "tpl_Hand_ce2/img", "startFrame": 1, "endFrame": 251, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Hand_ce2/Hand_ce2_gt.txt"},
            {"name": "tpl_Kite_ce2", "path": "tpl_Kite_ce2/img", "startFrame": 1, "endFrame": 658, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Kite_ce2/Kite_ce2_gt.txt"},
            {"name": "tpl_Walking", "path": "tpl_Walking/img", "startFrame": 1, "endFrame": 412, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Walking/Walking_gt.txt"},
            {"name": "tpl_David", "path": "tpl_David/img", "startFrame": 300, "endFrame": 770, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_David/David_gt.txt"},
            {"name": "tpl_Boat_ce1", "path": "tpl_Boat_ce1/img", "startFrame": 1, "endFrame": 377, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Boat_ce1/Boat_ce1_gt.txt"},
            {"name": "tpl_Airport_ce", "path": "tpl_Airport_ce/img", "startFrame": 1, "endFrame": 148, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Airport_ce/Airport_ce_gt.txt"},
            {"name": "tpl_Tiger2", "path": "tpl_Tiger2/img", "startFrame": 1, "endFrame": 365, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Tiger2/Tiger2_gt.txt"},
            {"name": "tpl_Suitcase_ce", "path": "tpl_Suitcase_ce/img", "startFrame": 1, "endFrame": 184, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Suitcase_ce/Suitcase_ce_gt.txt"},
            {"name": "tpl_TennisBall_ce", "path": "tpl_TennisBall_ce/img", "startFrame": 1, "endFrame": 288, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_TennisBall_ce/TennisBall_ce_gt.txt"},
            {"name": "tpl_Singer_ce1", "path": "tpl_Singer_ce1/img", "startFrame": 1, "endFrame": 214, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Singer_ce1/Singer_ce1_gt.txt"},
            {"name": "tpl_Pool_ce2", "path": "tpl_Pool_ce2/img", "startFrame": 1, "endFrame": 133, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Pool_ce2/Pool_ce2_gt.txt"},
            {"name": "tpl_Surf_ce3", "path": "tpl_Surf_ce3/img", "startFrame": 1, "endFrame": 279, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Surf_ce3/Surf_ce3_gt.txt"},
            {"name": "tpl_Bird", "path": "tpl_Bird/img", "startFrame": 1, "endFrame": 99, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Bird/Bird_gt.txt"},
            {"name": "tpl_Crossing", "path": "tpl_Crossing/img", "startFrame": 1, "endFrame": 120, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Crossing/Crossing_gt.txt"},
            {"name": "tpl_Plate_ce1", "path": "tpl_Plate_ce1/img", "startFrame": 1, "endFrame": 142, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Plate_ce1/Plate_ce1_gt.txt"},
            {"name": "tpl_Cup", "path": "tpl_Cup/img", "startFrame": 1, "endFrame": 303, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Cup/Cup_gt.txt"},
            {"name": "tpl_Surf_ce2", "path": "tpl_Surf_ce2/img", "startFrame": 1, "endFrame": 391, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Surf_ce2/Surf_ce2_gt.txt"},
            {"name": "tpl_Busstation_ce2", "path": "tpl_Busstation_ce2/img", "startFrame": 6, "endFrame": 400, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Busstation_ce2/Busstation_ce2_gt.txt"},
            {"name": "tpl_Charger_ce", "path": "tpl_Charger_ce/img", "startFrame": 1, "endFrame": 298, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Charger_ce/Charger_ce_gt.txt"},
            {"name": "tpl_Pool_ce1", "path": "tpl_Pool_ce1/img", "startFrame": 1, "endFrame": 166, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Pool_ce1/Pool_ce1_gt.txt"},
            {"name": "tpl_MountainBike", "path": "tpl_MountainBike/img", "startFrame": 1, "endFrame": 228, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_MountainBike/MountainBike_gt.txt"},
            {"name": "tpl_Guitar_ce1", "path": "tpl_Guitar_ce1/img", "startFrame": 1, "endFrame": 268, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Guitar_ce1/Guitar_ce1_gt.txt"},
            {"name": "tpl_Busstation_ce1", "path": "tpl_Busstation_ce1/img", "startFrame": 1, "endFrame": 363, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Busstation_ce1/Busstation_ce1_gt.txt"},
            {"name": "tpl_Diving", "path": "tpl_Diving/img", "startFrame": 1, "endFrame": 231, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Diving/Diving_gt.txt"},
            {"name": "tpl_Skating_ce1", "path": "tpl_Skating_ce1/img", "startFrame": 1, "endFrame": 409, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Skating_ce1/Skating_ce1_gt.txt"},
            {"name": "tpl_Hurdle_ce2", "path": "tpl_Hurdle_ce2/img", "startFrame": 27, "endFrame": 330, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Hurdle_ce2/Hurdle_ce2_gt.txt"},
            {"name": "tpl_Plate_ce2", "path": "tpl_Plate_ce2/img", "startFrame": 1, "endFrame": 181, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Plate_ce2/Plate_ce2_gt.txt"},
            {"name": "tpl_CarDark", "path": "tpl_CarDark/img", "startFrame": 1, "endFrame": 393, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_CarDark/CarDark_gt.txt"},
            {"name": "tpl_Singer_ce2", "path": "tpl_Singer_ce2/img", "startFrame": 1, "endFrame": 999, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Singer_ce2/Singer_ce2_gt.txt"},
            {"name": "tpl_Shaking", "path": "tpl_Shaking/img", "startFrame": 1, "endFrame": 365, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Shaking/Shaking_gt.txt"},
            {"name": "tpl_Iceskater", "path": "tpl_Iceskater/img", "startFrame": 1, "endFrame": 500, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Iceskater/Iceskater_gt.txt"},
            {"name": "tpl_Badminton_ce2", "path": "tpl_Badminton_ce2/img", "startFrame": 1, "endFrame": 705, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Badminton_ce2/Badminton_ce2_gt.txt"},
            {"name": "tpl_Spiderman_ce", "path": "tpl_Spiderman_ce/img", "startFrame": 1, "endFrame": 351, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Spiderman_ce/Spiderman_ce_gt.txt"},
            {"name": "tpl_Kite_ce1", "path": "tpl_Kite_ce1/img", "startFrame": 1, "endFrame": 484, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Kite_ce1/Kite_ce1_gt.txt"},
            {"name": "tpl_Skyjumping_ce", "path": "tpl_Skyjumping_ce/img", "startFrame": 1, "endFrame": 938, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Skyjumping_ce/Skyjumping_ce_gt.txt"},
            {"name": "tpl_Ball_ce1", "path": "tpl_Ball_ce1/img", "startFrame": 1, "endFrame": 391, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Ball_ce1/Ball_ce1_gt.txt"},
            {"name": "tpl_Yo_yos_ce2", "path": "tpl_Yo_yos_ce2/img", "startFrame": 1, "endFrame": 454, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Yo_yos_ce2/Yo-yos_ce2_gt.txt"},
            {"name": "tpl_Ironman", "path": "tpl_Ironman/img", "startFrame": 1, "endFrame": 166, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Ironman/Ironman_gt.txt"},
            {"name": "tpl_FaceOcc1", "path": "tpl_FaceOcc1/img", "startFrame": 1, "endFrame": 892, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_FaceOcc1/FaceOcc1_gt.txt"},
            {"name": "tpl_Surf_ce1", "path": "tpl_Surf_ce1/img", "startFrame": 1, "endFrame": 404, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Surf_ce1/Surf_ce1_gt.txt"},
            {"name": "tpl_Ring_ce", "path": "tpl_Ring_ce/img", "startFrame": 1, "endFrame": 201, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Ring_ce/Ring_ce_gt.txt"},
            {"name": "tpl_Surf_ce4", "path": "tpl_Surf_ce4/img", "startFrame": 1, "endFrame": 135, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Surf_ce4/Surf_ce4_gt.txt"},
            {"name": "tpl_Ball_ce4", "path": "tpl_Ball_ce4/img", "startFrame": 1, "endFrame": 538, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Ball_ce4/Ball_ce4_gt.txt"},
            {"name": "tpl_Bikeshow_ce", "path": "tpl_Bikeshow_ce/img", "startFrame": 1, "endFrame": 361, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Bikeshow_ce/Bikeshow_ce_gt.txt"},
            {"name": "tpl_Kobe_ce", "path": "tpl_Kobe_ce/img", "startFrame": 1, "endFrame": 582, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Kobe_ce/Kobe_ce_gt.txt"},
            {"name": "tpl_Tiger1", "path": "tpl_Tiger1/img", "startFrame": 1, "endFrame": 354, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Tiger1/Tiger1_gt.txt"},
            {"name": "tpl_Skiing", "path": "tpl_Skiing/img", "startFrame": 1, "endFrame": 81, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Skiing/Skiing_gt.txt"},
            {"name": "tpl_Tennis_ce1", "path": "tpl_Tennis_ce1/img", "startFrame": 1, "endFrame": 454, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Tennis_ce1/Tennis_ce1_gt.txt"},
            {"name": "tpl_Carchasing_ce4", "path": "tpl_Carchasing_ce4/img", "startFrame": 1, "endFrame": 442, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Carchasing_ce4/Carchasing_ce4_gt.txt"},
            {"name": "tpl_Walking2", "path": "tpl_Walking2/img", "startFrame": 1, "endFrame": 500, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Walking2/Walking2_gt.txt"},
            {"name": "tpl_Sailor_ce", "path": "tpl_Sailor_ce/img", "startFrame": 1, "endFrame": 402, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Sailor_ce/Sailor_ce_gt.txt"},
            {"name": "tpl_Railwaystation_ce", "path": "tpl_Railwaystation_ce/img", "startFrame": 1, "endFrame": 413,
             "nz": 4, "ext": "jpg", "anno_path": "tpl_Railwaystation_ce/Railwaystation_ce_gt.txt"},
            {"name": "tpl_Bee_ce", "path": "tpl_Bee_ce/img", "startFrame": 1, "endFrame": 90, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Bee_ce/Bee_ce_gt.txt"},
            {"name": "tpl_Girl", "path": "tpl_Girl/img", "startFrame": 1, "endFrame": 500, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Girl/Girl_gt.txt"},
            {"name": "tpl_Subway", "path": "tpl_Subway/img", "startFrame": 1, "endFrame": 175, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Subway/Subway_gt.txt"},
            {"name": "tpl_David3", "path": "tpl_David3/img", "startFrame": 1, "endFrame": 252, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_David3/David3_gt.txt"},
            {"name": "tpl_Electricalbike_ce", "path": "tpl_Electricalbike_ce/img", "startFrame": 1, "endFrame": 818,
             "nz": 4, "ext": "jpg", "anno_path": "tpl_Electricalbike_ce/Electricalbike_ce_gt.txt"},
            {"name": "tpl_Michaeljackson_ce", "path": "tpl_Michaeljackson_ce/img", "startFrame": 1, "endFrame": 393,
             "nz": 4, "ext": "jpg", "anno_path": "tpl_Michaeljackson_ce/Michaeljackson_ce_gt.txt"},
            {"name": "tpl_Woman", "path": "tpl_Woman/img", "startFrame": 1, "endFrame": 597, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Woman/Woman_gt.txt"},
            {"name": "tpl_TableTennis_ce", "path": "tpl_TableTennis_ce/img", "startFrame": 1, "endFrame": 198, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_TableTennis_ce/TableTennis_ce_gt.txt"},
            {"name": "tpl_Motorbike_ce", "path": "tpl_Motorbike_ce/img", "startFrame": 1, "endFrame": 563, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Motorbike_ce/Motorbike_ce_gt.txt"},
            {"name": "tpl_Baby_ce", "path": "tpl_Baby_ce/img", "startFrame": 1, "endFrame": 296, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Baby_ce/Baby_ce_gt.txt"},
            {"name": "tpl_Gym", "path": "tpl_Gym/img", "startFrame": 1, "endFrame": 766, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Gym/Gym_gt.txt"},
            {"name": "tpl_Matrix", "path": "tpl_Matrix/img", "startFrame": 1, "endFrame": 100, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Matrix/Matrix_gt.txt"},
            {"name": "tpl_Kite_ce3", "path": "tpl_Kite_ce3/img", "startFrame": 1, "endFrame": 528, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Kite_ce3/Kite_ce3_gt.txt"},
            {"name": "tpl_Fish_ce1", "path": "tpl_Fish_ce1/img", "startFrame": 1, "endFrame": 401, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Fish_ce1/Fish_ce1_gt.txt"},
            {"name": "tpl_Hand_ce1", "path": "tpl_Hand_ce1/img", "startFrame": 1, "endFrame": 401, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Hand_ce1/Hand_ce1_gt.txt"},
            {"name": "tpl_Doll", "path": "tpl_Doll/img", "startFrame": 1, "endFrame": 3872, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Doll/Doll_gt.txt"},
            {"name": "tpl_Carchasing_ce3", "path": "tpl_Carchasing_ce3/img", "startFrame": 1, "endFrame": 572, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Carchasing_ce3/Carchasing_ce3_gt.txt"},
            {"name": "tpl_Thunder_ce", "path": "tpl_Thunder_ce/img", "startFrame": 1, "endFrame": 375, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Thunder_ce/Thunder_ce_gt.txt"},
            {"name": "tpl_Singer2", "path": "tpl_Singer2/img", "startFrame": 1, "endFrame": 366, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Singer2/Singer2_gt.txt"},
            {"name": "tpl_Basketball", "path": "tpl_Basketball/img", "startFrame": 1, "endFrame": 725, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Basketball/Basketball_gt.txt"},
            {"name": "tpl_Hand", "path": "tpl_Hand/img", "startFrame": 1, "endFrame": 244, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Hand/Hand_gt.txt"},
            {"name": "tpl_Cup_ce", "path": "tpl_Cup_ce/img", "startFrame": 1, "endFrame": 338, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Cup_ce/Cup_ce_gt.txt"},
            {"name": "tpl_MotorRolling", "path": "tpl_MotorRolling/img", "startFrame": 1, "endFrame": 164, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_MotorRolling/MotorRolling_gt.txt"},
            {"name": "tpl_Boat_ce2", "path": "tpl_Boat_ce2/img", "startFrame": 1, "endFrame": 412, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Boat_ce2/Boat_ce2_gt.txt"},
            {"name": "tpl_CarScale", "path": "tpl_CarScale/img", "startFrame": 1, "endFrame": 252, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_CarScale/CarScale_gt.txt"},
            {"name": "tpl_Sunshade", "path": "tpl_Sunshade/img", "startFrame": 1, "endFrame": 172, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Sunshade/Sunshade_gt.txt"},
            {"name": "tpl_Football1", "path": "tpl_Football1/img", "startFrame": 1, "endFrame": 74, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Football1/Football1_gt.txt"},
            {"name": "tpl_Singer1", "path": "tpl_Singer1/img", "startFrame": 1, "endFrame": 351, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Singer1/Singer1_gt.txt"},
            {"name": "tpl_Hurdle_ce1", "path": "tpl_Hurdle_ce1/img", "startFrame": 1, "endFrame": 300, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Hurdle_ce1/Hurdle_ce1_gt.txt"},
            {"name": "tpl_Basketball_ce3", "path": "tpl_Basketball_ce3/img", "startFrame": 1, "endFrame": 441, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Basketball_ce3/Basketball_ce3_gt.txt"},
            {"name": "tpl_Toyplane_ce", "path": "tpl_Toyplane_ce/img", "startFrame": 1, "endFrame": 405, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Toyplane_ce/Toyplane_ce_gt.txt"},
            {"name": "tpl_Skating1", "path": "tpl_Skating1/img", "startFrame": 1, "endFrame": 400, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Skating1/Skating1_gt.txt"},
            {"name": "tpl_Juice", "path": "tpl_Juice/img", "startFrame": 1, "endFrame": 404, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Juice/Juice_gt.txt"},
            {"name": "tpl_Biker", "path": "tpl_Biker/img", "startFrame": 1, "endFrame": 180, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Biker/Biker_gt.txt"},
            {"name": "tpl_Boy", "path": "tpl_Boy/img", "startFrame": 1, "endFrame": 602, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Boy/Boy_gt.txt"},
            {"name": "tpl_Jogging1", "path": "tpl_Jogging1/img", "startFrame": 1, "endFrame": 307, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Jogging1/Jogging1_gt.txt"},
            {"name": "tpl_Deer", "path": "tpl_Deer/img", "startFrame": 1, "endFrame": 71, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Deer/Deer_gt.txt"},
            {"name": "tpl_Panda", "path": "tpl_Panda/img", "startFrame": 1, "endFrame": 241, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Panda/Panda_gt.txt"},
            {"name": "tpl_Coke", "path": "tpl_Coke/img", "startFrame": 1, "endFrame": 291, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Coke/Coke_gt.txt"},
            {"name": "tpl_Carchasing_ce1", "path": "tpl_Carchasing_ce1/img", "startFrame": 1, "endFrame": 501, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Carchasing_ce1/Carchasing_ce1_gt.txt"},
            {"name": "tpl_Badminton_ce1", "path": "tpl_Badminton_ce1/img", "startFrame": 1, "endFrame": 579, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Badminton_ce1/Badminton_ce1_gt.txt"},
            {"name": "tpl_Trellis", "path": "tpl_Trellis/img", "startFrame": 1, "endFrame": 569, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Trellis/Trellis_gt.txt"},
            {"name": "tpl_Face_ce2", "path": "tpl_Face_ce2/img", "startFrame": 1, "endFrame": 148, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Face_ce2/Face_ce2_gt.txt"},
            {"name": "tpl_Ball_ce2", "path": "tpl_Ball_ce2/img", "startFrame": 1, "endFrame": 603, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Ball_ce2/Ball_ce2_gt.txt"},
            {"name": "tpl_Skiing_ce", "path": "tpl_Skiing_ce/img", "startFrame": 1, "endFrame": 511, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Skiing_ce/Skiing_ce_gt.txt"},
            {"name": "tpl_Jogging2", "path": "tpl_Jogging2/img", "startFrame": 1, "endFrame": 307, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Jogging2/Jogging2_gt.txt"},
            {"name": "tpl_Bike_ce1", "path": "tpl_Bike_ce1/img", "startFrame": 1, "endFrame": 801, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Bike_ce1/Bike_ce1_gt.txt"},
            {"name": "tpl_Bike_ce2", "path": "tpl_Bike_ce2/img", "startFrame": 1, "endFrame": 812, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Bike_ce2/Bike_ce2_gt.txt"},
            {"name": "tpl_Ball_ce3", "path": "tpl_Ball_ce3/img", "startFrame": 1, "endFrame": 273, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Ball_ce3/Ball_ce3_gt.txt"},
            {"name": "tpl_Girlmov", "path": "tpl_Girlmov/img", "startFrame": 1, "endFrame": 1500, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Girlmov/Girlmov_gt.txt"},
            {"name": "tpl_Bolt", "path": "tpl_Bolt/img", "startFrame": 1, "endFrame": 350, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Bolt/Bolt_gt.txt"},
            {"name": "tpl_Basketball_ce2", "path": "tpl_Basketball_ce2/img", "startFrame": 1, "endFrame": 455, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Basketball_ce2/Basketball_ce2_gt.txt"},
            {"name": "tpl_Bicycle", "path": "tpl_Bicycle/img", "startFrame": 1, "endFrame": 271, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Bicycle/Bicycle_gt.txt"},
            {"name": "tpl_Face_ce", "path": "tpl_Face_ce/img", "startFrame": 1, "endFrame": 620, "nz": 4, "ext": "jpg",
             "anno_path": "tpl_Face_ce/Face_ce_gt.txt"},
            {"name": "tpl_Basketball_ce1", "path": "tpl_Basketball_ce1/img", "startFrame": 1, "endFrame": 496, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Basketball_ce1/Basketball_ce1_gt.txt"},
            {"name": "tpl_Messi_ce", "path": "tpl_Messi_ce/img", "startFrame": 1, "endFrame": 272, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Messi_ce/Messi_ce_gt.txt"},
            {"name": "tpl_Tennis_ce2", "path": "tpl_Tennis_ce2/img", "startFrame": 1, "endFrame": 305, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Tennis_ce2/Tennis_ce2_gt.txt"},
            {"name": "tpl_Microphone_ce2", "path": "tpl_Microphone_ce2/img", "startFrame": 1, "endFrame": 103, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Microphone_ce2/Microphone_ce2_gt.txt"},
            {"name": "tpl_Guitar_ce2", "path": "tpl_Guitar_ce2/img", "startFrame": 1, "endFrame": 313, "nz": 4,
             "ext": "jpg", "anno_path": "tpl_Guitar_ce2/Guitar_ce2_gt.txt"}

        ]

        otb_sequences = ['tpl_Skating2', 'tpl_Lemming', 'tpl_Board', 'tpl_Soccer', 'tpl_Liquor', 'tpl_Couple', 'tpl_Walking', 'tpl_David', 'tpl_Tiger2', 'tpl_Bird', 'tpl_Crossing', 'tpl_MountainBike',
                         'tpl_Diving', 'tpl_CarDark', 'tpl_Shaking', 'tpl_Ironman', 'tpl_FaceOcc1', 'tpl_Tiger1', 'tpl_Skiing', 'tpl_Walking2', 'tpl_Girl', 'tpl_Girlmov', 'tpl_Subway', 'tpl_David3', 'tpl_Woman',
                         'tpl_Gym', 'tpl_Matrix', 'tpl_Doll', 'tpl_Singer2', 'tpl_Basketball', 'tpl_MotorRolling', 'tpl_CarScale', 'tpl_Football1', 'tpl_Singer1', 'tpl_Skating1', 'tpl_Biker',
                         'tpl_Boy', 'tpl_Jogging1', 'tpl_Deer', 'tpl_Panda', 'tpl_Coke', 'tpl_Trellis', 'tpl_Jogging2', 'tpl_Bolt', ]
        if exclude_otb:
            sequence_info_list_nootb = []
            for seq in sequence_info_list:
                if seq['name'] not in otb_sequences:
                    sequence_info_list_nootb.append(seq)

            sequence_info_list = sequence_info_list_nootb

        return sequence_info_list
