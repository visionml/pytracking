import numpy as np


def convert_vot_anno_to_rect(vot_anno, type):
    if len(vot_anno) == 4:
        return vot_anno

    if type == 'union':
        x1 = min(vot_anno[0::2])
        x2 = max(vot_anno[0::2])
        y1 = min(vot_anno[1::2])
        y2 = max(vot_anno[1::2])
        return [x1, y1, x2 - x1, y2 - y1]
    elif type == 'preserve_area':
        if len(vot_anno) != 8:
            raise ValueError

        vot_anno = np.array(vot_anno)
        cx = np.mean(vot_anno[0::2])
        cy = np.mean(vot_anno[1::2])

        x1 = min(vot_anno[0::2])
        x2 = max(vot_anno[0::2])
        y1 = min(vot_anno[1::2])
        y2 = max(vot_anno[1::2])

        A1 = np.linalg.norm(vot_anno[0:2] - vot_anno[2: 4]) * np.linalg.norm(vot_anno[2: 4] - vot_anno[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1

        x = cx - 0.5*w
        y = cy - 0.5*h
        return [x, y, w, h]
    else:
        raise ValueError
