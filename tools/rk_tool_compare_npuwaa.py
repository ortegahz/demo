import argparse
import glob
import logging
import os

import numpy as np

from utils.logging import set_logging


def cos_sim_calc(array_a, array_b):
    dot = np.dot(array_a, array_b)
    a_norm = np.sqrt(np.square(array_a).sum())
    b_norm = np.sqrt(np.square(array_b).sum())
    cos_sim = dot / (a_norm * b_norm)
    return cos_sim


def run(args):
    path_order = os.path.join(args.dir_root_aa, 'order.txt')
    path_dict = os.path.join(args.dir_root_npu, 'NodeID_To_LayerName.txt')
    with open(path_dict, 'r') as f:
        lines = f.readlines()
    dict_id2name = dict()
    dict_name2id = dict()
    for line in lines[1:]:
        id, name = '', ''
        if len(line.split()) == 2:
            id, name = line.split()
        dict_id2name[id] = name
        dict_name2id[name] = id

    paths_npu = glob.glob(os.path.join(args.dir_root_npu, 'NodeID_*out*.txt'))
    paths_aa = glob.glob(os.path.join(args.dir_root_aa, '*.tensor'))

    with open(path_order, 'r') as f:
        lines_order = f.readlines()

    for line_order in lines_order:
        path_aa_pick, path_npu_pick = '', ''
        idx = line_order.find('_out0_')
        layer_name = line_order[:idx]
        if layer_name not in dict_name2id.keys():
            continue
        id = dict_name2id[layer_name]
        if dict_id2name[id] != layer_name:
            continue
        for path_aa in paths_aa:
            if layer_name in path_aa:
                path_aa_pick = path_aa
                break
        for path_npu in paths_npu:
            key_name_npu = f'NodeID_{id}'
            if key_name_npu in path_npu:
                path_npu_pick = path_npu
                break
        if not path_aa_pick or not path_npu_pick:
            continue
        logging.info((os.path.split(path_npu_pick)[-1], os.path.split(path_aa_pick)[-1]))
        file_name = os.path.split(path_npu_pick)[-1]
        file_name_split = file_name.split('_')
        w, h, c = int(file_name_split[4]), int(file_name_split[6]), int(file_name_split[8])
        with open(path_npu_pick, 'r') as f:
            lines_npu = f.readlines()
        with open(path_aa_pick, 'r') as f:
            lines_aa = f.readlines()
        array_npu = np.array([float(x) for x in lines_npu])
        array_aa = np.array([float(x) for x in lines_aa])
        array_npu = array_npu.reshape((c, h, w))
        array_npu_perm = array_npu.transpose(1, 2, 0)  # c h w --> h w c
        if len(array_npu_perm.flatten()) == len(array_aa):
            cos_sim = cos_sim_calc(array_npu_perm.flatten(), array_aa)
            logging.info(f'cos sim --> {cos_sim}')


def parse_ars():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_root_npu', default='/home/manu/nfs/rv1126/install/rknn_yolov5_demo', type=str)
    parser.add_argument('--dir_root_aa',
                        default='/media/manu/kingstop/workspace/rknn-toolkit/examples/onnx/acfree/snapshot_hq/fp32',
                        type=str)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_ars()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
