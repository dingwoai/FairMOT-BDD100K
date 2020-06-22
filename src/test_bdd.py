from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch
import json

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
import datasets.dataset.bdd100k as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts

class_names = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
            "other person", "trailer", "other vehicle"]

def write_results(filename, results, data_type):
    assert data_type == 'bdd'
    jsons = []
    for name, tlwhs, track_ids, c_ids in results:
        json_result = {}
        json_result['name'] = name
        labels = []
        for tlwh, track_id, c_id in zip(tlwhs, track_ids, c_ids):
            l_dict = {}
            l_dict['category'] = class_names[c_id]
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            l_dict['box2d'] = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
            l_dict['id'] = track_id
            labels.append(l_dict)
        json_result['labels'] = labels
        jsons.append(json_result)
    # print(jsons)
    with open('/data/BDD100K/data/MOT/labels_with_ids/test_sub.json', 'w') as f:
        json.dump(jsons, f)



def eval_det(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    for path, img, img0 in dataloader:
        if frame_id<302:
            frame_id+=1
            continue
        elif frame_id>302:
            break
        else:
            print(frame_id)
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
            # if frame_id>20:
            #     break

        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        dets = tracker.detect(blob, img0)
        # print(path, dets)
        tlbrs = []
        scores = []
        class_ids = []
        for det in dets:
            tlbrs.append(det[:4])
            scores.append(det[4])
            class_ids.append(int(det[5]-1))
        # print(class_ids)
        if show_image or save_dir is not None:
            online_im = vis.plot_detections(img0, tlbrs, scores=None, ids=class_ids)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1

def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    tid_max, tid_temp = 1, 1
    for path, img, img0 in dataloader:
        if frame_id<300:
            frame_id+=1
            continue
        elif frame_id>302:
            break
        else:
            print(frame_id)
        filename = path.split('/')[-1]
        if '0000001' in path:
            tid_max = tid_temp + 1
            print(path, tid_max)
            tracker = JDETracker(opt, frame_rate=frame_rate)
        
        if frame_id % 100 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        # if frame_id >20:
        #     break
        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets = tracker.update_sep3(blob, img0, conf_thres=[0.4, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.5])
        # print(online_targets)
        online_tlwhs = []
        online_ids = []
        online_cids = [] #class id
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id + tid_max
            tcid = t.class_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area:# and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_cids.append(tcid)
            tid_temp = max(tid, tid_temp)
        timer.toc()
        # save results
        results.append((filename, online_tlwhs, online_ids, online_cids))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'bdd'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        # dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        dataloader = datasets.LoadImages(data_root, opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        # meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        # frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        frame_rate = 30
        # eval_det(opt, dataloader, data_type, result_filename,
        #          save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()

    if opt.val_bdd100k:
        seqs_str = '''bdd
                      '''
        data_root = os.path.join(opt.data_dir, 'images/test')
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='BDD_test_all_hrnet32',
         show_image=False,
         save_images=True,
         save_videos=False)
