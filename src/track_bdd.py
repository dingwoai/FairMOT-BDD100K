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

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.bdd100k as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts

# class_names = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
#             "other person", "trailer", "other vehicle"]
class_names_valid = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]

def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    elif data_type == 'bdd':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,{c},-1,-1\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, class_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, c_id in zip(tlwhs, track_ids, class_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, c=c_id)
                f.write(line)
    logger.info('save results to {}'.format(filename))

def eval_det(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    for path, img, img0 in dataloader:
        if frame_id<1120:
            frame_id+=1
            continue
        elif frame_id>1120:
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


def read_gt_txts(gt_path):
    res = []
    with open(gt_path, 'r') as f:
        for line in f.readlines():
            linelist = line.split(' ')
            tlwh = np.array(tuple(map(float, linelist[2:6])))
            tlwh[0]*=1280
            tlwh[2]*=1280
            tlwh[1]*=720
            tlwh[3]*=720
            tlwh[0]-=(tlwh[2]/2)    ## convert from center to top-left
            tlwh[1]-=(tlwh[3]/2)
            tid = int(linelist[1])
            cid = int(linelist[0])
            res.append((tlwh, tid, cid))
    return res

def eval_seq(opt, dataloader, data_type, result_filename, gt_filename, save_dir=None, show_image=True, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    gts = []
    frame_id = 0
    tid_max, tid_temp = 1, 1
    for path, img, img0 in dataloader:
        if frame_id % 100 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
            # if frame_id>400:
            #     break
        if '0000001' in path:
            tid_max = tid_temp
            tracker = JDETracker(opt, frame_rate=frame_rate)
        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        # online_targets = tracker.update(blob, img0)
        online_targets = tracker.update_sep3(blob, img0, conf_thres=[0.4, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.5])  ## use class-separated tracker
        # print(online_targets)
        online_tlwhs = []
        online_ids = []
        online_cids = [] #class id
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id + tid_max
            tcid = t.class_id
            # vertical = tlwh[2] / tlwh[3] > 1.6
            # if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
            if tlwh[2] * tlwh[3] > opt.min_box_area:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_cids.append(tcid)
            tid_temp = max(tid, tid_temp)
        timer.toc()
        gt_tlwhs = []
        gt_ids = []
        gt_cids = []
        gt_path = path.replace('images', 'labels_with_ids').replace('jpg', 'txt')
        gt_targets = read_gt_txts(gt_path)
        for (tlwh, tid, tcid) in gt_targets:
            gt_tlwhs.append(tlwh)
            gt_ids.append(tid)
            gt_cids.append(tcid)
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids, online_cids))
        # save gts
        gts.append((frame_id + 1, gt_tlwhs, gt_ids, gt_cids))
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
    # save gts
    write_results(gt_filename, gts, data_type)
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
        gt_filename = os.path.join(result_root, '{}_gt.txt'.format(seq))
        frame_rate = 30
        # eval_det(opt, dataloader, data_type, result_filename,
        #          save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename, gt_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type, gt_filename)
        # accs.append(evaluator.eval_file(result_filename))
        for i in range(len(class_names_valid)):
            accs.append(evaluator.eval_file_sp(i, result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, names=class_names_valid, metrics=metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    print('mMOTA:', summary['mota'].mean())
    # Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()

    if opt.val_bdd100k:
        seqs_str = '''bdd
                      '''
        data_root = os.path.join(opt.data_dir, 'images/val')
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='BDD_val_all_dla34',
         show_image=False,
         save_images=False,
         save_videos=False)

    