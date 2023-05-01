import copy
import csv
import os
import pdb
import time
from collections import defaultdict, deque
import logging

import cv2
import numpy as np

from evaluation.f1 import compute_f1, evaluate_frame
from utils.utils import interpolation
from requesthandler import RequestHandler
from scheduler import Scheduler
import logging

DEBUG = False
# DEBUG = True
YELLOW = (0, 255, 255)
BLACK = (0, 0, 0)


def debug_print(msg):
    """Debug print."""
    if DEBUG:
        print(msg)

aws = RequestHandler()

class main_pipeline():

    def __init__(self, profile_log, scheduler: Scheduler, tracking_parallel=False, invoking_mode="local",
                 profile_traces_save_path=None, mask_flag=True):
        """Load the configs.

        Args
            profile_log: profile log name
        """
        self.writer = csv.writer(open(profile_log, 'w', 1))
        self.writer.writerow(['clip', 'frame_idx'])
        self.profile_traces_save_path = profile_traces_save_path
        self.scheduler = scheduler
        self.tracking_parallel = tracking_parallel
        self.invoking_mode = invoking_mode
        self.mask_flag = mask_flag
        self.frame_slot = 0
        self.frame_interval = 0
        self.frame_last_triggered = 0
        self.update_queue = deque()
        # keep a dictionary mapping frame_id_obj_id to an opencv tracker
        self.parallel_trackers_dict = {}
        self.trackers_dict = {}

    def pipeline_clear(self):
        self.frame_slot = 0
        self.frame_interval = 0
        self.frame_last_triggered = 0
        self.update_queue.clear()
        self.parallel_trackers_dict = {}
        self.trackers_dict = {}

    def init_trackers(self, frame_idx, frame, boxes):
        """Return the tracked bounding boxes on input frame."""
        resolution = (frame.shape[1], frame.shape[0])
        # original shape is 1080, 1920
        frame_copy = cv2.resize(frame, (640, 480))
        # resize use width, height
        self.trackers_dict = {}
        for box in boxes:
            xmin, ymin, xmax, ymax, t, score, obj_id = box
            # t is class of box
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame_copy, (int(xmin * 640 / resolution[0]),
                                      int(ymin * 480 / resolution[1]),
                                      int((xmax - xmin) * 640 / resolution[0]),
                                      int((ymax - ymin) * 480 / resolution[1])))
            key = '_'.join([str(frame_idx), str(obj_id), str(t)])
            self.trackers_dict[key] = tracker

    def init_parallel_trackers(self, frame_idx, frame, boxes):
        '''init parallel tracking'''
        resolution = (frame.shape[1], frame.shape[0])
        # (1080, 1920)
        frame_copy = cv2.resize(frame, (640, 480))
        trackers_dict = {}
        ret_boxes = []
        for index, box in enumerate(boxes):
            xmin, ymin, xmax, ymax, t, score = box
            t = int(float(t))
            # change t from 2.0 to 2
            obj_id = index + 1
            # t is class of box
            ret_boxes.append([int(xmin), int(ymin), int(xmax), int(ymax), int(t), score, obj_id])
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame_copy, (int(xmin * 640 / resolution[0]),
                                      int(ymin * 480 / resolution[1]),
                                      int((xmax - xmin) * 640 / resolution[0]),
                                      int((ymax - ymin) * 480 / resolution[1])))
            key = '_'.join([str(frame_idx), str(obj_id), str(t)])
            trackers_dict[key] = tracker
        return trackers_dict, ret_boxes
    
    def update_object_detection(self, video, frame_idx, frame_end, model_name: str):
        debug_print("updating object detection at frame {}".format(frame_idx))
        if self.invoking_mode == "aws":
            frame_path = video.get_frame_image_name(frame_idx)
            with open(frame_path, "rb") as image:
                image_str = image.read()
                result = aws.sendRequest(model_name, image_str)
                dets = result['dets']
        elif self.invoking_mode == "local":
            dets = video.get_frame_model_detection(model_name, frame_idx)
        else:
            raise Exception
            # frame_time = 1000 // video.frame_rate
            # detect_time = self.detect_time
            # assert frame_time > detect_time
            # frame_needed = int(int((result['model_time'] + self.rtt) * 1000) // (frame_time - detect_time))
        boxes = None
        self.parallel_trackers_dict, boxes = self.init_parallel_trackers(frame_idx, video.get_frame_image(frame_idx), dets)
        if self.tracking_parallel:
            frame_needed = 10
            for i in range(frame_idx + 1, frame_idx + frame_needed + 1):
                if i > frame_end:
                    break
                boxes = self.update_parallel_trackers(video.get_frame_image(i))
        else:
            frame_needed = 0
        self.update_queue.append({"frame_idx": frame_idx + frame_needed, "trackers_dict": self.parallel_trackers_dict, "boxes": boxes})
        return


    def update_trackers(self, frame):
        """Return the tracked bounding boxes on input frame."""
        resolution = (frame.shape[1], frame.shape[0])
        # original 1080 1920
        frame_copy = cv2.resize(frame, (640, 480))
        tracking_time = 0
        boxes = []
        to_delete = []
        for obj, tracker in self.trackers_dict.items():
            _, obj_id, t = obj.split('_')
            start_t = time.perf_counter()
            ok, bbox = tracker.update(frame_copy)
            end_t = time.perf_counter()
            tracking_time += (end_t - start_t)
            if ok:
                # tracking succeded
                x, y, w, h = bbox
                boxes.append([int(x*resolution[0]/640),
                              int(y*resolution[1]/480),
                              int((x+w)*resolution[0]/640),
                              int((y+h)*resolution[1]/480), int(float(t)),
                              1, obj_id])
            else:
                # tracking failed
                # record the trackers that need to be deleted
                to_delete.append(obj)
        for obj in to_delete:
            self.trackers_dict.pop(obj)
        return boxes, tracking_time

    def update_parallel_trackers(self, frame):
        """Return the tracked bounding boxes on input frame."""
        resolution = (frame.shape[1], frame.shape[0])
        frame_copy = cv2.resize(frame, (640, 480))
        start_t = time.time()
        boxes = []
        to_delete = []
        for obj, tracker in self.parallel_trackers_dict.items():
            _, obj_id, t = obj.split('_')
            ok, bbox = tracker.update(frame_copy)
            if ok:
                # tracking succeded
                x, y, w, h = bbox
                boxes.append([int(x*resolution[0]/640),
                              int(y*resolution[1]/480),
                              int((x+w)*resolution[0]/640),
                              int((y+h)*resolution[1]/480), int(float(t)),
                              1, obj_id])
            else:
                # tracking failed
                # record the trackers that need to be deleted
                to_delete.append(obj)
        for obj in to_delete:
            self.parallel_trackers_dict.pop(obj)
        debug_print("tracking used: {}s".format(time.time()-start_t))
        return boxes

    def profile(self, video, frame_start, frame_end, model_name, frame_interval, frame_slot):
        self.pipeline_clear()
        pipeline_result = defaultdict(list)
        pipeline_result[frame_start] = video.get_frame_detection(frame_start)
        self.init_trackers(frame_start, video.get_frame_image(frame_start), video.get_frame_detection(frame_start))
        interval_elipsed = 1
        frame_last_triggered = frame_start
        for i in range(frame_start + 1, frame_end + 1):
            if interval_elipsed == frame_slot:
                debug_print("interval end, renew interval info")
                interval_elipsed = 0
            debug_print("processing frame {}".format(i))
            assert i <= frame_last_triggered + frame_interval
            if i == frame_last_triggered + frame_interval or interval_elipsed == 0:
                debug_print("frame triggered at idx {}".format(i))
                self.update_object_detection(video, i, frame_end, model_name)
                frame_last_triggered = i
            new_boxes = None
            if len(self.update_queue) > 0 and i == self.update_queue[0]["frame_idx"]:
                debug_print("update from queue at {}".format(i))
                for item in self.update_queue:
                    debug_print("update index {}".format(item['frame_idx']))
                self.trackers_dict = self.update_queue[0]['trackers_dict']
                new_boxes = self.update_queue[0]['boxes']
                self.update_queue.popleft()
            else:
                new_boxes, t = self.update_trackers(video.get_frame_image(i))
            pipeline_result[i] = new_boxes
            interval_elipsed += 1
        f1, precison, recall = eval_pipeline_accuracy(frame_start, frame_end, video.get_video_detection(), pipeline_result)
        return f1, precison, recall

    def pipeline(self, clip, video, frame_start, frame_end, target_accuracy):
        frames_log = []
        # log information on every frame
        interval_log = []
        # log information on every interval
        pipeline_result = defaultdict(list)
        # result of every frame i(bounding box...)
        triggered = set()
        model_list = []
        # list of frames sent to aws lambda
        cost = []
        interval_accuracy = []
        interval_frame_rate = []
        interval_cost = 0
        self.pipeline_clear()

        pipeline_result[frame_start] = video.get_frame_detection(frame_start)
        self.scheduler.renew(pipeline_result[frame_start])
        triggered.add(frame_start)

        self.init_trackers(frame_start, video.get_frame_image(frame_start),
                           pipeline_result[frame_start])
        info = self.scheduler.get_time_interval_info(target_accuracy, frame_start)
        interval_log.append(info)
        single_cost = info["interval_cost"]
        self.frame_slot = info['frame_slot']
        interval_elipsed = 1
        last_interval_start = frame_start


        frame_log = {
            'frame id': frame_start,
            'tracking error': 0,
            'detection': pipeline_result[frame_start],
            'last_triggered': frame_start,
            'frame_interval': info["frame_interval"],
            'frame_slot': self.frame_slot,
            'last_time_slot_start': frame_start
        }
        self.frame_last_triggered = frame_start
        self.frame_interval = info["frame_interval"]
        frames_log.append(frame_log)

        tracking_t_elapsed = list()

        # run the pipeline for the rest of the frames
        for i in range(frame_start + 1, frame_end + 1):
            # update interval
            if interval_elipsed == self.frame_slot:
                debug_print("interval end, renew interval info")
                f1, precison, recall = eval_pipeline_accuracy(last_interval_start, i, video.get_video_detection(), pipeline_result)
                interval_accuracy.append({'f1': f1, 'precision': precison, 'recall': recall})
                interval_frame_rate.append(self.frame_interval)
                model_list.append(info["model"])
                last_interval_start = i
                info = self.scheduler.get_time_interval_info(target_accuracy, i)
                interval_log.append(info)
                cost.append(interval_cost)
                single_cost = info["interval_cost"]
                interval_cost = 0
                self.frame_slot = info['frame_slot']
                interval_elipsed = 0
            debug_print("processing frame {}".format(i))
            # trigger frames
            assert i <= self.frame_last_triggered + self.frame_interval
            if i == self.frame_last_triggered + self.frame_interval or interval_elipsed == 0:
                debug_print("frame triggered at idx {}".format(i))
                triggered.add(i)
                interval_cost += single_cost
                self.frame_interval = info["frame_interval"]
                model_name = info["model"]
                self.update_object_detection(video, i, frame_end, model_name)
                self.frame_last_triggered = i
            new_boxes = None
            if len(self.update_queue) > 0 and i == self.update_queue[0]["frame_idx"]:
                debug_print("update from queue at {}".format(i))
                for item in self.update_queue:
                    debug_print("update index {}".format(item['frame_idx']))
                self.trackers_dict = self.update_queue[0]['trackers_dict']
                new_boxes = self.update_queue[0]['boxes']
                logging.info("update detect at frame {} successed".format(i))
                self.update_queue.popleft()
            else:
                new_boxes, t = self.update_trackers(video.get_frame_image(i))
                tracking_t_elapsed.append(t)
            pipeline_result[i] = new_boxes
            self.scheduler.renew(pipeline_result[i])
            frame_log = {
                'frame id': i,
                'tracking error': 0,
                'detection': pipeline_result[i],
                'last_triggered': self.frame_last_triggered,
                'frame_interval': self.frame_interval,
                'frame_slot': self.frame_slot,
                'last_time_slot_start': i - interval_elipsed
            }
            frames_log.append(frame_log)
            interval_elipsed += 1
        cost.append(interval_cost)
        f1, precison, recall = eval_pipeline_accuracy(last_interval_start, frame_end, video.get_video_detection(), pipeline_result)
        interval_accuracy.append({'f1': f1, 'precision': precison, 'recall': recall})
        interval_frame_rate.append(info['frame_interval'])
        model_list.append(info["model"])
        # last interval may be not complete
        averaged_tracking_time = (sum(tracking_t_elapsed, 0) / len(tracking_t_elapsed)) if len(tracking_t_elapsed) else 0
        # change cost to single invocatioin cost
        total_cost = sum(cost, 0)
        f1, precison, recall = eval_pipeline_accuracy(frame_start, frame_end, video.get_video_detection(), pipeline_result)
        return frames_log, pipeline_result, averaged_tracking_time, triggered, f1, precison, recall, total_cost, cost, interval_accuracy, interval_log, interval_frame_rate, model_list

def frame_difference(old_frame, new_frame, bboxes_last_triggered, bboxes,
                     thresh=35):
    """Compute the sum of pixel differences which are greater than thresh."""
    # thresh = 35 is used in Glimpse paper
    # pdb.set_trace()
    start_t = time.time()
    diff = np.absolute(new_frame.astype(int) - old_frame.astype(int))
    mask = np.greater(diff, thresh)
    pix_change = np.sum(mask)
    time_elapsed = time.time() - start_t
    debug_print('frame difference used: {}'.format(time_elapsed*1000))
    pix_change_obj = 0
    # obj_region = np.zeros_like(new_frame)
    # for box in bboxes_last_triggered:
    #     xmin, ymin, xmax, ymax = box[:4]
    #     obj_region[int(ymin):int(ymax), int(xmin):int(xmax)] = 1
    # for box in bboxes:
    #     xmin, ymin, xmax, ymax = box[:4]
    #     obj_region[int(ymin):int(ymax), int(xmin):int(xmax)] = 1
    # pix_change_obj += np.sum(mask * obj_region)
    pix_change_bg = pix_change - pix_change_obj

    # cv2.imshow('frame diff', np.repeat(
    #     mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8))
    # cv2.moveWindow('frame diff', 1280, 0)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    # cv2.destroyWindow('frame diff')

    return pix_change, pix_change_obj, pix_change_bg, time_elapsed


def tracking_boxes(vis, oldFrameGray, newFrameGray, new_frame_id, old_boxes,
                   tracking_error_thresh):
    """
    Tracking the bboxes between frames via optical flow.

    Arg
        vis(numpy array): an BGR image which helps visualization
        oldFrameGray(numpy array): a grayscale image of previous frame
        newFrameGray(numpy array): a grayscale image of current frame
        new_frame_id(int): frame index
        old_boxes(list): a list of boxes in previous frame
        tracking_error_thresh(float): tracking error threshold
    Return
        tracking status(boolean) - tracking success or failure
        new bboxes tracked by optical flow
    """
    # define colors for visualization
    yellow = (0, 255, 255)
    black = (0, 0, 0)

    # define optical flow parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2,  # 5,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))

    # define good feature compuration parameters
    feature_params = dict(maxCorners=50, qualityLevel=0.01,
                          minDistance=7, blockSize=7)

    # mask = np.zeros_like(oldFrameGray)
    start_t = time.time()
    old_corners = []
    for x, y, xmax, ymax, t, score, obj_id in old_boxes:
        # mask[y:ymax, x:xmax] = 255
        corners = cv2.goodFeaturesToTrack(oldFrameGray[y:ymax, x:xmax],
                                          **feature_params)
        if corners is not None:
            corners[:, 0, 0] = corners[:, 0, 0] + x
            corners[:, 0, 1] = corners[:, 0, 1] + y
            old_corners.append(corners)
    # print('compute feature {}seconds'.format(time.time() - start_t))
    if not old_corners:
        # cannot find available corners and treat as objects disappears
        return True, [], 0
    else:
        old_corners = np.concatenate(old_corners)

    # old_corners = cv2.goodFeaturesToTrack(oldFrameGray, mask=mask,
    #                                       **feature_params)
    # old_corners = cv2.goodFeaturesToTrack(oldFrameGray, 26, 0.01, 7,
    #                                       mask=mask)
    new_corners, st, err = cv2.calcOpticalFlowPyrLK(oldFrameGray, newFrameGray,
                                                    old_corners, None,
                                                    **lk_params)
    # old_corners_r, st, err = cv2.calcOpticalFlowPyrLK(newFrameGray,
    #                                                   oldFrameGray,
    #                                                   old_corners, None,
    #                                                   **lk_params)
    # d = abs(old_corners-old_corners_r).reshape(-1, 2).max(-1)
    # good = d < 1
    # new_corners_copy = new_corners.copy()
    # pdb.set_trace()
    new_corners = new_corners[st == 1].reshape(-1, 1, 2)
    old_corners = old_corners[st == 1].reshape(-1, 1, 2)
    # new_corners = new_corners[good]
    # old_corners = old_corners[good]

    for new_c, old_c in zip(new_corners, old_corners):
        # new corners in yellow circles
        cv2.circle(vis, (new_c[0][0], new_c[0][1]), 5, yellow, -1)
        # old corners in black circles
        cv2.circle(vis, (old_c[0][0], old_c[0][1]), 5, black, -1)

    new_boxes = []

    for x, y, xmax, ymax, t, score, obj_id in old_boxes:
        indices = []
        for idx, (old_c, new_c) in enumerate(zip(old_corners, new_corners)):
            if old_c[0][0] >= x and old_c[0][0] <= xmax and \
               old_c[0][1] >= y and old_c[0][1] <= ymax:
                indices.append(idx)
        if not indices:
            debug_print('frame {}: object {} disappear'.format(new_frame_id,
                                                               obj_id))
            continue
        indices = np.array(indices)

        # checking tracking error threshold condition
        displacement_vectors = []
        dist_list = []
        for old_corner, new_corner in zip(old_corners[indices],
                                          new_corners[indices]):
            dist_list.append(np.linalg.norm(new_corner-old_corner))
            displacement_vectors.append(new_corner-old_corner)
        tracking_err = np.std(dist_list)
        # print('tracking error:', tracking_err)
        if tracking_err > tracking_error_thresh:
            # tracking failure, this is a trigger frame
            debug_print('frame {}: '
                        'object {} std {} > tracking error thresh {}, '
                        'tracking fails'.format(new_frame_id, obj_id,
                                                np.std(dist_list),
                                                tracking_error_thresh))
            return False, [], tracking_err

        # update bouding box translational movement and uniform scaling
        # print('corner number:', old_corners[indices].shape)
        affine_trans_mat, inliers = cv2.estimateAffinePartial2D(
            old_corners[indices], new_corners[indices])
        if affine_trans_mat is None or np.isnan(affine_trans_mat).any():
            # the bbox is too small and not enough good features obtained to
            # compute reliable affine transformation matrix.
            # consider the object disappeared
            continue

        assert affine_trans_mat.shape == (2, 3)
        # print('old box:', x, y, xmax, ymax)
        # print(affine_trans_mat)
        scaling = np.linalg.norm(affine_trans_mat[:, 0])
        translation = affine_trans_mat[:, 2]
        new_x = int(np.round(scaling * x + translation[0]))
        new_y = int(np.round(scaling * y + translation[1]))
        new_xmax = int(np.round(scaling * xmax + translation[0]))
        new_ymax = int(np.round(scaling * ymax + translation[1]))
        # print('new box:', new_x, new_y, new_xmax, new_ymax)
        if new_x >= vis.shape[1] or new_xmax <= 0:
            # object disappears from the right/left of the screen
            continue
        if new_y >= vis.shape[0] or new_ymax <= 0:
            # object disappears from the bottom/top of the screen
            continue

        # The bbox are partially visible in the screen
        if new_x < 0:
            new_x = 0
        if new_xmax > vis.shape[1]:
            new_xmax = vis.shape[1]
        if new_y < 0:
            new_y = 0
        if new_ymax > vis.shape[0]:
            new_ymax = vis.shape[0]
        assert 0 <= new_x <= vis.shape[1], "new_x {} is out of [0, {}]".format(
            new_x, vis.shape[1])
        assert 0 <= new_xmax <= vis.shape[1], "new_xmax {} is out of [0, {}]"\
            .format(new_xmax, vis.shape[1])
        assert 0 <= new_y <= vis.shape[0], "new_y {} is out of [0, {}]".format(
            new_y, vis.shape[0])
        assert 0 <= new_ymax <= vis.shape[0], "new_ymax {} is out of [0, {}]"\
            .format(new_ymax, vis.shape[0])
        # pdb.set_trace()
        new_boxes.append([new_x, new_y, new_xmax, new_ymax, t, score, obj_id])
        # cv2.rectangle(vis, (x, y), (xmax, ymax), black, 2)
        # cv2.rectangle(vis, (new_x, new_y), (new_xmax, new_ymax), yellow, 2)

    # img_title = 'frame {}'.format(new_frame_id)
    # cv2.imshow(img_title, vis)
    # cv2.moveWindow(img_title, 0, 0)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    # else:
    #     cv2.destroyWindow(img_title)
    return True, new_boxes, 0


def eval_pipeline_accuracy(frame_start, frame_end,
                           gt_annot, dt_glimpse, iou_thresh=0.5):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    gt_cn = 0
    dt_cn = 0
    for i in range(frame_start, frame_end + 1):
        gt_boxes_final = gt_annot[i].copy()
        dt_boxes_final = dt_glimpse[i].copy()
        gt_cn += len(gt_boxes_final)
        dt_cn += len(dt_boxes_final)
        tp[i], fp[i], fn[i] = evaluate_frame(gt_boxes_final, dt_boxes_final,
                                             iou_thresh)

    tp_total = sum(tp.values())
    fn_total = sum(fn.values())
    fp_total = sum(fp.values())

    return compute_f1(tp_total, fp_total, fn_total)


def object_appearance(start, end, gt):
    """Take start frame, end frame, and groundtruth.

    Return
        object to frame range (dict)
        frame id to new object id (dict)

    """
    obj_to_frame_range = dict()
    frame_to_new_obj = dict()
    for frame_id in range(int(start), int(end)+1):
        if frame_id not in gt:
            continue
        boxes = gt[frame_id]
        for box in boxes:
            try:
                obj_id = int(box[-1])
            except ValueError:
                obj_id = box[-1]

            if obj_id in obj_to_frame_range:
                start, end = obj_to_frame_range[obj_id]
                obj_to_frame_range[obj_id][0] = min(int(frame_id), start)
                obj_to_frame_range[obj_id][1] = max(int(frame_id), end)
            else:
                obj_to_frame_range[obj_id] = [int(frame_id), int(frame_id)]

    for obj_id in obj_to_frame_range:
        if obj_to_frame_range[obj_id][0] in frame_to_new_obj:
            frame_to_new_obj[obj_to_frame_range[obj_id][0]].append(obj_id)
        else:
            frame_to_new_obj[obj_to_frame_range[obj_id][0]] = [obj_id]

    return obj_to_frame_range, frame_to_new_obj


def compute_target_frame_rate(frame_rate_list, f1_list, target_f1=0.9):
    """Compute target frame rate when target f1 is achieved."""
    index = frame_rate_list.index(max(frame_rate_list))
    f1_list_normalized = [x/f1_list[index] for x in f1_list]
    result = [(y, x) for x, y in sorted(zip(f1_list_normalized,
                                            frame_rate_list))]
    # print(list(zip(frame_rate_list,f1_list_normalized)))
    frame_rate_list_sorted = [x for (x, _) in result]
    f1_list_sorted = [y for (_, y) in result]
    index = next(x[0] for x in enumerate(f1_list_sorted) if x[1] > target_f1)
    if index == 0:
        target_frame_rate = frame_rate_list_sorted[0]
        return target_frame_rate, -1, f1_list_sorted[index], -1,\
            frame_rate_list_sorted[index]
    else:
        point_a = (f1_list_sorted[index-1], frame_rate_list_sorted[index-1])
        point_b = (f1_list_sorted[index], frame_rate_list_sorted[index])

        target_frame_rate = interpolation(point_a, point_b, target_f1)
        return target_frame_rate, f1_list_sorted[index - 1], \
            f1_list_sorted[index], frame_rate_list_sorted[index-1], \
            frame_rate_list_sorted[index]
