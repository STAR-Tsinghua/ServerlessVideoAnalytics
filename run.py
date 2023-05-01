import csv
import os
import json
import time

from main_pipeline import main_pipeline
from videos import get_dataset_class, get_seg_paths
import scheduler

scheduler_dict = {
    "baseline": scheduler.BaselineScheduler,
    "adaptive": scheduler.AdaptiveScheduler
}

DEBUG=False
def debug_print(*args):
    if DEBUG:
        print(*args)

def pipeline(dataset_class, seg_paths, runtime_cfg):
    pipeline_cfg = runtime_cfg["pipeline_cfg"]
    scheduler_class = scheduler_dict[pipeline_cfg["scheduler"]]()
    target_accuracy = pipeline_cfg["target_accuracy"]
    pipeline_mode = pipeline_cfg["pipeline_mode"]
    invoking_mode = pipeline_cfg["invoking_mode"]
    model_list = pipeline_cfg["model_list"]
    tracking_parallel = False
    if pipeline_mode == "tracking":
        tracking_parallel = True
    # load configs
    for seg_path in seg_paths:
        print(seg_path)
        seg_name = os.path.basename(seg_path)
        pipeline = main_pipeline(f"{seg_path}/result/profile.csv", scheduler_class, tracking_parallel, invoking_mode)
        f_out = open(f"{seg_path}/result/{pipeline_cfg['scheduler']}_result.csv", 'w', 1)
        writer = csv.writer(f_out)
        writer.writerow(["video_name", 'averaged_tracking_time', 'f1', 'precision', 'recall', 'target_accuracy', 'cost', 'triggered_times', 'pipeline_time'])
        print(seg_path)
        pipeline.scheduler.load_video_profile(seg_path)
        # loading videos
        video = dataset_class(seg_path, seg_name, model_list, runtime_cfg['ground_truth_model'], filter_flag=False)
        start_frame = 1
        end_frame = video.end_frame_index
        if DEBUG:
            start_frame = 1601
        print('Evaluate {} start={} end={}'.format(
            seg_name, start_frame, end_frame))
        start_t = time.perf_counter()
        frames_log, dt_glimpse, averaged_tracking_time, triggered, f1, precision, recall, total_cost, cost, interval_accuracy, interval_log, interval_frame_rate, model_list = pipeline.pipeline(seg_name, video, start_frame, end_frame, target_accuracy)
        end_t = time.perf_counter()
        interval_f1 = [item['f1'] for item in interval_accuracy]
        interval_recall = [item['recall'] for item in interval_accuracy]
        pipelien_time = end_t - start_t
        
        # start_t = time.perf_counter()
        # f1, precision, recall = pipeline.profile(video, start_frame, end_frame, "yolov5x", 1, 100)
        # end_t = time.perf_counter()
        # print(f"profile uses {end_t - start_t}, {f1}, {precision}, {recall}")

        writer.writerow([seg_name, averaged_tracking_time, f1, precision, recall, target_accuracy, total_cost, len(triggered), pipelien_time])
        with open(f"{seg_path}/result/{pipeline_cfg['scheduler']}_interval_information.csv", 'w') as f:
            interval_writer = csv.writer(f)
            interval_writer.writerow(['cost', *cost])
            interval_writer.writerow(['interval_f1', *interval_f1])
            interval_writer.writerow(['interval_recall', *interval_recall])
            interval_writer.writerow(['interval_frame_rate', *interval_frame_rate])
            interval_writer.writerow(['model', *model_list])
        print(cost)
        print(interval_f1)
        print(interval_recall)
        print(interval_frame_rate)
        print(f"pipeline uses {pipelien_time}, {f1}, {precision}, {recall}")
        if DEBUG:
            print("{} pipeline ended".format(seg_name))
            print("target accuracy is {}".format(target_accuracy))
            print("average tracking time is {}".format(averaged_tracking_time))
            print("average f1 score is {}".format(f1))
            print("average recall rate is {}".format(recall))
            print("average precision is {}".format(precision))
            print("overall_cost is {}".format(total_cost))
            print("triggered {}".format(len(triggered)))

def iteration(dataset_class, seg_paths, runtime_cfg):
    iteration_cfg = runtime_cfg["iteration_cfg"]
    scheduler_class = scheduler_dict[iteration_cfg["scheduler"]]()
    target_accuracy_list = iteration_cfg["target_accuracy_list"]
    pipeline_mode = iteration_cfg["pipeline_mode"]
    invoking_mode = iteration_cfg["invoking_mode"]
    model_list = iteration_cfg["model_list"]
    iteration_time = iteration_cfg["iteration_time"]
    tracking_parallel = False
    if pipeline_mode == "tracking":
        tracking_parallel = True
    for seg_path in seg_paths:
        print(seg_path)
        seg_name = os.path.basename(seg_path)
        pipeline = main_pipeline(f"{seg_path}/result/profile.csv", scheduler_class, tracking_parallel, invoking_mode)
        f_out = open(f"{seg_path}/result/{iteration_cfg['scheduler']}_iteration_result.csv", 'w', 1)
        writer = csv.writer(f_out)
        writer.writerow(["video_name", 'averaged_tracking_time', 'f1', 'precision', 'recall', 'target_accuracy', 'cost', 'triggered_times', 'pipeline_time'])
        print(seg_path)
        pipeline.scheduler.load_video_profile(seg_path)
        video = dataset_class(seg_path, seg_name, model_list, runtime_cfg['ground_truth_model'], filter_flag=False)
        for target_accuracy in target_accuracy_list:
            for _ in range(iteration_time):
                start_frame = 1
                end_frame = video.end_frame_index
                if DEBUG:
                    start_frame = 1601
                print('Evaluate {} start={} end={}'.format(
                    seg_name, start_frame, end_frame))
                start_t = time.perf_counter()
                frames_log, dt_glimpse, averaged_tracking_time, triggered, f1, precision, recall, total_cost, cost, interval_accuracy, interval_log, interval_frame_rate, model_list = pipeline.pipeline(seg_name, video, start_frame, end_frame, target_accuracy)
                interval_f1 = [item['f1'] for item in interval_accuracy]
                interval_recall = [item['recall'] for item in interval_accuracy]
                end_t = time.perf_counter()
                pipelien_time = end_t - start_t
                writer.writerow([seg_name, averaged_tracking_time, f1, precision, recall, target_accuracy, total_cost, len(triggered), pipelien_time])
                print(cost)
                print(interval_f1)
                print(interval_recall)
                print(interval_frame_rate)
                print(f"pipeline uses {end_t - start_t}")
                if DEBUG:
                    print("{} pipeline ended".format(seg_name))
                    print("target accuracy is {}".format(target_accuracy))
                    print("average tracking time is {}".format(averaged_tracking_time))
                    print("average f1 score is {}".format(f1))
                    print("average recall rate is {}".format(recall))
                    print("average precision is {}".format(precision))
                    print("overall_cost is {}".format(total_cost))
                    print("triggered {}".format(len(triggered)))

def profile(dataset_class, runtime_cfg):
    profile_cfg = runtime_cfg["profile_cfg"]
    pipeline_mode = profile_cfg["pipeline_mode"]
    invoking_mode = profile_cfg["invoking_mode"]
    model_list = profile_cfg["model_list"]
    tracking_parallel = False
    if pipeline_mode == "tracking":
        tracking_parallel = True
    profile_cfg = runtime_cfg["profile_cfg"]
    profile_paths = profile_cfg["profile_paths"]
    frame_slot = profile_cfg["frame_slot"]
    profile_model_frame_interval = profile_cfg["profile_model_frame_interval"]
    for path in profile_paths:
        print(path)
        seg_name = os.path.basename(path)
        pipeline = main_pipeline(f"{path}/result/profile.csv", scheduler.BaselineScheduler(), tracking_parallel, invoking_mode)
        f = open(f"{path}/profile/profile.csv", 'w')
        writer = csv.writer(f)
        writer.writerow(["profile_slot", "model", "accuracy", "recall", "frame_interval"])
        video = dataset_class(path, seg_name, model_list, runtime_cfg['ground_truth_model'], filter_flag=False)
        result = {}
        result['frame_slot'] = frame_slot
        profile_slot_count = video.frame_count // frame_slot + 1
        result["model_accuracy_to_frame_rate"] = {}
        for i in range(profile_slot_count):
            print(f"{i} of {profile_slot_count}")
            result["model_accuracy_to_frame_rate"][i] = {}
            for model in profile_model_frame_interval:
                result["model_accuracy_to_frame_rate"][i][model] = {"accuracy": [], "frame_rate": [], "recall": []}
                for frame_interval in profile_model_frame_interval[model]:
                    pipeline.scheduler.change_internal_state(model, frame_interval)
                    clip = seg_name + '_' + str(i)
                    start_frame = i * frame_slot + video.start_frame_index
                    end_frame = min((i + 1) * frame_slot, video.end_frame_index)
                    debug_print('{} start={} end={}'.format(clip, start_frame, end_frame))
                    debug_print(f"iteration: {frame_interval} {model}")
                    debug_print(f"scheduler: {pipeline.scheduler.get_next_frame_interval()} {pipeline.scheduler.get_next_frame_interval_model_name()}")
                    f1, precision, recall = pipeline.profile(video, start_frame, end_frame, model, frame_interval, frame_slot)
                    debug_print(f"{f1} {frame_interval} {recall}")
                    result["model_accuracy_to_frame_rate"][i][model]["accuracy"].append(f1)
                    result["model_accuracy_to_frame_rate"][i][model]["frame_rate"].append(frame_interval)
                    result["model_accuracy_to_frame_rate"][i][model]["recall"].append(recall)
                    writer.writerow([i, model, f1, recall, frame_interval])
                    if (f1 < 0.7):
                        break
        with open(f"{path}/profile/profile.json", 'w') as f:
            json.dump(result, f)

def run(args):
    dataset_class = get_dataset_class(args.dataset)
    seg_paths = get_seg_paths(args.data_root, args.dataset, args.video)
    # video is under data_root, if video is not None, data_root will join video
    config = args.config
    if config is None:
        config = './configs/runtime_cfg.json'
    runtime_cfg = None
    with open(config) as cfg:
        runtime_cfg = json.load(cfg)
    assert runtime_cfg != None

    mode = runtime_cfg["mode"]

    if mode == "pipeline":
        pipeline(dataset_class, seg_paths, runtime_cfg)
    if mode == "iteration":
        iteration(dataset_class, seg_paths, runtime_cfg)
    if mode == "profile":
        profile(dataset_class, runtime_cfg)
    
