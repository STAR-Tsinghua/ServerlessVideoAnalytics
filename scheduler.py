import random
import os
import numpy
import json
import csv

DEBUG=False
def debug_print(*args):
    if DEBUG:
        print(*args)
class Scheduler():
    def __init__(self):
        self.frame_interval = 10
        self.frame_slot = 10
        self.previous_slot_info = {}
        self.model = ""
        self.interval_info = {
            "frame_slot": self.frame_slot,
            "frame_interval": self.frame_interval,
            "model": "yolov5n",
            "interval_cost": 0
        }
    
    def algorithm(self, *args, **kwargs):
        pass
    
    def renew(self, *args, **kwargs):
        pass

    def get_next_frame_interval(self):
        pass
    def load_video_profile(self, path):
        pass

    def get_next_frame_interval_model_name(self):
        pass

    def get_time_interval_info(self, accuracy=0.5):
        self.algorithm(accuracy)
        return self.interval_info

class BaselineScheduler(Scheduler):
    def __init__(self):
        super().__init__()
        with open("./configs/baseline_scheduler.json") as cfg:
            scheduler_cfg = json.load(cfg)
            self.frame_interval = scheduler_cfg["frame_interval"]
            self.frame_slot = scheduler_cfg["frame_slot"]
            self.model = scheduler_cfg["model"]
            self.model_cost = scheduler_cfg["model_cost"]
            self.interval_info["frame_slot"] = self.frame_slot
            self.interval_info["frame_interval"] = self.frame_interval
            self.interval_info["model"] = self.model
            self.interval_info["interval_cost"] = self.model_cost[self.model]
    # TODO add profile to baseline

    def get_time_interval_info(self, *args):
        return self.interval_info

    def change_internal_state(self, model, frame_interval):
        self.model = model
        self.frame_interval = frame_interval
        self.interval_info["frame_slot"] = self.frame_slot
        self.interval_info["frame_interval"] = self.frame_interval
        self.interval_info["model"] = self.model
        self.interval_info["interval_cost"] = self.model_cost[self.model]

class AdaptiveScheduler(Scheduler):
    def __init__(self, debug_func=debug_print):
        super().__init__()
        with open("./configs/adaptive_scheduler.json") as cfg:
            scheduler_cfg = json.load(cfg)
            self.frame_slot = scheduler_cfg["frame_slot"]
            self.max_iteration = scheduler_cfg.get("max_iteration", 500)
            self.change_threshold = scheduler_cfg.get("change_threshold", 100)
            self.max_no_change_iteration = scheduler_cfg.get("max_no_change_iteration", 20)
            self.r = scheduler_cfg.get("r", 10)
            self.model_list = scheduler_cfg["model_list"]
            self.interval_info["frame_slot"] = self.frame_slot
            self.model_memory = scheduler_cfg["model_memory"]
            self.model_cost = scheduler_cfg["model_cost"]
        self.debug_func = debug_func
        self.get_frame_rate = self._get_frame_rate
        self.frame_rate_store = {}

    # @debug_func.setter
    # def debug_func(self, func):
    #     self.debug_func = func
    # @property
    # def r(self):
    #     return self.r
    # @r.setter
    # def r(self, new_r):
    #     self.r = new_r
    def load_video_profile(self, path):
        with open(f"{path}/profile/profile.json", 'r') as cfg:
            profile = json.load(cfg)
            memory_profile = profile.get("model_memory", None)
            model_cost = profile.get("model_cost", None)
            if memory_profile is not None:
                self.model_memory = memory_profile
            if model_cost is not None:
                self.model_cost = model_cost
            self.model_accuracy_to_frame_rate = profile["model_accuracy_to_frame_rate"]
        individual_profile = f"{path}/profile/scheduler.json"
        if os.path.exists(individual_profile):
            with open(f"{path}/profile/scheduler.json") as s_cfg:
                scheduler_cfg = json.load(s_cfg)
                self.frame_slot = scheduler_cfg["frame_slot"]
                self.max_iteration = scheduler_cfg.get("max_iteration", 500)
                self.change_threshold = scheduler_cfg.get("change_threshold", 100)
                self.max_no_change_iteration = scheduler_cfg.get("max_no_change_iteration", 20)
                self.r = scheduler_cfg.get("r", 10)
                self.model_list = scheduler_cfg["model_list"]
                self.interval_info["frame_slot"] = self.frame_slot
        else:
            tmp = {}
            tmp["frame_slot"] = self.frame_slot
            tmp["max_iteration"] = self.max_iteration
            tmp["change_threshold"] = self.change_threshold
            tmp["max_no_change_iteration"] = self.max_no_change_iteration
            tmp["model_list"] = self.model_list
            tmp["r"] = self.r
            tmp["model_list"] = self.model_list
            tmp["model_memory"] = self.model_memory
            tmp["model_cost"] = self.model_cost
            with open(f"{path}/profile/scheduler.json", 'w') as s_cfg:
                json.dump(tmp, s_cfg)

            
        self.current_frame_slot = None
        self.current_frame_slot_profile = None



    def _change_state(self, accuracy, feasible_model_list, model=""):
        if model == "":
            current_model = self._get_model(feasible_model_list)
        else:
            current_model = model
        current_memory = self._get_memory(current_model)
        current_frame_rate = self.get_frame_rate(current_model, current_memory, accuracy)
        current_cost = self._get_cost(current_model, current_memory, current_frame_rate)
        return current_model, current_memory, current_frame_rate, current_cost


    def algorithm(self, accuracy=0.8, debug_print=debug_print, *args, **kwargs):
        feasible_model_list = self._get_feasible_models(accuracy)
        i = 0
        no_change_iteration_count = 0
        current_model, current_memory, current_frame_rate, current_cost = self._change_state(accuracy, feasible_model_list)
        debug_print(current_model, current_memory, current_frame_rate, current_cost)
        single_cost = self.model_cost[current_model]
        model = self._get_model(feasible_model_list)
        while(i < self.max_iteration and no_change_iteration_count < self.max_no_change_iteration):
            model, memory, frame_rate, cost = self._change_state(accuracy, feasible_model_list, model)
            possibility = 1 / (1 + numpy.exp((cost - current_cost) / self.r))
            if random.random() < possibility:
                current_model = model
                current_memory = memory
                current_frame_rate = frame_rate
                current_cost = cost
                single_cost = self.model_cost[model]
            else:
                model = self._get_model(feasible_model_list)
            debug_print(current_model, current_memory, current_frame_rate, current_cost, possibility)
            if (numpy.abs(current_cost - cost) < self.change_threshold):
                no_change_iteration_count += 1
            else:
                no_change_iteration_count = 0
            i += 1
        return current_model, current_memory, current_frame_rate, single_cost

    def _get_model(self, model_list) -> str:
        seed = random.random()
        return model_list[int(seed * len(model_list))]

    def _get_memory(self, model:str):
        memory = self.model_memory.get(model, None)
        assert memory != None
        return memory
    
    def _get_feasible_models(self, accuracy):
        assert self.current_frame_slot_profile != None
        ret = []
        for model in self.model_list:
            accuracy_list = self.current_frame_slot_profile[model]["accuracy"]
            if accuracy > max(accuracy_list):
                continue
            ret.append(model)
        return ret

        

    def _get_frame_rate(self, model, memory, accuracy):
        assert self.current_frame_slot_profile != None
        accuracy_list = self.current_frame_slot_profile[model]["accuracy"]
        frame_rate_list = self.current_frame_slot_profile[model]["frame_rate"]
        assert len(accuracy_list) == len(frame_rate_list)
        # because of _get_feasible_model, there must exists one value greater than accuracy
        index = -1
        current_diff = 1
        for i in range(len(accuracy_list)):
            diff = accuracy_list[i] - accuracy
            if diff >= 0 and diff < current_diff:
                index = i
                current_diff = diff
        return frame_rate_list[index]
    
    def _get_frame_rate_test(self, model, memory, accuracy):
        assert self.current_frame_slot_profile != None
        accuracy_list = self.current_frame_slot_profile[model]["accuracy"]
        frame_rate_list = self.current_frame_slot_profile[model]["frame_rate"]
        assert len(accuracy_list) == len(frame_rate_list)
        if self.frame_rate_store.get(model, None) == None:
            self.frame_rate_store[model] = []
            # because of _get_feasible_model, there must exists one value greater than accuracy
            index = -1
            for i in range(len(accuracy_list)):
                diff = accuracy_list[i] - accuracy
                if diff >= 0:
                    self.frame_rate_store[model].append(i)
        model_len = len(self.frame_rate_store[model])
        seed = random.random()
        return frame_rate_list[int(seed * model_len)]
        
        return frame_rate_list[index]
        

    def _get_cost(self, model, memory, frame_rate):
        times = self.frame_slot // frame_rate
        if not self.frame_slot % frame_rate:
            times += 1
        return self.model_cost[model] * times

    def get_time_interval_info(self, accuracy=0.5, frame_idx=1):
        frame_slot = (frame_idx - 1) // self.frame_slot
        if frame_slot != self.current_frame_slot:
            self.current_frame_slot_profile = self.model_accuracy_to_frame_rate[str(frame_slot)]
            current_model, _, current_frame_rate, current_cost = self.algorithm(accuracy, self.debug_func)
            self.interval_info["frame_interval"] = current_frame_rate
            self.interval_info["model"] = current_model
            self.interval_info["interval_cost"] = current_cost
        return self.interval_info

def main():
    test_scheduler = AdaptiveScheduler()
    test_scheduler.load_video_profile("/your/path/to/this/repository/videoanalysis/demo2")
    path = "./algorithm_result"
    for item in [ 60, 80, 100, 150, 300]:
        with open(f"{path}/algorithm_with_{item}.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['model', 'memory', 'frame_rate', 'single_cost', 'possibility_to_change'])
            def f(*args):
                writer.writerow([*args])
            test_scheduler.debug_func = f
            test_scheduler.get_frame_rate = test_scheduler._get_frame_rate_test
            test_scheduler.r = item
            test_scheduler.get_time_interval_info(0.85, 1)
        
if __name__ == "__main__":
    main()