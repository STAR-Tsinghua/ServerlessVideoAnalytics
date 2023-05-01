"""Object detection module main function."""
from parser import parse_args
from infer import infer
import os
import json

if __name__ == '__main__':
    args = parse_args()
    print(os.path.abspath(__file__))
    print(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.dirname(os.path.abspath(__file__)) + '/../configs/runtime_cfg.json') as cfg:
        runtime_cfg = json.load(cfg)
    model_list = runtime_cfg["yolo_models"]
    for item in model_list:
        print("processing {}".format(item["model_name"]))
        infer(args.input_path, args.output_path, item["model_name"], item["weight_path"], args.width, args.height, args.qp, args.crop)
