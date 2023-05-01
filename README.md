# serverless on aws

This repository open-sources the code for paper "Edge-assisted Adaptive Configuration for Serverless-based Video Analytics" in ICDCD'23.

This repository contains the codes for edge server and AWS Lambda.

The AWS Lambda part is contains under folder `awsmodel`.
Detailed instruction on how to set up models on AWS is documented by README under `awsmodel`.

The remaining parts of the code is for edge server.

> demo picture and demo video are stored under test folder

## setup

## install dependencies

Install `conda` and `ffmpeg`.

Build a conda virtual environment `conda create -n <yourenvname> python=3.8`.

Activate conda environment you just created by `conda activate <yourenvname>`.

Install python dependencies with `requirements.txt`

## set up interaction with AWS

The `requesthandler.py` is responsible for send pictures to AWS Lambda and get results back.

You should set up AWS Lambda functions and get function URLs following README in awsmodel.

After you get function URLs of 5 models, put them into `./config/model_server.json`.

How to test your interaction?

Use `python requesthandler.py`.

It will send `1.picture` in `./test` to AWS Lambda and get bounding box back.

The first invocation have a non-ignorable cold-start overhead, just wait for a while.

## set up profiling models

Download the corresponding model weight of yolov5 models.

We use yolov5n, yolov5s, yolov5m, yolov5l, yolov5x.

So, you should donwload them all.

Put the weight file into `./model/weights` folder.

> the weight file can be found in [ultralytics/yolov5][] under pretrained checkpoints tag.
> the names of weight files should be `{model}.pt` like `yolov5x.pt`.
> if not, modify the settings `yolo_models` in `./config/runtime_cfg.json`.

[ultralytics/yolov5]: https://github.com/ultralytics/yolov5

## set up data folder

Video should be processed to be inputted into our system.

Our code use a data folder like `./demo`.

The directory should be organized like `./demo`.

```text
demo
|   demo.mp4
|
----profile
|   | yolov5l_1920x1080_23_detections.csv
|   | yolov5l_1920x1080_23_smoothed_detections.csv
|   | yolov5l_1920x1080_23_profile.csv
|   | yolov5m_1920x1080_23_detections.csv
|   | yolov5m_1920x1080_23_smoothed_detections.csv
|   | yolov5m_1920x1080_23_profile.csv
|   | ...
|
|---1080p
|   | 1920x1080 frame images ...
----result
|   | some result csv files
```

### create folder

If you want to use other videos as input, you should make a similar folder structure.

First, create a folder `your_folder_name` under this folder and move target `your_video.mp4` to the folder.
Then, create `./your_folder_name` folder and `./your_folder_name/profile`, `./your_folder_name/1080p` and `./your_folder_name/result` folder.

### extract frames

edit input_video option in `extract_frame.sh` to the absolute path of `./your_folder_name/video.mp4` and output_image_path to `./your_folder_name/1080p`.
Then, run the script with `bash extract_frame.sh`, extracted frames will be saved in `./your_folder_name/1080p`

### object detection

Change directory to `./model` and edit `yolov5.sh`.
Change input_path option to absolute path of `./your_folder_name/1080p`
Change output_path to absolute path of `./your_folder_name/profile`
Finally, run the script with `bash yolov5.sh` and profile csvs will be generated under `./your_folder_name/profile`.

### profiling

change the profile_paths option in `./configs/runtime_cfg.json`.
Add the absolute path of `your_folder_name` into the list.
Change the mode option into "profile".
Change directory back to project root and edit `main.sh`.
Change video option to the video folder name we created, in our example, it's `your_folder_name`
Then change data_root option to absolute path of parent folder of `your_folder_name`
Run by `bash main.sh`.
The profile will be generated under `./your_folder_name/profile` named `profile.csv`.
Copy the accuracy list under each model in `profile.csv` and change the model's accuracy under `./config/adaptive_scheduler.json`

## run the pipeline

Change the mode option in `./configs/runtime_cfg.json` into "pipeline".
Run the pipeline with `bash main.sh`
The result will be generated under `./your_folder_name/results/`.

## change settings

### change algorithm

At present, there are two algorithms for deciding video configurations.
One is baseline algorithm which sends frames at constant frame rate.
The other is adaptive algorithm.

To switch between them, change json key `scheduler` in `configs/runtime_cfg.json` between `baseline` and `adaptive` and rerun program

### configs

#### runtime_cfg

This file is located under `./configs`

Our program have three modes.

The first one is `pipeline`.
The second is `iteration`.
The third is `profile`.

To change modes, edit `runtime_cfg.json`'s key mode.
The default mode is pipeline.
Pipeline mode will try to minimize cost under target accuracy.
Under pipeline mode, To set target accuracy, change the target_accuracy key.

The iteration mode will iterates the target_accuracy_list and runs the pipeline for each target accuracy.

The profiling mode makes profile for several videos easier.
You can put video folders into profile_paths and mode will generates profiles for each folder.

We list the meaning of settings as bellow

- ground_truth_model
  take a model name from model_list, results outputted from this model will be used as ground truth in pipeline evaluation.
- pipeline
  take `notracking` or `tracking` as value, decide wether starting parallel tracks after getting object detections from server.
- invoking_mode
  take `local` or `aws` as value, decide from where the program get object detection results.
- model_list
  decide the models to use.
- yolo_models
  associates yolo_models with weights file.
- scheduler
  take `adaptive` or `baseline` as value, decide which scheduler to use.
- iteration_time
  determine how many times pipeline will run under a target accuracy.

#### baseline_scheduler

`./configs/baseline_scheduler.json` stores the settings of baseline scheduler.

#### adaptive_scheduler

`./configs/adaptive_scheduler.json` stores the settings of adaptive scheduler.

After running profile under profile mode, we can get `profile.csv` under `./your_folder_name/profile`.
Put the results into this file.
The accuracy list and frame_rate is corresponding by index.
