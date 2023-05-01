# awsmodel

This folder contains what you need to deploy YOLOv5 model as AWS Lambda function on AWS.

This README will instruct you to set up functions on AWS Lambda.

This code modifies YOLOv5 code from [ultralytics/yolov5][] and encapsulates the code to be invoked as a AWS Lambda function.

[ultralytics/yolov5]: https://github.com/ultralytics/yolov5

YOLOv5 have several versions, ranging from small to large, namely yolov5n, yolov5s, yolov5m, yolov5l, yolov5x.

> install dependencies using `requirements.txt` and install Docker.

## deploy models as function on AWS Lambda

In our implementation, each version of models is deployed as a separated function.

To deploy a YOLOv5 model, for example yolov5n, you need to do the following step 1~6:

1. download the corresponding model weight

> the weight file can be found in [ultralytics/yolov5][] under pretrained checkpoints tag.

[ultralytics/yolov5]: https://github.com/ultralytics/yolov5

2. put the weight file into `awsmodel/weights` folder.

> for example, the weight file is yolov5n.pt in this case. Put it into `awsmodel\weights` folder. **make sure the folder only contains the weight file of the model you want to deploy this time.**

3. change the `weight_path` variable in `awsmodel\detect_simple.py`.

> the variable is like `weights/yolov5n.py`, if you want to deploy yolov5n.

4. build docker image with `docker build -t awsmodel .`.

5. upload the image to the Amazon ECR follow the instruction from [Amazon][].

[Amazon]: https://docs.aws.amazon.com/lambda/latest/dg/images-create.html

> Remember to tag each Docker image uniquely.

6. create AWS Lambda function follow [instructions][] from AWS.

[instructions]: https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-images.html

now you have deployed a model, repeat step 1~6 for every yolov5 model you want to deploy.

## config url access to functions

After you deploys functions, you can access function in AWS console.

To access functions from outside, you should config Lambda function URLs using [instructions][] from AWS.

[instructions]: https://docs.aws.amazon.com/lambda/latest/dg/urls-configuration.html

After configuration, for each function, you have a URL.

To test the function, use `awsmodel/posttest.sh`

> replace `function URL here` in `awsmodel/posttest.sh` with your URL and run the script in shell.

## some tests

You can test the handler locally using `awsmodel/test_handler.py`

What is handler?
check [handler][]

[handler]: https://docs.aws.amazon.com/lambda/latest/dg/python-handler.html

If you want to check if the model inference properly, change the `weight_path` variable in `awsmodel\detect_simple.py` then run the file by `python detect_simple.py`.

> the variable is like `weights/yolov5n.py`, if you want to deploy yolov5n.

> for example, the weight file is yolov5n.pt in this case. Put it into `awsmodel\weights` folder. **make sure the folder only contains the weight file of the model you want to deploy this time.**
