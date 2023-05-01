import json
import base64
import requests
import hashlib
import time


class RequestHandler():
    model_dict = {}
    result_callback = None
    def __init__(self, result_callback=None) -> None:
        with open('./configs/model_server.json') as cfg:
            self.model_dict = json.load(cfg)
        if result_callback is not None:
            self.result_callback = None

    def registerCallback(self, callback_func):
        if not callable(callback_func):
            raise TypeError
        self.result_callback = callback_func

    def sendRequest(self, model_name: str, image: str, frame_num=0, **kwargs) -> dict:
        model_config = self.model_dict.get(model_name, None)
        if model_config is None:
            raise Exception
        url = model_config["url"]
        file_md5 = hashlib.md5(image).hexdigest()
        file_b64 = base64.b64encode(image).decode()
        request_data = {
            "file_hash": file_md5,
            "file_b64": file_b64,
            "frame_num": frame_num,
            "request_type": ""
        }
        start = time.perf_counter()
        response_dict = requests.post(url, json=request_data).json()
        end = time.perf_counter()
        print(end - start)
        dets = None
        if model_name == "yolo_model":
            box_string = response_dict['box_string']
            box_string_dict = json.loads(box_string)
            dets = []
            for i in range(box_string_dict['len']):
                tmp = []
                tmp.extend(box_string_dict['boxes'][i])
                tmp.append(box_string_dict['classes'][i])
                tmp.append(box_string_dict['scores'][i])
                tmp.append(i + 1)
                dets.append(tmp)
        else:
            dets = response_dict['box_string']
            
        result = {
            "file_hash": response_dict['file_hash'],
            "frame_num": response_dict['frame_num'],
            "model_time": response_dict['model_time'],
            "dets": dets
        }
        if self.result_callback:
            self.result_callback(result)
        return result
    def warmUp(self, model_name = ""):
        request_list = []
        if (model_name != ""):
            model_config = self.model_dict.get(model_name, None)
            request_list.append(model_config["url"])
        else:
            for item in self.model_dict:
                request_list.append(self.model_dict[item]['url'])
        request_data = {
            "request_type": "test"
        }
        print("warming up lambda functions")

        for url in request_list:
            try:
                response_dict = requests.post(url, json=request_data).json()
            except:
                print("model time out")
                continue
        


def main():
    '''
    test RequestHandler's utility
    '''
    result = None
    aws = RequestHandler()
    with open("./test/1.jpg", "rb") as test_image:
        image_str = test_image.read()
        for model in ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]:
            start = time.perf_counter()
            result = aws.sendRequest(model, image_str)
            end = time.perf_counter()
            print("send image to model {} spend {} second".format(model, end - start))
            print(result)

if __name__ == "__main__":
    main()
