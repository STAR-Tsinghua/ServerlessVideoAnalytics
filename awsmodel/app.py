import json
from detect_simple import image_to_box, test, weight_path, model_name
import time
'''
body: {
    "file_hash": "md5",
    "file_b64": "b64",
    "frame_num": number,
    "request_type": ""
}
'''
def handler(event, context):
    try:
        body = event["body"]
        body = json.loads(body)
        if body.get("request_type", None) is not None and body["request_type"] == "test":
            start = time.perf_counter()
            det = test()
            end = time.perf_counter()
            file_hash = "1234"
            frame_num = 0
        else:
            file_hash = body['file_hash']
            file_b64 = body['file_b64']
            frame_num = body['frame_num']
            start = time.perf_counter()
            det = image_to_box(file_b64, weight_path)
            end = time.perf_counter()
        time_elipesd = end - start
    except AssertionError:
        return {
            'statusCode': 500,
            'message': "assertion error"
        }
    except json.JSONDecodeError:
        return {
            'statusCode': 500,
            'message': "json decode error"
        }
    except Exception:
        return {
            'statusCode': 500,
            'message': "execuate logic error"
        }

    # TODO check file hash
    return {
        'statusCode': 200,
        'body': {
            'frame_num': frame_num,
            'file_hash': file_hash,
            'box_string': det,
            'model_time': time_elipesd,
            'model_name': model_name()
        }
        
    }