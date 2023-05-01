from app import handler
import base64
import json

def main():
    with open("data/images/1.jpg", "rb") as imagestring:
        image_string = base64.b64encode(imagestring.read())
        mock_data = {
            "request_type": "test"
        }
        event = json.dumps(mock_data)
        print("test handler func")
        print(handler({"body": event}, None))
        # seperator
        mock_data = {
            "file_hash": "1234",
            "file_b64": image_string.decode(),
            "frame_num": 0,
        }
        event = json.dumps(mock_data)
        print("test handler main func")
        print(handler({"body": event}, None))
        
if __name__ == "__main__":
    main()