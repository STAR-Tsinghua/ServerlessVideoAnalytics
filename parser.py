import argparse


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="serverless video analytics")
    # data related
    parser.add_argument("--video", type=str, default=None, help="video name")
    parser.add_argument("--data_root", type=str, required=True,
                        help='root path to video mp4/frame data')
    parser.add_argument("--dataset", type=str, default='youtube',
                        choices=['kitti', 'mot15',
                                 'mot16', 'waymo', 'youtube'],
                        help="dataset name")
    parser.add_argument("--mode", type=str, default="pipeline",
                        help="pipeline, iteration or profile")
    parser.add_argument("--config", type=str, default=None,
                        help="pipeline, iteration or profile")
    args = parser.parse_args()
    return args
