from ast import arg
import cv2
import numpy as np
from models.our_detector import Our
import argparse

def parseArg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input image"
    , required=True) 
    parser.add_argument("--model_path", help="path to model file"
    , required=True) 

    args = parser.parse_args()
    return args

# main
def main():
    global args
    args = parseArg()

    img = cv2.imread(args.input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., np.newaxis]

    detector = Our()
    detector.init_detector(args.model_path)
    score_map, keypoints, descs = detector.detect(img, 1024, 0.2)

    np.save('s_map.npy', score_map)
    np.save('kps.npy', keypoints)
    np.save('descs.npy', descs)

    print("Done!")

main()