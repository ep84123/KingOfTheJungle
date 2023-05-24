from operator import itemgetter

import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *


def get_imgs():
    video = open_video("../../data/Untitled video - Made with Clipchamp.mp4", 0)
    ret, img1 = video.read()
    for i in range(5):
        ret, img2 = video.read()
    return img1, img2


def track_points():
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    for i in range(10):
        img1, img2 = get_imgs()

        # find the keypoints and descriptors with SIFT
        kp1, desc1 = sift.detectAndCompute(img1, mask=None)
        kp2, desc2 = sift.detectAndCompute(img2, mask=None)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1, desc2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.25 * n.distance:
                good_matches.append(m)

        print('len(good_matches) =', len(good_matches))
        get_elem = itemgetter(*filter_moving_points(100, good_matches, kp1, kp2))
        result = get_elem(good_matches)

        # cv2.drawMatchesKnn expects list of lists as matches.
        # img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, (255, 0, 0), (0, 100, 255))
        img = cv2.drawMatches(img1, kp1, img2, kp2, result, None, (255, 0, 0), (0, 100, 255))

        # img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, (255, 0, 0), (0, 100, 255))

        plt.figure(figsize=(20, 10))
        plt.imshow(img)

        plt.show()


def filter_moving_points(threshold, good_matches, kp1, kp2):
    match_idx = [(m.queryIdx, m.trainIdx, idx) for idx, m in enumerate(good_matches)]
    match_by_dist = []
    for idx in match_idx:
        match_by_dist.append((np.linalg.norm(np.array(kp1[idx[0]].pt) - np.array(kp2[idx[1]].pt)), idx[2]))
    match_by_dist = sorted(match_by_dist, reverse=True, key=lambda x: x[0])
    print(match_by_dist)
    most_moving = [m[1] for m in match_by_dist[:threshold]]
    return most_moving


if __name__ == '__main__':
    track_points()
