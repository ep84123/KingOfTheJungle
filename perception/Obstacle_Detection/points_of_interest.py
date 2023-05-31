from operator import itemgetter
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *


def get_imgs(video):
    ret, img1 = video.read()
    for i in range(5):
        ret, img2 = video.read()
    return img1, img2


def track_points():
    t0 = time.time()
    video = open_video("../../data/Untitled video - Made with Clipchamp.mp4", 0)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    print("1, time pass: ", time.time() - t0)
    for i in range(1):


        # # Find the keypoints and descriptors with SIFT.
        # kp1, descr1 = sift.detectAndCompute(img1, None)
        # kp2, descr2 = sift.detectAndCompute(img2, None)
        # print("1, time pass: ", time.time() - t0)
        #
        # # FLANN parameters.
        # FLANN_INDEX_KDTREE = 0
        # index_params = dict(algorithm=0, trees=5)
        # search_params = dict(checks=50)  # or pass empty dictionary
        # print("before match, time pass: ", time.time() - t0)
        #
        # # FLANN based matcher with implementation of k nearest neighbour.
        # flann = cv2.FlannBasedMatcher(index_params, search_params)
        # matches = flann.knnMatch(descr1, descr2, k=2)
        # print("after match, time pass: ", time.time() - t0)
        #
        # # selecting only good matches.
        # matchesMask = [[0, 0] for i in range(len(matches))]
        #
        # # ratio test.
        # for i, (m, n) in enumerate(matches):
        #     if (m.distance < 0.1 * n.distance):
        #         matchesMask.append([1, 0])

        # find the key points and descriptors with SIFT

        img1, img2 = get_imgs(video)
        print("1, time pass: ", time.time() - t0)

        # Initiating the SIFT detector
        sift = cv2.SIFT_create()
        print("1, time pass: ", time.time() - t0)

        kp1, desc1 = sift.detectAndCompute(img1, mask=None)
        print("2, time pass: ", time.time() - t0)
        kp2, desc2 = sift.detectAndCompute(img2, mask=None)
        print("3, time pass: ", time.time() - t0)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        print("4, time pass: ", time.time() - t0)

        matches = bf.knnMatch(desc1, desc2, k=2)
        print("5, time pass: ", time.time() - t0)



        good_matches = []
        # Apply ratio test
        for m, n in matches:
            if m.distance < 0.2 * n.distance:
                good_matches.append(m)

        print("6, time pass: ", time.time() - t0)

        print('len(good_matches) =', len(good_matches))
        get_elem = itemgetter(*filter_moving_points(100, good_matches, kp1, kp2))
        print("7, time pass: ", time.time() - t0)

        result = get_elem(good_matches)
        print("8, time pass: ", time.time() - t0)

        # drawing nearest neighbours
        # draw_params = dict(matchColor=(0, 255, 0),
        #                    singlePointColor=(255, 0, 0),
        #                    matchesMask=matchesMask, flags=0)
        # img = cv2.drawMatchesKnn(img1,
        #                          kp1,
        #                          img2,
        #                          kp2,
        #                          matchesMask,
        #                          None,
        #                          **draw_params)
        # return img
        # cv2.drawMatchesKnn expects list of lists as matches.
        #img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, (255, 0, 0), (0, 100, 255))
        img = cv2.drawMatches(img1, kp1, img2, kp2, result, None, (255, 0, 0), (0, 100, 255))
        #
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
    # print(match_by_dist)
    most_moving = [m[1] for m in match_by_dist[:threshold]]
    return most_moving


if __name__ == '__main__':
    track_points()
