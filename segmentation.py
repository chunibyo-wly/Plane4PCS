from random import random

import numpy as np


def angle(i, j, plane):
    v1, v2 = np.array(plane[i][:-1]), np.array(plane[j][:-1])
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) / np.pi * 180


def plane_segmented(pcd, threshold=0.5, planes=10):
    outlier_cloud = pcd
    inlier_list, plane_list = [], []

    for i in range(planes):
        plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=threshold,
                                                           ransac_n=3,
                                                           num_iterations=1000)
        [a, b, c, d] = plane_model
        # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        inlier_cloud = outlier_cloud.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([random(), random(), random()])
        inlier_list.append(inlier_cloud)
        plane_list.append(plane_model)

        outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)
    return inlier_list, plane_list
