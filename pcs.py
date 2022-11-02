import logging

LOGGING_FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT, filename='log.txt', filemode='w')

import copy
import os
import shutil
import subprocess
from os.path import join, exists

import meshio
import numpy as np
import open3d as o3d

from segmentation import plane_segmented

_tmp = r"/tmp"
_target = r"/mnt/c/Users/35980/Desktop/target2.ply"
_source = r"/mnt/c/Users/35980/Desktop/source2.ply"
s4pcs = r"/mnt/d/workspace/OpenGR/cmake-build-debug/apps/Super4PCS/Super4PCS"
_tmp_output = join(_tmp, 'out.ply')
_tmp_matrix = join(_tmp, 'output.txt')
tmp_source = join(_tmp, 'source.ply')
tmp_target = join(_tmp, 'target.ply')


def scale_estimation():
    # TODO: 计算高度
    pass


def lcp(tree, pcd, threshold):
    best_lcp = 0
    for point in pcd.points:
        [k, idx, distance] = tree.search_knn_vector_3d(point, 1)
        if k == 0:
            continue
        if distance[0] <= threshold:
            best_lcp += 1
    return best_lcp


def match_ratio(cloud1, cloud2, threshold):
    tree = o3d.geometry.KDTreeFlann(cloud1)
    _lcp = lcp(tree, cloud2, threshold)
    return max(_lcp / len(cloud1.points), _lcp / len(cloud2.points))


def match_plane(cloud1, cloud2, threshold, beta, mat=np.identity(4)) -> bool:
    cloud1_copy = copy.deepcopy(cloud1)
    return match_ratio(cloud1_copy.transform(mat), cloud2, threshold) >= beta


def match_planes(source_list, target_list, threshold, beta, mat=np.identity(4)):
    cnt = 0
    for source in source_list:
        for target in target_list:
            if match_plane(source, target, threshold, beta, mat):
                cnt += 1
    return cnt


def write_pcd(file_path, pcd):
    o3d.io.write_point_cloud(file_path, pcd)
    mesh = meshio.read(file_path)
    mesh.points = mesh.points.astype(np.float32)
    mesh.point_data['nx'] = mesh.point_data['nx'].astype(np.float32)
    mesh.point_data['ny'] = mesh.point_data['ny'].astype(np.float32)
    mesh.point_data['nz'] = mesh.point_data['nz'].astype(np.float32)
    mesh.point_data.pop('red', None)
    mesh.point_data.pop('green', None)
    mesh.point_data.pop('blue', None)
    meshio.write(file_path, mesh)


def show():
    source_pcd = o3d.io.read_point_cloud(_source)
    source_pcd.transform(np.loadtxt('out'))
    o3d.io.write_point_cloud('out.ply', source_pcd)


def main():
    shutil.rmtree('output', ignore_errors=False)
    os.mkdir('output')

    n = 10

    target_pcd = o3d.io.read_point_cloud(_target)
    source_pcd = o3d.io.read_point_cloud(_source)
    target_list, target_plane = plane_segmented(target_pcd, planes=n)
    source_list, source_plane = plane_segmented(source_pcd, planes=n)
    logging.info("* segmented done! *")

    cnt = 0
    mat = None
    for i in range(n - 1):
        for a in range(n - 1):
            print(i, a)
            write_pcd(tmp_target, target_list[i])
            write_pcd(tmp_source, source_list[a])

            subprocess.run(
                [s4pcs, "-i", tmp_target, tmp_source, "-r", _tmp_output, "-o", "0.8", "-d", "0.3", "-m",
                 _tmp_matrix, "-x"])

            tmp_mat = np.loadtxt(_tmp_matrix)
            tmp_cnt = match_planes(source_list, target_list, threshold=0.3, beta=0.3,
                                   mat=tmp_mat)
            if tmp_cnt > cnt:
                cnt = tmp_cnt
                mat = tmp_mat

            o3d.io.write_point_cloud(f'output/{i}_{a}_{tmp_cnt}.ply',
                                     copy.deepcopy(source_pcd).transform(tmp_mat))
            logging.info(tmp_cnt)

    if mat is None:
        logging.error("* ERROR !!! *")
    else:
        np.savetxt("out.txt", mat)
        o3d.io.write_point_cloud('out.ply', copy.deepcopy(source_pcd).transform(mat))


if __name__ == '__main__':
    main()
