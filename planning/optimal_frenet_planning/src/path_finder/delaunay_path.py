#! /usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import time
from scipy.spatial import KDTree

from sklearn.cluster import DBSCAN
import matplotlib.tri as mtri
from scipy import interpolate

import pyproj
def latlon_to_utm(lat, lon):
    proj = '+proj=utm +zone=52 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
    latlon_to_utm = pyproj.Proj(proj, preserve_units=True)
    return latlon_to_utm(lon, lat)

def is_point_in_rectangle(rect_points, point):
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    A, B, C, D = rect_points

    cp1 = cross_product(A, B, point)
    cp2 = cross_product(B, C, point)
    cp3 = cross_product(C, D, point)
    cp4 = cross_product(D, A, point)
    
    return (cp1 >= 0 and cp2 >= 0 and cp3 >= 0 and cp4 >= 0) or (cp1 <= 0 and cp2 <= 0 and cp3 <= 0 and cp4 <= 0)

class Cones:
    def __init__(self, xs, ys):
        self.x = xs
        self.y = ys

class DelaunayPath():
    def __init__(self, ref_path, car_pose):
        self.ref_path = ref_path
        self.labels = []
        self.cones = []
        self.prev_midpoints = None
        self.is_started = False
        self.is_finished = False
        self.start_time = None
        self.last_time = None

        path_coords = np.vstack((self.ref_path['x'], self.ref_path['y'])).T
        self.path_kdtree = KDTree(path_coords)

        self.finish_zone = [latlon_to_utm(37.240232, 126.775310),
                            latlon_to_utm(37.240241, 126.775359),
                            latlon_to_utm(37.241672, 126.775322),
                            latlon_to_utm(37.241678, 126.775291)]
        
        return
    
    def find_path(self, car_pose, clusters_msg):
        is_finish_zone_inside = is_point_in_rectangle(self.finish_zone, car_pose[:2])
        print("안에?", is_finish_zone_inside)

        if clusters_msg is None or is_finish_zone_inside:
            print("클러스터링 X 또는 finish zone 입성")
            path_xs, path_ys, path_yaws = self.make_normal_path(car_pose)
            return [], [path_xs, path_ys, path_yaws], False, 0, False
        
        is_cone_detected, cones_local, self.labels = self.classify_cones(clusters_msg)
        self.cones = self.cones_local_to_global(cones_local, car_pose)

        cones_viz = [Cones(self.cones[:,0], self.cones[:,1])]
        
        if is_cone_detected:
            if self.is_started is False:
                self.is_started = True
                self.start_time = time.time()

            print("라바콘 2개 이상 감지")
            # ref_path_dist, _ = self.path_kdtree.query([car_pose[0], car_pose[1]])
            # if ref_path_dist < 1.5 and time.time() - self.start_time > 7:
            #     path_xs, path_ys, path_yaws = self.make_normal_path(car_pose)

            delaunay_triangles = mtri.Triangulation(self.cones[:,0], self.cones[:,1])
            triangles = np.array(delaunay_triangles.get_masked_triangles(), dtype=np.uint8) # [[a,b,c], [d,e,f], ...]
            triangles_inlier = self.get_inlier_triangle(triangles)

            # 모두 같은 색 라바콘인 경우에는
            if len(triangles_inlier) == 0:
                midpoints = self.prev_midpoints
            else:
                midpoints = self.get_midpoints(triangles_inlier, car_pose)

            path_xs, path_ys, path_yaws = self.get_bspline_path(midpoints, car_pose)

            self.prev_midpoints = midpoints
            self.last_time = time.time()

            return cones_viz, [path_xs, path_ys, path_yaws], False, 0, False
        
        else:
            ref_path_dist, _ = self.path_kdtree.query([car_pose[0], car_pose[1]])
            if ref_path_dist < 1.5:
                path_xs, path_ys, path_yaws = self.make_normal_path(car_pose)
            
            elif self.last_time is not None and time.time() - self.last_time < 1:
                if self.prev_midpoints is not None:
                    path_xs, path_ys, path_yaws = self.get_bspline_path(self.prev_midpoints, car_pose)

                else:
                    path_xs, path_ys, path_yaws = self.make_normal_path(car_pose)
            else:
                path_xs, path_ys, path_yaws = self.make_normal_path(car_pose)
            return cones_viz, [path_xs, path_ys, path_yaws], False, 0, False
        
    def make_normal_path(self, car_pose):
        car_x, car_y = car_pose[0], car_pose[1]
        
        _, cur_idx = self.path_kdtree.query([car_x, car_y])
        n_back = 4
        n_forward = 10
        path_len = len(self.ref_path['x'])

        start_index = max(cur_idx - n_back, 0)
        end_index = min(cur_idx + n_forward, path_len)
        
        # 인근 waypoints 추출
        path_xs = self.ref_path['x'][start_index:end_index + 1]
        path_ys = self.ref_path['y'][start_index:end_index + 1]
        path_yaws = self.ref_path['yaw'][start_index:end_index + 1]
        return path_xs, path_ys, path_yaws
    
    def classify_cones(self, cluster_msg):
        labels = []
        cones = []
        is_cone_detected = False

        num = 0 
        for id, obs in enumerate(cluster_msg.markers):
            xmin = obs.points[1].x
            xmax = obs.points[0].x
            ymin = obs.points[3].y
            ymax = obs.points[4].y

            depth = abs(xmax - xmin)
            width = abs(ymax - ymin)

            if depth <= 0.4 and width <= 0.4:
                x = (xmin + xmax) / 2 + 1.0
                y = (ymin + ymax) / 2

                print(x, y)
                # Yellow : y > mx + b
                m, b = -0.35, 0.0
                if (0.0 < x < 8 and m * x + b <= y and y > -1.0): # 1_Yellow
                # if (-2 < x < 8 and m * x + b <= y < 2.0) or (-1 < x <= 0 and -0.6 <= y < 1.5): # 1_Yellow
                    left_cone = [x, y]
                    cones.append(left_cone)
                    labels.append(1)

                    right_cone = [x-1.0, y-4.0]
                    cones.append(right_cone)
                    labels.append(0)

                    num += 1

        if num >= 2:
            is_cone_detected = True
        else:
            is_cone_detected = False

        virtual_cones = [[-1, -2.0], [-1.5, -2.0], [-2, -2.0],
                         [-1, 2.0], [-1.5, 2.0], [-2, 2.0]]
        
        # virtual_cones = [[-1, -0.5], [-1.5, -0.5], [-2, -0.5],
        #                  [-1, 0.5], [-1.5, 0.5], [-2, 0.5]]
        
        virtual_labels = [0, 0, 0,
                          1, 1, 1]

        for virtual_cone, virtual_label in zip(virtual_cones, virtual_labels):
            cones.append(virtual_cone)
            labels.append(virtual_label)

        return is_cone_detected, cones, labels
        
    def cones_local_to_global(self, cones_local, car_pose):
        car_x, car_y, car_yaw = car_pose

        rotation_matrix = np.array([[np.cos(car_yaw), -np.sin(car_yaw)],
                                    [np.sin(car_yaw), np.cos(car_yaw)]])
        
        cones_local = np.array(cones_local)

        cones_global = (rotation_matrix @ cones_local.T).T + np.array([car_x, car_y])
        return cones_global

    def get_inlier_triangle(self, triangles):
        deltri_inlier = []
        for triangle in triangles:
            if not self._is_same_color(triangle):
                deltri_inlier.append(triangle)
                
        return deltri_inlier
    
    def get_midpoints(self, triangles, car_pose):
        graph = {tuple(sorted(triangle)): [] for triangle in triangles}
        for triangle in triangles:
            for other_triangle in triangles:
                if triangle is not other_triangle and len(set(triangle) & set(other_triangle)) == 2:
                    graph[tuple(sorted(triangle))].append(tuple(sorted(other_triangle)))
        
        # 시작 삼각형 찾기
        first_tri = None
        dist_min = np.inf

        print("car pose:", car_pose)
        offset_dist = -4.0
        target_point = np.array([car_pose[0] + offset_dist*np.cos(car_pose[2]),
                                  car_pose[1] + offset_dist*np.sin(car_pose[2])])
        print("target point:", target_point)

        for tri in graph.keys():
            if len(graph[tri]) == 1:
                dist = self._get_triangle_distance(tri, target_point)
                if dist < dist_min:
                    dist_min = dist
                    first_tri = tri        
        
        if first_tri is None:
            return self.prev_midpoints
        
        # graph 구조를 통해 삼각형 이어주기
        path_cones=[]
        path = [first_tri]
        outlier = []
        cur_node = first_tri
        for i in range(len(graph.keys())):
            neighbors = graph[cur_node]
            for neighbor in neighbors:
                if neighbor not in path and neighbor not in outlier:
                    common_indicies = list(set(cur_node) & set(neighbor))
                    try:
                        if self.labels[common_indicies[0]] != self.labels[common_indicies[1]]:
                            path.append(neighbor)
                            cur_node = neighbor
                            path_cones.append([common_indicies[0], common_indicies[1]])
                        else: 
                            outlier.append(neighbor)
                    
                    except:
                        print('common_indicies: ', common_indicies)
            #     else:
            #         cnt += 1
            # if cnt > 2:
            #     break
            
        # 첫 번째 삼각형에서 미추가된 라바콘 세트 추가
        common_indicies = list(set(first_tri) & set(path_cones[0]))
        noncommon_index = [e for e in first_tri if e not in common_indicies][0]
        if self.labels[common_indicies[0]] != self.labels[noncommon_index]:
            path_cones.insert(0, [common_indicies[0], noncommon_index])
        else:
            path_cones.insert(0, [common_indicies[1], noncommon_index])
        
        # 마지막 삼각형에서 미추가된 라바콘 세트 추가
        last_tri = path[-1]
        common_indicies = list(set(last_tri) & set(path[-2]))
        noncommon_index = [e for e in last_tri if e not in common_indicies][0]
        if self.labels[common_indicies[0]] != self.labels[noncommon_index]:
            path_cones.append([common_indicies[0], noncommon_index])
        else:
            path_cones.append([common_indicies[1], noncommon_index])
        # print(path_cones)
        
        # midpoint 계산
        midpoints = np.empty((0,2))
        for cone_1, cone_2 in path_cones:
            midpoint = (self.cones[cone_1] + self.cones[cone_2])/2
            midpoints = np.vstack((midpoints, [midpoint]))

        return midpoints
    
    @staticmethod
    def get_bspline_path(midpoints, car_pose):
        x=midpoints[:,0]
        y=midpoints[:,1]
        
        l=len(x)
        Order = 3

        t=np.linspace(0,1,l-(Order-1),endpoint=True)
        t=np.append(np.zeros(Order),t)
        t=np.append(t,np.zeros(Order)+1)
        
        tck=[t,[x,y],Order]
        u3=np.linspace(0,1,(max(l*2,35)),endpoint=True)
        out = np.array(interpolate.splev(u3,tck)) # xs: out[0] / ys: out[1]
        out = out.T
        
        # Extract car_pose's position
        car_x, car_y, _ = car_pose
        
        # Compute distances from car_pose to all points in the B-spline path
        distances = np.sqrt((out[:, 0] - car_x) ** 2 + (out[:, 1] - car_y) ** 2)
        
        # Find index of the closest point to car_pose
        closest_idx = np.argmin(distances)
        
        # Initialize lists for path x, y, yaw values
        path_x, path_y, path_yaw = [], [], []
        
        # Iterate from closest point onward to generate the path
        for i in range(closest_idx, len(out) - 1):
            x, y = out[i]
            x_n, y_n = out[i + 1]
            
            # Compute the slope and yaw (orientation)
            yaw = np.arctan2((y_n - y), (x_n - x))
            
            path_x.append(x)
            path_y.append(y)
            path_yaw.append(yaw)
    
        return path_x, path_y, path_yaw


    
    def _get_triangle_distance(self, triangle, target_point):
        a_idx, b_idx, c_idx = triangle[0], triangle[1], triangle[2]
        centroid = (self.cones[a_idx] + self.cones[b_idx] + self.cones[c_idx]) / 3
        distance = np.linalg.norm(centroid - target_point)
        return distance
    
    def _is_same_color(self, triangle):
        a_idx, b_idx, c_idx = triangle[0], triangle[1], triangle[2]
        
        if self.labels[a_idx] == self.labels[b_idx] == self.labels[c_idx]:
                return True
        return False
    
    def _get_triangle_color(self, triangle):
        a_idx, b_idx, c_idx = triangle[0], triangle[1], triangle[2]
        return self.labels[a_idx], self.labels[b_idx], self.labels[c_idx]
    