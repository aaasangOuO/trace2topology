import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer, CRS
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString
import numpy as np
import alphashape
import geopandas as gpd
from shapely.ops import unary_union,nearest_points
from scipy.spatial import Delaunay
import json
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, LineString, MultiPoint, GeometryCollection, MultiLineString
from scipy.spatial.distance import cdist
from math import sqrt
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import PatchCollection
from scipy.interpolate import splprep, splev, interp1d
from collections import defaultdict
from scipy.interpolate import CubicSpline
import bezier
import matplotlib.patches as patches
from heapq import heappush, heappop
from scipy.ndimage import gaussian_filter1d
import math
import heapq
from multiprocessing import Pool

# 定义坐标转换函数
def GPSToGK(lat, lon, transformer=None, crs_WGS84=CRS.from_epsg(4326)):
    if transformer is None:
        d = int((lon + 1.5) / 3)
        format_str = '+proj=tmerc +lat_0=0 +lon_0=' + str(d * 3) + ' +k=1 +x_0=500000 +y_0=0 +ellps=WGS84 +units=m +no_defs'
        crs_GK = CRS.from_proj4(format_str)
        transformer = Transformer.from_crs(crs_WGS84, crs_GK, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y, transformer

# 读取 CSV 文件
df = pd.read_csv('new1ok13.csv')

# 初始化变换器为 None
transformer = None

# 应用坐标转换
transformed_coords = []
for index, row in df.iterrows():
    x, y, transformer = GPSToGK(row['latitude'], row['longitude'], transformer)
    transformed_coords.append((x, y))

# 将转换后的坐标添加到 DataFrame
df['x'], df['y'] = zip(*transformed_coords)

# 生成 Point 对象的列表
points = [Point(xy) for xy in zip(df['x'], df['y'])]

# 创建 GeoDataFrame
gdf = gpd.GeoDataFrame(geometry=points)

# 使用 alphashape.alphashape() 函数计算凹包
alpha = 0.4  # Alpha 参数，可根据数据调整,0-1之间，越大越贴合
concave_hull = alphashape.alphashape(gdf, alpha)

# 可视化结果
fig, ax = plt.subplots()
gdf.plot(ax=ax, color='blue', markersize=5)
gpd.GeoSeries([concave_hull]).plot(ax=ax, color='none', edgecolor='red')
ax.set_title('Road Boundary Extraction Using Alpha Shape')
plt.show()


alpha_shape = concave_hull  # 从之前的步骤获取凹包

# 去除轨迹点过少的数据
filtered_df = df.groupby('uuid').filter(lambda x: len(x) >= 100)
filtered_df['trackedTimes'] = filtered_df['trackedTimes'] / 1000  # 将时间转换为秒
filtered_df = filtered_df.groupby('uuid').filter(lambda x: 5 <= (x['trackedTimes'].max() - x['trackedTimes'].min()) <= 200)

# 滤波处理函数：移动平均滤波器
def moving_average_filter(group, window_size=10): # 窗口大小，表示用于计算平均值的数据点数量
    group['x_filtered'] = group['x'].rolling(window=window_size, min_periods=1, center=True).mean()
    group['y_filtered'] = group['y'].rolling(window=window_size, min_periods=1, center=True).mean()
    return group

filtered_df = filtered_df.groupby('uuid', group_keys=False).apply(moving_average_filter)

# 轨迹采样函数
def resample_trajectory(group, max_points=30):
    line = LineString(zip(group['x_filtered'], group['y_filtered']))
    num_points = len(group)
    num_sample_points = int(np.clip(np.interp(num_points, [10, max_points], [10, max_points]), 10, max_points))
    distances = np.linspace(0, line.length, num_sample_points)
    sampled_points = [line.interpolate(distance) for distance in distances]
    resampled_group = pd.DataFrame({'uuid': group['uuid'].iloc[0], 'x': [p.x for p in sampled_points], 'y': [p.y for p in sampled_points], 'trackedTimes': np.linspace(group['trackedTimes'].min(), group['trackedTimes'].max(), num_sample_points)})
    return resampled_group

# 对 df 进行轨迹采样
df_copy = df.copy(deep=True)
df = filtered_df.groupby('uuid', group_keys=False).apply(resample_trajectory)

# 计算贝塞尔曲线拟合的航向角
def calculate_heading_bezier(group):
    x = group['x'].values
    y = group['y'].values
    
    tck, u = splprep([x, y], s=30)
    unew = np.linspace(0, 1, len(x))
    out = splev(unew, tck)
    
    dx = np.gradient(out[0])
    dy = np.gradient(out[1])
    heading = np.arctan2(dy, dx) * 180 / np.pi  # 计算航向角
    group['heading'] = heading
    return group

df = df.groupby('uuid', group_keys=False).apply(calculate_heading_bezier)

# 使用贝塞尔曲线长度进行过滤
def bezier_curve_length(x, y):
    tck, u = splprep([x, y], s=0)
    unew = np.linspace(0, 1, 100)
    out = splev(unew, tck)
    curve = LineString(zip(out[0], out[1]))
    return curve.length

valid_uuids = []
grouped = df.groupby('uuid')
for name, group in grouped:
    if bezier_curve_length(group['x'], group['y']) >= 45:  # 使用贝塞尔曲线长度进行过滤
        valid_uuids.append(name)

# 筛选出 valid_uuids 代表的轨迹点
df = df[df['uuid'].isin(valid_uuids)]
df_copy=df_copy[df_copy['uuid'].isin(valid_uuids)]
# 打印结果 DataFrame
print(df)
# 绘制轨迹
fig, ax = plt.subplots()
# 初始化一个空的DataFrame来存储所有贝塞尔曲线数据
bezier_curves_df = pd.DataFrame()
for uuid in df['uuid'].unique():
    subset = df[df['uuid'] == uuid]
    # 检查是否有足够的不同点
    if subset[['x', 'y']].drop_duplicates().shape[0] < 4:
        #print(f"UUID {uuid} 的数据点不足。")
        continue
    
    # 准备数据，移除重复点
    x, y = subset[['x', 'y']].drop_duplicates().T.values
    
    # 尝试拟合样条曲线
    try:
        tck, u = splprep([x, y], s=20)
        unew = np.linspace(0, 1, 70)
        out = splev(unew, tck)
        plt.plot(out[0], out[1], '-', label=f'UUID {uuid} 贝塞尔曲线')
        # 将贝塞尔曲线数据转换为DataFrame
        curve_df = pd.DataFrame({'x': out[0], 'y': out[1], 'uuid': uuid})
        
        # 将当前曲线数据追加到总的DataFrame中
        bezier_curves_df = pd.concat([bezier_curves_df, curve_df], ignore_index=True)
    except ValueError as e:
        print(f"处理UUID {uuid} 时出错: {e}")
        continue

plt.title('Vehicle Trajectories in GK Coordinates')
plt.xlabel('X')
plt.ylabel('Y')
ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
plt.show()

def find_tcpps_central(df, heading_threshold=45, time_min=2, time_max=27):
    df = df.copy()  # 复制 DataFrame，避免 SettingWithCopyWarning
    df.sort_values(by=['uuid', 'trackedTimes'], inplace=True)
    result = []
    grouped = df.groupby('uuid')

    for name, group in grouped:
        headings = group['heading'].to_numpy()
        times = group['trackedTimes'].to_numpy()
        xs = group['x'].to_numpy()
        ys = group['y'].to_numpy()

        min_time_diff = float('inf')
        best_pair = None

        # 向量化时间差计算
        time_diffs = np.abs(times[:, None] - times)  # 改为绝对值
        # 创建时间差掩码，筛选出符合条件的时间差
        valid_time_mask = (time_diffs >= time_min) & (time_diffs <= time_max)
        
        # 向量化方向差计算
        heading_diffs = np.abs(headings[:, None] - headings)
        # 创建方向差掩码，筛选出符合条件的方向差
        valid_heading_mask = heading_diffs >= heading_threshold
        
        # 合并时间差和方向差掩码
        valid_mask = valid_time_mask & valid_heading_mask

        # 遍历所有有效的点对
        for i in range(len(group)):
            # 获取满足条件的点对索引
            valid_indices = np.where(valid_mask[i])[0]
            for j in valid_indices:
                if i < j:  # 确保每个点对只计算一次
                    # 计算时间间隔
                    time_diff = time_diffs[i, j]
                    # 更新最小时间间隔和最佳点对
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        best_pair = (group.iloc[i], group.iloc[j])

        # 如果找到符合条件的最佳点对，将其添加到结果列表中
        if best_pair:
            result.append(best_pair[0])
            result.append(best_pair[1])
        else:
            print(f"No valid TCPPs found for uuid {name}")

    # 将结果列表转换为 DataFrame 并返回
    return pd.DataFrame(result)

# 使用函数并打印结果
tcpps = find_tcpps_central(df)
print(tcpps)

# 提取所有TCPPs的x和y坐标
x_coords = tcpps['x'].values
y_coords = tcpps['y'].values

# 计算x和y坐标的均值
center_x = np.mean(x_coords)
center_y = np.mean(y_coords)

# 提取所有点的坐标
points = tcpps[['x', 'y']].values

# 计算所有点之间的距离矩阵
dist_matrix = np.linalg.norm(points[:, np.newaxis] - points, axis=2)

# 获取距离矩阵中的最大值
max_distance = np.max(dist_matrix)

print(f"Maximum distance between points: {max_distance}")

tcpps_uuids = tcpps['uuid'].unique()

# 从 df 中筛选出 tcpps 中包含的 uuid 的轨迹点
filtered_df = df[df['uuid'].isin(tcpps_uuids)]

# 绘制轨迹
plt.figure(figsize=(10, 10))

# 遍历每个 uuid 并绘制其轨迹
for uuid in tcpps_uuids:
    subset = filtered_df[filtered_df['uuid'] == uuid]
    plt.plot(subset['x'], subset['y'], label=f'UUID {uuid}')

plt.xlabel('X Coordinate (m)')
plt.ylabel('Y Coordinate (m)')
plt.title('Trajectories of Selected UUIDs')
plt.legend()
plt.show()

# 假设 compute_similarity 函数没有改变
def compute_similarity(TCPPi, TCPPk, D=30, omega1=0.5, omega2=0.5):
    # 计算距离差
    diff_D = np.linalg.norm(TCPPi[:2] - TCPPk[:2]) / D
    # 计算角度差
    angle_i = np.arctan2(TCPPi[1, 1] - TCPPi[0, 1], TCPPi[1, 0] - TCPPi[0, 0])
    angle_k = np.arctan2(TCPPk[1, 1] - TCPPk[0, 1], TCPPk[1, 0] - TCPPk[0, 0])
    diff_A = np.abs(angle_i - angle_k)
    
    # 相似度计算
    sim = omega1 * np.exp(-diff_D) + omega2 * np.exp(-diff_A)
    return sim

def cluster_tcpps(tcpps, D=30, omega1=0.5, omega2=0.5, Ts=0.75):
    # 预处理：为每个 uuid 创建点对
    pairs = []
    grouped = tcpps.groupby('uuid')
    for name, group in grouped:
        if len(group) == 2:
            p1, p2 = group.iloc[0], group.iloc[1]
            pairs.append({'uuid': name, 'points': np.array([[p1['x'], p1['y']], [p2['x'], p2['y']]])})

    # 将点对转换为 DataFrame 以便后续处理
    pairs_df = pd.DataFrame(pairs)
    
    unclustered = pairs_df.copy()
    clusters = []
    cluster_id = 0
    
    while not unclustered.empty:
        seed_index = unclustered.sample(n=1).index[0]  # 随机选择一个种子点对
        seed_TCPP = unclustered.loc[seed_index]
        
        # 创建一个列表收集应该被删除的索引，初始化包括种子点对
        to_remove = [seed_index]
        cluster_data = [{'Cluster ID': cluster_id, 'uuid': seed_TCPP['uuid'], 'points': seed_TCPP['points']}]

        for idx, TCPP in unclustered.iterrows():
            if idx != seed_index:
                similarity = compute_similarity(seed_TCPP['points'], TCPP['points'], D, omega1, omega2)
                if similarity > Ts:
                    cluster_data.append({'Cluster ID': cluster_id, 'uuid': TCPP['uuid'], 'points': TCPP['points']})
                    to_remove.append(idx)

        # 将聚类到一起的点对从未分类列表中删除
        unclustered.drop(to_remove, inplace=True)
        
        # 增加当前聚类到聚类列表
        clusters.extend(cluster_data)
        cluster_id += 1  # 更新聚类编号

    # 将结果转换为 DataFrame 并返回
    cluster_results = []
    for cluster in clusters:
        for point in cluster['points']:
            cluster_results.append({
                'Cluster ID': cluster['Cluster ID'],
                'uuid': cluster['uuid'],
                'x': point[0],
                'y': point[1]
            })

    return pd.DataFrame(cluster_results)

clusters = cluster_tcpps(tcpps, D=30)
print(clusters)

# 提取所有TCPPs的x和y坐标
x_coords = tcpps['x'].values
y_coords = tcpps['y'].values

# 计算x和y坐标的均值
center_x = np.mean(x_coords)
center_y = np.mean(y_coords)

# 提取所有点的坐标
points = tcpps[['x', 'y']].values

# 计算所有点之间的距离矩阵
dist_matrix = np.linalg.norm(points[:, np.newaxis] - points, axis=2)

# 获取距离矩阵中的最大值
max_distance = np.max(dist_matrix)

print(f"Maximum distance between points: {max_distance}")

# 绘制圆形
plt.figure(figsize=(10, 10))

# 绘制轨迹
for uuid in df['uuid'].unique():
    subset = df[df['uuid'] == uuid]
    plt.plot(subset['x'], subset['y'], label=f'UUID {uuid}')

# 绘制圆形
circle = plt.Circle((center_x, center_y), max_distance/2, color='r', fill=False, linestyle='--')
plt.gca().add_patch(circle)

plt.xlabel('X Coordinate (m)')
plt.ylabel('Y Coordinate (m)')
plt.title('Vehicle Trajectories with Circle')
plt.legend()
plt.show()


# 获取圆形与凹包的交点
circle_center = Point(center_x, center_y)
circle_radius = max_distance / 2
circle = circle_center.buffer(circle_radius)

# 获取凹包的外边界线段
concave_hull_exterior = LineString(concave_hull.exterior.coords)

# 获取圆形的外边界线段
circle_exterior = LineString(circle.exterior.coords)

# 计算交点
intersection = concave_hull_exterior.intersection(circle_exterior)

# 打印交点信息
print("Intersection Points:")
print(intersection)

# 提取交点
intersection_points = []
if intersection.geom_type == 'MultiPoint':
    intersection_points = list(intersection.geoms)
elif intersection.geom_type == 'GeometryCollection':
    intersection_points = [geom for geom in intersection.geoms if geom.geom_type == 'Point']
elif intersection.geom_type == 'Point':
    intersection_points = [intersection]
elif intersection.geom_type == 'LineString':
    intersection_points = [Point(coords) for coords in intersection.coords]

# 计算交点在圆形上的角度
def calculate_angle(point, center):
    dx = point.x - center.x
    dy = point.y - center.y
    return np.arctan2(dy, dx)

# 对交点按角度进行排序
intersection_points.sort(key=lambda p: calculate_angle(p, circle_center))


# 对交点进行配对
def pair_points(points):
    num_points = len(points)
    paired_lines = []
    for i in range(num_points):
        p1 = points[i]
        p2 = points[(i + 1) % num_points]
        line = LineString([p1, p2])
        paired_lines.append(line)
    return paired_lines

paired_lines = pair_points(intersection_points)


    
# 将轨迹点转换为线段
def points_to_linestring(df):
    grouped = df.groupby('uuid')
    lines = []
    for name, group in grouped:
        line = LineString(zip(group['x'], group['y']))
        lines.append(line)
    return gpd.GeoSeries(lines)

# 将轨迹点转换为线段
trajectory_lines = points_to_linestring(df)

# 检查每对交点对应的线段是否与轨迹线相交及相交数量
valid_pairs = []
for line in paired_lines:
    intersect_count = sum(line.intersects(trajectory_line) for trajectory_line in trajectory_lines)
    if intersect_count > 0:
        valid_pairs.append(line)



# 可视化结果
fig, ax = plt.subplots()
gdf.plot(ax=ax, color='blue', markersize=5)
gpd.GeoSeries([concave_hull]).plot(ax=ax, color='none', edgecolor='red')
gpd.GeoSeries([circle]).plot(ax=ax, color='none', edgecolor='black', linestyle='--')

# 绘制交点
if intersection_points:
    x_vals = [point.x for point in intersection_points]
    y_vals = [point.y for point in intersection_points]
    plt.scatter(x_vals, y_vals, color='orange', zorder=5)

# 绘制有效的交点连线
if valid_pairs:
    gpd.GeoSeries(valid_pairs).plot(ax=ax, color='green', linewidth=2)

ax.set_title('Valid Intersection Pairs')
plt.show()

lines =valid_pairs

# 导入 SHP 文件
hd_lane_gdf = gpd.read_file('hd_lane.shp')

# 过滤 vt_type 不等于 1 的车道线
hd_lane_gdf = hd_lane_gdf[hd_lane_gdf['vt_type'] != 1]
# 创建 Transformer 对象
transformer = None

# 转换车道线的坐标
converted_lines = []
for line in hd_lane_gdf.geometry:
    converted_coords = [GPSToGK(lat, lon, transformer)[:2] for lon, lat in line.coords]
    converted_line = LineString(converted_coords)
    converted_lines.append(converted_line)

# 创建包含转换后坐标的 GeoDataFrame
lane_gdf = gpd.GeoDataFrame(geometry=converted_lines)

# 计算交点数量并保存交点坐标和对应的 lane_id
results = []
lane_intersections = []
for i, line in enumerate(lines):
    intersecting_lanes = lane_gdf[lane_gdf.intersects(line)]
    num_intersections = len(intersecting_lanes)
    intersections = []
    lane_ids = []
    for idx in intersecting_lanes.index:
        lane = lane_gdf.geometry[idx]
        intersection = line.intersection(lane)
        if intersection.geom_type == 'MultiPoint':
            intersections.extend(list(intersection.geoms))
        elif intersection.geom_type == 'Point':
            intersections.append(intersection)
        elif intersection.geom_type == 'GeometryCollection':
            intersections.extend([geom for geom in intersection.geoms if geom.geom_type == 'Point'])
        lane_ids.extend([hd_lane_gdf.iloc[idx]['lane_id']] * len(intersections)) 

        # 保存车道线 ID 及其交点坐标
        for point in intersections:
            lane_intersections.append({
                'lane_id': hd_lane_gdf.iloc[idx]['lane_id'],
                'intersection_coord': (point.x, point.y)
            })
    
    # 对 lane_ids 列表去重
    lane_ids = list(dict.fromkeys(lane_ids)) 
    intersection_coords = [(point.x, point.y) for point in intersections]
    results.append({'line_id': i+1, 'num_intersections': num_intersections, 'intersection_coords': intersection_coords, 'lane_ids': lane_ids}) #lane_id值ROI线段id，lane_ids指车道线id，从hd_lane.json中获取


# 创建一个字典来存储每个UUID与线段的交点信息
uuid_intersections = defaultdict(list)

# 遍历每个UUID的贝塞尔曲线
for uuid in df['uuid'].unique():
    # 获取该UUID的曲线数据
    curve_data = df[df['uuid'] == uuid]
    curve_line = LineString(list(zip(curve_data['x'], curve_data['y'])))
    
    # 检查该曲线与每条线段的交点数量
    for i, line in enumerate(lines):
        if curve_line.intersects(line):
            intersection_points = curve_line.intersection(line)
            # intersection_points 可能是一个点或多个点
            if intersection_points.geom_type == 'Point':
                uuid_intersections[uuid].append((i+1, (intersection_points.x, intersection_points.y)))
            elif intersection_points.geom_type == 'MultiPoint':
                for point in intersection_points.geoms:
                    uuid_intersections[uuid].append((i+1, (point.x, point.y)))

# 使用交点计数来筛选符合条件的UUID
intersections_1 = {uuid: details for uuid, details in uuid_intersections.items() if len(details) == 1}
intersections_2 = {uuid: details for uuid, details in uuid_intersections.items() if len(details) == 2}


def calculate_distance(point1, point2):
    """计算两点之间的欧几里得距离"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
def replace_with_closest_lane_coords(intersections, result):
    for uuid, details in intersections.items():
        for i, (line_id, coord) in enumerate(details):
            for res in result:
                if res['line_id'] == line_id:
                    # 找到匹配的line_id，寻找距离最近的交点
                    min_distance = float('inf')
                    closest_res_coord = None
                    closest_lane_id = None
                    for j, res_coord in enumerate(res['intersection_coords']):
                        distance = calculate_distance(coord, res_coord)
                        if distance < min_distance:
                            min_distance = distance
                            closest_res_coord = res_coord
                            closest_lane_id = res['lane_ids'][j]
                    if closest_res_coord is not None:
                        # 替换原始交点为最近的车道线交点，并保存对应的车道线ID
                        details[i] = (line_id, closest_res_coord, closest_lane_id)
    return intersections


# 使用上述函数替换 intersections_1 和 intersections_2 中的交点坐标，并存储在新的数据结构中
new_intersections_1 = replace_with_closest_lane_coords(intersections_1, results)
new_intersections_2 = replace_with_closest_lane_coords(intersections_2, results)

# 提取并去重 new_intersections_2 中的车道线ID对
lane_id_pairs = defaultdict(list)
for uuid, details in new_intersections_2.items():
    if len(details) == 2:
        # 提取按轨迹顺序的车道线ID对
            lane_id_pair = (details[0][2], details[1][2])
            lane_id_pairs[lane_id_pair].append((uuid, details))
print(lane_id_pairs)
# 提取并去重 intersections_1 中的车道线ID
unique_lane_ids = set()

# 打印 details 以确认其结构
for uuid, details in intersections_1.items():
    print(f"UUID: {uuid}, Details: {details}")
    for detail in details:
        print(f"Detail: {detail}")
        line_id, coord = detail[:2]  # 根据结构解包前两个值
        unique_lane_ids.add(line_id)

# 将去重后的车道线ID转换为列表并输出
unique_lane_ids_list = list(unique_lane_ids)
print("Unique lane IDs in intersections_1:")
print(unique_lane_ids_list)

# 计算欧几里得距离的函数
def calculate_distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

# 计算方向角的函数
def calculate_angle(start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle

# 从 results 中提取交点和车道线ID的映射
intersection_to_lane = {}
for result in results:
    for i, coord in enumerate(result['intersection_coords']):
        intersection_to_lane[coord] = result['lane_ids'][i]

# 从 df 中提取轨迹末端信息
def get_trajectory_end(df, uuid):
    trajectory = df[df['uuid'] == uuid]
    if trajectory.empty:
        return None
    end_point = trajectory.iloc[-1][['x', 'y']].values
    return tuple(end_point)

# 匹配末端与车道线交点
def match_end_to_lane(intersections, df, results):
    matched_intersections = {}
    
    for uuid, details in intersections.items():
        if len(details) == 0:
            continue
        
        # 获取末端之前的line_id和coord
        prev_line_id, prev_coord, prev_lane_id = details[-1]
        
        # 从 df 中获取末端信息
        end_coord = get_trajectory_end(df, uuid)
        if end_coord is None:
            continue
        
        # 从 results 中找到最近的交点和车道线ID
        min_distance = float('inf')
        closest_coord = None
        closest_lane_id = None
        
        for result in results:
            for i, coord in enumerate(result['intersection_coords']):
                distance = calculate_distance(end_coord, coord)
                if distance < min_distance:
                    min_distance = distance
                    closest_coord = coord
                    closest_lane_id = result['lane_ids'][i]
        
        # 如果找到最近的交点，则添加到 matched_intersections
        if closest_coord is not None and closest_lane_id is not None:
            matched_intersections[uuid] = (prev_coord, prev_lane_id, closest_coord, closest_lane_id)
    
    return matched_intersections

# 匹配末端与车道线交点
matched_intersections = match_end_to_lane(intersections_1, df, results)

# 输出匹配结果
print(matched_intersections)



lane_ids_pairs = defaultdict(list)
# 遍历 matched_intersections 并转换为新的结构
for uuid, (prev_coord, prev_lane_id, closest_coord, closest_lane_id) in matched_intersections.items():
    lane_ids_pair = (prev_lane_id, closest_lane_id)
    details = [(prev_coord, prev_lane_id), (closest_coord, closest_lane_id)]
    lane_ids_pairs[lane_ids_pair].append((uuid, details))
   


# 输出去重后的车道线ID对
print("Unique lane ID pairs in new_intersections_1:")
for lane_ids_pair, trajectories in lane_ids_pairs.items():
    print(f"{lane_ids_pair}: {[uuid for uuid, _ in trajectories]}")

# 计算距离的函数
def calculate_distance(a, b):
    return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

def calculate_loss(current, next, grid, angle_weight=0.5):
    """计算从当前点到下一个点的损失。"""
    dx, dy = next[0] - current[0], next[1] - current[1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)

    # 计算密度损失
    density_loss = grid[next]['counts']

    # 计算方向损失
    if grid[next]['angles']:
        avg_angle = np.mean(grid[next]['angles'])
        direction_loss = min(abs(angle_deg - avg_angle), 360 - abs(angle_deg - avg_angle))
    else:
        direction_loss = 180  # 如果没有方向信息，假设方向损失最大

    # 综合损失，可以调整 angle_weight 来平衡两种损失的权重
    total_loss = 1 / (1 + density_loss) + angle_weight * direction_loss
    return total_loss

def heuristic(a, b):
    """计算启发式距离（欧几里得距离）。"""
    return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

def a_star(grid, start, goal, num_x_grids, num_y_grids, angle_weight=0.5):
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))

    while oheap:
        current = heappop(oheap)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)
            if neighbor in close_set:
                continue

            # Ensure the neighbor is within grid bounds
            if 0 <= neighbor[0] < num_x_grids and 0 <= neighbor[1] < num_y_grids:
                if neighbor not in grid or grid[neighbor]['counts'] == 0:
                    continue  # Skip if grid cell is not passable or does not exist

                tentative_g_score = gscore[current] + calculate_loss(current, neighbor, grid, angle_weight)

                if tentative_g_score < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heappush(oheap, (fscore[neighbor], neighbor))
            else:
                continue

    return False

def calculate_grid_density_and_direction(df, grid_size, x_min, x_max, y_min, y_max):
    """计算网格的密度和方向"""
    num_x_grids = int((x_max - x_min) / grid_size) + 1
    num_y_grids = int((y_max - y_min) / grid_size) + 1

    # 初始化density，给每个网格初始密度为1，初始角度为0
    density = defaultdict(lambda: {'counts': 1, 'angles': [0]})

    for uuid in df['uuid'].unique():
        trajectory = df[df['uuid'] == uuid]
        for i in range(len(trajectory) - 1):
            x1, y1 = trajectory.iloc[i][['x', 'y']]
            x2, y2 = trajectory.iloc[i + 1][['x', 'y']]
            grid_x1, grid_y1 = int((x1 - x_min) / grid_size), int((y1 - y_min) / grid_size)
            grid_x2, grid_y2 = int((x2 - x_min) / grid_size), int((y2 - y_min) / grid_size)
            angle = math.atan2(y2 - y1, x2 - x1)
            density[(grid_x1, grid_y1)]['counts'] += 1
            density[(grid_x1, grid_y1)]['angles'].append(angle)
            density[(grid_x2, grid_y2)]['counts'] += 1
            density[(grid_x2, grid_y2)]['angles'].append(angle)

    direction = {k: np.mean(v['angles']) for k, v in density.items()}

    # 打印网格密度信息以调试
    for k, v in density.items():
        print(f"Grid {k}: Counts = {v['counts']}, Angles = {v['angles']}")

    return density, direction, num_x_grids, num_y_grids

# 设置网格大小
grid_size = 1

# 构建 lane_id 到 intersection_coord 的映射
lane_id_to_coord = {entry['lane_id']: entry['intersection_coord'] for entry in lane_intersections}

# 获取轨迹数据的总体边界
x_min, x_max = df_copy['x'].min(), df_copy['x'].max()
y_min, y_max = df_copy['y'].min(), df_copy['y'].max()

# 定义邻居节点
neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

# 绘制图形
plt.figure(figsize=(10, 10))
paths_results_2=[]
# 计算并绘制每类轨迹的路径
for lane_id_pair, trajectories in lane_id_pairs.items():
    print(f"Lane ID Pair: {lane_id_pair}")

    if lane_id_pair[0] not in lane_id_to_coord or lane_id_pair[1] not in lane_id_to_coord:
        print(f"Lane ID {lane_id_pair[0]} or {lane_id_pair[1]} not found in lane_id_to_coord.")
        continue

    start_coord = lane_id_to_coord[lane_id_pair[0]]
    goal_coord = lane_id_to_coord[lane_id_pair[1]]

    # 打印起点和终点的坐标
    print(f"Start lane ID: {lane_id_pair[0]}, Start coordinate: {start_coord}")
    print(f"Goal lane ID: {lane_id_pair[1]}, Goal coordinate: {goal_coord}")

    # 获取这类轨迹的数据
    class_df = df_copy[df_copy['uuid'].isin([uuid for uuid, _ in trajectories])]

    if class_df.empty:
        print(f"No trajectory data found for lane ID pair {lane_id_pair}.")
        continue

    # 计算这类轨迹的网格密度和方向
    density, direction, num_x_grids, num_y_grids = calculate_grid_density_and_direction(class_df, grid_size, x_min, x_max, y_min, y_max)

    # 将坐标转换为网格坐标
    start = (int((start_coord[0] - x_min) / grid_size), int((start_coord[1] - y_min) / grid_size))
    goal = (int((goal_coord[0] - x_min) / grid_size), int((goal_coord[1] - y_min) / grid_size))

    print(f"Start grid position: {start}, Goal grid position: {goal}")

    # 检查起点和目标点是否可通行
    if start not in density or density[start]['counts'] == 0:
        print(f"Start point {start} is not passable. Density: {density.get(start, 'Not in density')}")
        # 在附近寻找最近的可通过单元
        for i, j in neighbors:
            neighbor = (start[0] + i, start[1] + j)
            if 0 <= neighbor[0] < num_x_grids and 0 <= neighbor[1] < num_y_grids and density.get(neighbor, {}).get('counts', 0) > 0:
                start = neighbor
                print(f"New start point found: {start}")
                break
        else:
            continue

    if goal not in density or density[goal]['counts'] == 0:
        print(f"Goal point {goal} is not passable. Density: {density.get(goal, 'Not in density')}")
        # 在附近寻找最近的可通过单元
        for i, j in neighbors:
            neighbor = (goal[0] + i, goal[1] + j)
            if 0 <= neighbor[0] < num_x_grids and 0 <= neighbor[1] < num_y_grids and density.get(neighbor, {}).get('counts', 0) > 0:
                goal = neighbor
                print(f"New goal point found: {goal}")
                break
        else:
            continue

    # 使用 A* 算法进行路径搜索
    path = a_star(density, start, goal, num_x_grids, num_y_grids, angle_weight=0.5)
    print(f"    Path: {path}")
    
    if path:
        # 存储路径结果，加入可信度字段
        paths_results_2.append({
            'start_lane_id': lane_id_pair[0],
            'path': path,
            'end_lane_id': lane_id_pair[1],
            'confidence': 0.8  # 设置可信度为0.8
        })

    # 绘制路径
    if path:
        path_x = [x_min + p[0] * grid_size for p in path]
        path_y = [y_min + p[1] * grid_size for p in path]
        if len(path_x) > 3:
            smooth_x = gaussian_filter1d(path_x, sigma=4)
            smooth_y = gaussian_filter1d(path_y, sigma=4)
            plt.plot(smooth_x, smooth_y, '-', label=f'Path for {lane_id_pair}')
        else:
            plt.plot(path_x, path_y, 'o-', label=f'Path for {lane_id_pair}')

# 保存图像
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Paths for Different Lane ID Pairs')
plt.legend()
plt.grid(True)
plt.savefig("trajectories_paths.png")
plt.show()


grid_size = 2
# 绘制图形
plt.figure(figsize=(10, 10))

paths_results=[]
# 计算并绘制每类轨迹的路径
for lane_ids_pair, trajectories in lane_id_pairs.items():
    print(f"Lane ID Pair: {lane_ids_pair}")

    if lane_ids_pair[0] not in lane_id_to_coord or lane_ids_pair[1] not in lane_id_to_coord:
        print(f"Lane ID {lane_ids_pair[0]} or {lane_ids_pair[1]} not found in lane_id_to_coord.")
        continue

    start_coord = lane_id_to_coord[lane_ids_pair[0]]
    goal_coord = lane_id_to_coord[lane_ids_pair[1]]

    # 打印起点和终点的坐标
    print(f"Start lane ID: {lane_ids_pair[0]}, Start coordinate: {start_coord}")
    print(f"Goal lane ID: {lane_ids_pair[1]}, Goal coordinate: {goal_coord}")

    # 获取这类轨迹的数据
    class_df = df_copy[df_copy['uuid'].isin([uuid for uuid, _ in trajectories])]

    if class_df.empty:
        print(f"No trajectory data found for lane ID pair {lane_ids_pair}.")
        continue

    # 计算这类轨迹的网格密度和方向
    density, direction, num_x_grids, num_y_grids = calculate_grid_density_and_direction(class_df, grid_size, x_min, x_max, y_min, y_max)

    # 将坐标转换为网格坐标
    start = (int((start_coord[0] - x_min) / grid_size), int((start_coord[1] - y_min) / grid_size))
    goal = (int((goal_coord[0] - x_min) / grid_size), int((goal_coord[1] - y_min) / grid_size))

    print(f"Start grid position: {start}, Goal grid position: {goal}")

    # 检查起点和目标点是否可通行
    if start not in density or density[start]['counts'] == 0:
        print(f"Start point {start} is not passable. Density: {density.get(start, 'Not in density')}")
        # 在附近寻找最近的可通过单元
        found_start = False
        for i in range(1, 5):  # 检查不同半径范围内的网格单元
            for dx in range(-i, i + 1):
                for dy in range(-i, i + 1):
                    neighbor = (start[0] + dx, start[1] + dy)
                    if 0 <= neighbor[0] < num_x_grids and 0 <= neighbor[1] < num_y_grids and density.get(neighbor, {}).get('counts', 0) > 0:
                        start = neighbor
                        print(f"New start point found: {start}")
                        found_start = True
                        break
                if found_start:
                    break
            if found_start:
                break
        if not found_start:
            continue

    if goal not in density or density[goal]['counts'] == 0:
        print(f"Goal point {goal} is not passable. Density: {density.get(goal, 'Not in density')}")
        # 在附近寻找最近的可通过单元
        found_goal = False
        for i in range(1, 5):  # 检查不同半径范围内的网格单元
            for dx in range(-i, i + 1):
                for dy in range(-i, i + 1):
                    neighbor = (goal[0] + dx, goal[1] + dy)
                    if 0 <= neighbor[0] < num_x_grids and 0 <= neighbor[1] < num_y_grids and density.get(neighbor, {}).get('counts', 0) > 0:
                        goal = neighbor
                        print(f"New goal point found: {goal}")
                        found_goal = True
                        break
                if found_goal:
                    break
            if found_goal:
                break
        if not found_goal:
            continue

    # 使用 A* 算法进行路径搜索
    path = a_star(density, start, goal, num_x_grids, num_y_grids, angle_weight=0.5)
    print(f"    Path: {path}")

    if path:
        # 存储路径结果，加入可信度字段
        paths_results.append({
            'start_lane_id': lane_ids_pair[0],
            'path': path,
            'end_lane_id': lane_ids_pair[1],
            'confidence': 0.5  # 设置可信度为0.5
        })

        # 绘制路径
        path_x = [x_min + p[0] * grid_size for p in path]
        path_y = [y_min + p[1] * grid_size for p in path]
        if len(path_x) > 3:
            smooth_x = gaussian_filter1d(path_x, sigma=4)
            smooth_y = gaussian_filter1d(path_y, sigma=4)
            plt.plot(smooth_x, smooth_y, '-', label=f'Path for {lane_ids_pair}')
        else:
            plt.plot(path_x, path_y, 'o-', label=f'Path for {lane_ids_pair}')

# 输出路径结果
print("Paths Results:")
for result in paths_results:
    print(result)

# 保存图像
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Paths for Different Lane ID Pairs')
plt.legend()
plt.grid(True)
plt.savefig("trajectories_paths.png")
plt.show()


# 合并两个列表
merged_paths_results = paths_results + paths_results_2

# 输出合并后的结果
print("Merged Paths Results:")
for result in merged_paths_results:
    print(result)

# 从df中提取uuid和数量
uuids = df_copy['uuid'].unique().tolist()
num = len(uuids)
# 从df中提取uuid和数量
unique_uuids = df_copy['uuid'].unique().tolist()
num_uuids = len(unique_uuids)

# 提取circle_exterior的圆形边界数据
circle_exterior = LineString(circle.exterior.coords)  # 假设circle_exterior已经定义
range_data = [list(coord) for coord in circle_exterior.coords]

# 构建输入部分
input_data = [{'uuid': uuid, 'nums': num} for uuid in uuids]

# 构建元数据部分
metadata_data = [{'CRS': 'crs_WGS84', 'range': range_data}]

# 构建拓扑部分
topology_data = [{
    'lane_id': result['start_lane_id'],
    'connection': result['end_lane_id'],
    'reference path': result['path'],
    'confidence': result['confidence']
} for result in merged_paths_results]

# 构建最终的数据结构
final_data = {
    'input': {
        'traces': input_data,
        'trace nums': num_uuids
    },
    'metadata': {'Intersection': metadata_data},
    'topology': topology_data
}

# 将数据保存为JSON文件
with open('results.json', 'w') as json_file:
    json.dump(final_data, json_file, indent=4)

# 打印输出确认
print("JSON data saved to merged_paths_results.json")

''' #使用dscan算法对交点聚类,根据聚类中心数量获取车道线数量
def cluster_intersections(intersections):
    if len(intersections) < 2:
        return 0

    coords = np.array([(point.x, point.y) for point in intersections])
    clustering = DBSCAN(eps=0.5, min_samples=2).fit(coords)
    num_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    return num_clusters

# 计算并添加聚类数量到 lines
lines_with_clusters = []
for i, line in enumerate(valid_pairs):
    # 获取交点
    intersections = []
    for trajectory_line in trajectory_lines:  
        if line.intersects(trajectory_line):
            intersection = line.intersection(trajectory_line)
            if intersection.geom_type == 'MultiPoint':
                intersections.extend(list(intersection.geoms))
            elif intersection.geom_type == 'Point':
                intersections.append(intersection)
            elif intersection.geom_type == 'GeometryCollection':
                intersections.extend([geom for geom in intersection.geoms if geom.geom_type == 'Point'])

    # 聚类交点
    num_clusters = cluster_intersections(intersections)
    lines_with_clusters.append({'line': line, 'num_clusters': num_clusters, 'line_id': i + 1})

# 输出结果
for line_info in lines_with_clusters:
    print(f"Line {line_info['line_id']} has {line_info['num_clusters']} clusters.")

# 可视化结果
fig, ax = plt.subplots()
gdf.plot(ax=ax, color='blue', markersize=5)
gpd.GeoSeries([concave_hull]).plot(ax=ax, color='none', edgecolor='red')
lane_gdf.plot(ax=ax, color='orange', linewidth=1)
plt.title('Overall Map with Lane Information')
plt.show()

# 单独绘制 lines_with_clusters
fig, ax = plt.subplots()
for line_info in lines_with_clusters:
    gpd.GeoSeries([line_info['line']]).plot(ax=ax, color='green', linewidth=2)
    intersections = []
    for trajectory_line in trajectory_lines:
        if line_info['line'].intersects(trajectory_line):
            intersection = line_info['line'].intersection(trajectory_line)
            if intersection.geom_type == 'MultiPoint':
                intersections.extend(list(intersection.geoms))
            elif intersection.geom_type == 'Point':
                intersections.append(intersection)
            elif intersection.geom_type == 'GeometryCollection':
                intersections.extend([geom for geom in intersection.geoms if geom.geom_type == 'Point'])
    x_vals = [point.x for point in intersections]
    y_vals = [point.y for point in intersections]
    plt.scatter(x_vals, y_vals, color='orange', zorder=5)
    # 添加line的序号
    line_center = line_info['line'].interpolate(0.5, normalized=True)
    ax.annotate(str(line_info['line_id']), (line_center.x, line_center.y), color='red', fontsize=12, ha='center')

ax.set_title('Lines with Clusters')
plt.show()'''


