# -*- coding: utf-8 -*-
"""
@author: QSM
"""

import pandas as pd
from datetime import datetime
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import multiprocessing
import os
from tqdm import tqdm
from collections import defaultdict

# ===== Data file paths =====
output_dir = r'I:\qsm_file\跟驰参数默认\数据_20250817'
os.makedirs(output_dir, exist_ok=True)
filename = '仿真浮动车工况数据_1'

filename_path = r"I:\qsm_file\跟驰参数默认\数据_20250817\仿真浮动车工况数据_1.csv"

# edge_file = r"H:\qsm_file\仿真地图数据\SUMO地图_flow信息（最终版）0802.csv"
lane_file = r"H:\qsm_file\仿真地图数据\SUMO_NET_lane_attributes（edge）.csv"
length_file = r"H:\qsm_file\仿真地图数据\SUMO_NET_lane_lengths（edge）.csv"

# === Preload static data (for mapping) ===
lane_df = pd.read_csv(lane_file)
length_df = pd.read_csv(length_file)
length_df.rename(columns={'lane_id': 'vehicle_lane', 'length': 'lane_length'}, inplace=True)
lane_length_map = dict(zip(length_df['vehicle_lane'], length_df['lane_length']))

# === Build one-to-many lane connection mappings ===
lane_next_map = defaultdict(list)
lane_prev_map = defaultdict(list)

for _, row in lane_df.iterrows():
    lane_next_map[row['lane_id']].append(row['next_lane_id'])
    lane_prev_map[row['next_lane_id']].append(row['lane_id'])

# === Load main dataset ===
fcd_df = pd.read_csv(filename_path)
fcd_df = pd.merge(fcd_df, length_df, on='vehicle_lane', how='left')
fcd_df['vehicle_edge'] = fcd_df['vehicle_lane'].str.extract(r'(.*)_\d+$')[0]
fcd_df.sort_values(by=['vehicle_id', 'timestep_time'], inplace=True, ignore_index=True)
# fcd_df['vehicle_acceleration'] = fcd_df.groupby('vehicle_id')['vehicle_speed'].diff().fillna(0)
# fcd_df.reset_index(drop=True, inplace=True)
fcd_data = fcd_df[['timestep_time', 'vehicle_id', 'vehicle_lane', 'vehicle_speed', 'vehicle_pos']].copy()

#%%
def process_timestep_lane(data_tuple):
    timestep_time, lane_id, df = data_tuple
    lane_length = lane_length_map.get(lane_id, 0)

    # ---------- Candidate preceding vehicles (next) ----------
    next_lane_list = lane_next_map.get(lane_id, [])
    next_lanes, next_offsets = [], []

    # Same lane: include in preceding-vehicle matching, offset = 0
    next_lanes.append(lane_id)
    next_offsets.append(0)

    # First layer: offset = length of the current lane
    for nl in next_lane_list:
        next_lanes.append(nl)
        next_offsets.append(lane_length)  # Cumulative: up to the first-layer next

    # Second layer: offset = current lane length + previous layer's next-lane length
    for nl in next_lane_list:
        nl_len = lane_length_map.get(nl, 0)  # Length of the previous-layer next lane
        for n2 in lane_next_map.get(nl, []):
            next_lanes.append(n2)
            next_offsets.append(lane_length + nl_len)  # Cumulative up to the second layer

    # Deduplicate if needed: keep the smallest offset per lane
    tmp = {}
    for l, off in zip(next_lanes, next_offsets):
        tmp[l] = min(off, tmp.get(l, float('inf')))
    next_lanes = list(tmp.keys())
    next_offsets = [tmp[l] for l in next_lanes]

    next_length = dict(zip(next_lanes, next_offsets))
    next_df = fcd_data[
        (fcd_data['vehicle_lane'].isin(next_lanes)) & (fcd_data['timestep_time'] == timestep_time)
    ].copy().reset_index(drop=True)

    next_offset = next_df['vehicle_lane'].map(next_length).fillna(0)
    next_df['position_preceding'] = next_df['vehicle_pos'] + next_offset

    # ---------- Candidate following vehicles (prev) ----------
    previous_lane_list = lane_prev_map.get(lane_id, [])
    prev_lanes, prev_offsets = [], []

    # Same lane: include in following-vehicle matching, offset = 0
    prev_lanes.append(lane_id)
    prev_offsets.append(0)

    # First layer: offset = length of the predecessor lane
    # (because the following vehicle is on the predecessor lane, subtract this length)
    for pl in previous_lane_list:
        prev_lanes.append(pl)
        pl_length = lane_length_map.get(pl, 0)
        prev_offsets.append(pl_length)  # Cumulative: up to the first layer

    # Second layer: offset = length of first-layer predecessor + length of second-layer predecessor
    for pl in previous_lane_list:
        pl_len = lane_length_map.get(pl, 0)  # Length of the previous-layer predecessor lane
        for p2 in lane_prev_map.get(pl, []):
            prev_lanes.append(p2)
            pl2_len = lane_length_map.get(p2, 0)
            prev_offsets.append(pl_len + pl2_len)  # Cumulative up to the second layer

    prev_length = dict(zip(prev_lanes, prev_offsets))
    prev_df = fcd_data[
        (fcd_data['vehicle_lane'].isin(prev_lanes)) & (fcd_data['timestep_time'] == timestep_time)
    ].copy().reset_index(drop=True)

    prev_offset = prev_df['vehicle_lane'].map(prev_length).fillna(0)
    prev_df['position_following'] = prev_df['vehicle_pos'] - prev_offset

    # === Current vehicle data ===
    df = df.copy()
    df['position'] = df['vehicle_pos']
    df = df.sort_values('position')
    next_df = next_df.sort_values('position_preceding').reset_index(drop=True)
    prev_df = prev_df.sort_values('position_following').reset_index(drop=True)

    # === Preceding vehicle matching (merge_asof) ===
    result_front = pd.merge_asof(
        df,
        next_df,
        left_on='position',
        right_on='position_preceding',
        direction='forward',
        allow_exact_matches=False,
        suffixes=('', '_preceding')
    )
    result_front['following_headway_distance'] = result_front['position_preceding'] - result_front['position']

    # === Following vehicle matching (merge_asof) ===
    result_both = pd.merge_asof(
        result_front,
        prev_df,
        left_on='position',
        right_on='position_following',
        direction='backward',
        allow_exact_matches=False,
        suffixes=('', '_following')
    )
    result_both['preceding_headway_distance'] = result_both['position'] - result_both['position_following']

    # === Output field aggregation ===
    result = result_both[[
        'vehicle_id', 'timestep_time',
        'vehicle_id_preceding', 'vehicle_pos_preceding', 'vehicle_speed_preceding', 'vehicle_lane_preceding',
        'following_headway_distance',
        'vehicle_id_following', 'vehicle_pos_following', 'vehicle_speed_following', 'vehicle_lane_following',
        'preceding_headway_distance'
    ]].rename(columns={
        'vehicle_id_preceding': 'following_vehicle_id',
        'vehicle_pos_preceding': 'following_flow_pos',
        'vehicle_speed_preceding': 'following_vehicle_speed',
        'vehicle_lane_preceding': 'following_vehicle_lane',
        'vehicle_id_following': 'preceding_vehicle_id',
        'vehicle_pos_following': 'preceding_flow_pos',
        'vehicle_speed_following': 'preceding_vehicle_speed',
        'vehicle_lane_following': 'preceding_vehicle_lane'
    })

    return result.to_dict(orient='records')


#%%
if __name__ == '__main__':
    print("🚀 程序开始执行...")
    kaishi_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    global_start_time = time.time()

    cpu_count = multiprocessing.cpu_count()
    max_workers = min(61, cpu_count)
    print(f"💡 CPU核心数: {cpu_count}，使用并发进程数: {max_workers}")

    # === Build task list (group by timestep_time and lane_id) ===
    group_dict = defaultdict(list)
    for (timestep_time, lane_id), group in fcd_data.groupby(['timestep_time', 'vehicle_lane']):
        group_dict[(timestep_time, lane_id)].append(group)

    task_list = [(t, l, pd.concat(glist)) for (t, l), glist in group_dict.items()]
    print(f"📦 待处理的 timestep-lane 分组任务数：{len(task_list)}")

    # === Parallel processing (with progress bar) ===
    all_results = []
    # for idx, task in enumerate(task_list, 1):
    #     res = process_timestep_lane(task)
    #     all_results.extend(res)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(executor.map(process_timestep_lane, task_list), total=len(task_list), desc="🚗 正在处理"):
            all_results.extend(res)

    # === Merge and save results ===
    results_df = pd.DataFrame(all_results)
    merged_df = pd.merge(fcd_df, results_df, on=['vehicle_id', 'timestep_time'], how='left')
    merged_df.sort_values(by=['vehicle_id', 'timestep_time'], inplace=True, ignore_index=True)

    save_path = os.path.join(output_dir, f'{filename}_前后车辆识别.csv')
    merged_df.to_csv(save_path, index=False, encoding='utf-8')

    print(f"\n📄 结果保存至：{save_path}")
    print(f"🕒 程序开始时间: {kaishi_time}")
    print(f"✅ 程序结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    print(f"⏱ 总运行时长: {(time.time() - global_start_time) / 60:.2f} 分钟")
