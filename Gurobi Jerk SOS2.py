import os, gc, re
import sys
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from gurobipy import Model, GRB, quicksum
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import truncnorm
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置随机种子
seed = 42  # 固定一个整数即可
rng = np.random.default_rng(seed)

# ================= == 路径与日志配置 =============================
OUTPUT_DIR = r"I:\qsm_file\跟驰参数默认\数据_20250817"
os.makedirs(OUTPUT_DIR, exist_ok=True)

file_name = '仿真浮动车工况数据_1'
file_path = r"I:\qsm_file\跟驰参数默认\数据_20250817\仿真浮动车工况数据_1_工况划分.csv"

plt_dir = rf"I:\qsm_file\跟驰参数默认\数据_20250817\车辆速度对比_{file_name}"
os.makedirs(plt_dir, exist_ok=True)


# 日志配置
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
file_handler = logging.FileHandler('my.log', encoding='utf-8')
formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# =================== 包络线数据加载 =============================
# 加速度包络线 (Amin/Amax)
accel_env = pd.read_csv(r"H:\qsm_file\fcd_data\Jerk不分模式_final\Accel包络线_99.csv", encoding='utf-8')
speeds     = sorted(accel_env['speed_kmh'].astype(int).unique())
spd_dict   = {s: s/3.6 for s in speeds}  # km/h -> m/s
Amin_dict  = dict(zip(speeds, accel_env['q1_smooth'].astype(float)))
Amax_dict  = dict(zip(speeds, accel_env['q99_smooth'].astype(float)))

# 抖动包络线 (JerkMin/JerkMax)
jerk_env   = pd.read_csv(r"H:\qsm_file\fcd_data\Jerk不分模式_final\Jerk包络线_99.csv", encoding='utf-8')
accels     = sorted(jerk_env['accel_round'].astype(float).unique())
acc_dict   = {a: a for a in accels}  # accel_round -> accel
JerkMin_dict = dict(zip(accels, jerk_env['q1_smooth'].astype(float)))
JerkMax_dict = dict(zip(accels, jerk_env['q99_smooth'].astype(float)))

# 全局模型限值
V_MAX = 20      # 最大速度 (m/s)
A_UPPER = 5      # 加速度上界
A_LOWER = -4     # 加速度下界
JERK_UP = 4  # 抖动上界
JERK_LOW = -5  # 抖动下界


df_all = pd.read_csv(file_path)
df_all = df_all.drop_duplicates(subset=['工况ID', 'timestep_time'], keep='first', ignore_index=True)

df_all['distance_m'] = df_all['vehicle_speed']
df_all['cumulative_distance'] = df_all.groupby('vehicle_id')['distance_m'].cumsum()
first_pos = df_all.groupby('vehicle_id')['vehicle_pos'].first()
df_all['vehicle_odometer'] = df_all['vehicle_id'].map(first_pos) + df_all['cumulative_distance']
df_all['vehicle_odometer'] = df_all['vehicle_odometer'].round(4)

car_speed = df_all.groupby('vehicle_id').agg(expect_speed=('vehicle_speed', 'max')).reset_index()

df_all.sort_values(['vehicle_id', 'timestep_time'], inplace=True)
min_fhd = df_all['following_headway_distance'].min()
min_phd = df_all['preceding_headway_distance'].min()
safe_distance = min(min_fhd, min_phd)

df_all['preceding_mileage_pos'] = df_all['vehicle_odometer'] + df_all['preceding_headway_distance']
df_all['following_mileage_pos'] = df_all['vehicle_odometer'] - df_all['following_headway_distance']


# =================== 参数准备 =============================
def prepare_ampl_param(df_trip: pd.DataFrame, vehicle_id) -> dict:
    # 时间与轨迹
    time_vals = sorted(df_trip['timestep_time'].unique())
    base_v = df_trip.set_index('timestep_time')['vehicle_speed'].to_dict()
    base_x = df_trip.set_index('timestep_time')['vehicle_odometer'].to_dict()

    # 前后车约束
    preced_idx = df_trip['preceding_vehicle_id'].notna()
    follow_idx = df_trip['following_vehicle_id'].notna()
    preced_time = df_trip.loc[preced_idx, 'timestep_time'].tolist()
    follow_time = df_trip.loc[follow_idx, 'timestep_time'].tolist()
    preced_x = df_trip.set_index('timestep_time').loc[preced_time, 'preceding_mileage_pos'].to_dict()
    follow_x = df_trip.set_index('timestep_time').loc[follow_time, 'following_mileage_pos'].to_dict()

    v_max = float(car_speed.loc[car_speed['vehicle_id'] == vehicle_id, 'expect_speed'].iloc[0])

    # 软约束指示
    b_v = {k: int(base_v[k] == 0) for k in base_v}
    if b_v:
        keys = list(b_v)
        b_v[keys[0]] = 1
        b_v[keys[-1]] = 1
    b_x = b_v.copy()

    # 汇总参数字典
    return {
        'vehicle_id': vehicle_id,
        # 时间序列与约束时刻
        'time': time_vals, 'preced_time': set(preced_time), 'follow_time': set(follow_time),
        # 原始轨迹
        'base_v': base_v, 'base_x': base_x,
        'preced_x': preced_x, 'follow_x': follow_x,
        # 软约束指示
        'b_v': b_v, 'b_x': b_x,
        # 安全与大M
        'x_safe': safe_distance, 'M': 1e6,
        # 速度/加速度/抖动界限
        'v_limit': V_MAX, 'v_max': v_max,
        'a_upper': A_UPPER, 'a_lower': A_LOWER,
        'jerk_upper': JERK_UP, 'jerk_lower': JERK_LOW,
        # 包络线映射
        'accel': accels, 'acc': acc_dict,
        'Amin': Amin_dict, 'Amax': Amax_dict,
        'JerkMin': JerkMin_dict, 'JerkMax': JerkMax_dict,
        # 权重
        'w_a': 10, 'w_v': 1, 'w_x': 1,
    }


# =================== 可视化 =============================
def plt_speed(car_data, car_id):
    car_data = car_data.sort_values(by='timestep_time')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True, dpi=180)
    ax.plot(car_data['timestep_time'], car_data['opt_v'] * 3.6, label='后处理-jerk约束')
    ax.plot(car_data['timestep_time'], car_data['base_v'] * 3.6, label="SUMO默认方案")
    ax.set_title('Vehicle Speed Over Time', fontsize=16)
    ax.set_xlabel('Timestep', fontsize=14)
    ax.set_ylim(0, 90)
    ax.set_ylabel('Speed (km/h)', fontsize=14)
    ax.legend(title='Data Type', fontsize=12)
    ax.grid(True)
    car_id = re.sub(r'[\\/*?:"<>|.]', "_", str(car_id))
    plt.savefig(os.path.join(plt_dir, f'{car_id}.png'))
    plt.close()


# =================== 优化模型 =============================
def run_gurobi(params: dict, cond_id) -> pd.DataFrame:
    time = list(params['time'])             # eg. [1211,1212,...,1225] (ordered)
    accels = list(params['accel'])          # ordered accel grid (e.g. [-4.1, -4.0, ..., 4.1])
    times1 = time[1:]                       # ord(t) > 1 的集合（用 prev(t) 时有效）
    prev_of = { time[i]: time[i-1] for i in range(1, len(time)) }

    # 将 JerkMax/JerkMin 对齐到 accel 网格（键可为数值或索引）
    def _align_piecewise_table(table_dict):
        # 允许键是“加速度值”或“0..len(accels)-1 的索引”
        # 返回与 accels 同长度的数组
        arr = np.zeros(len(accels), dtype=float)
        # 尝试按值匹配
        val2idx = {float(v): i for i, v in enumerate(accels)}
        try_as_value = True
        for k, v in table_dict.items():
            kf = float(k)
            if kf not in val2idx:
                try_as_value = False
                break
            arr[val2idx[kf]] = float(v)
        if not try_as_value:
            # 按索引写入
            arr = np.zeros(len(accels), dtype=float)
            for k, v in table_dict.items():
                idx = int(k)
                if not (0 <= idx < len(accels)):
                    raise ValueError(f"Piecewise 表键越界：{idx}")
                arr[idx] = float(v)
        return arr

    try:
        jmax_arr = _align_piecewise_table(params['JerkMax'])
        jmin_arr = _align_piecewise_table(params['JerkMin'])
    except Exception as e:
        logger.exception(f"[工况{cond_id}] JerkMax/JerkMin 表对齐失败: {e}")
        return pd.DataFrame()

    # ------- 建模 -------
    try:
        m = Model()
        m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', 600)
        m.setParam('Threads', 2)

        # 变量
        jerk = m.addVars(time, lb=params['jerk_lower'], ub=params['jerk_upper'], name='jerk')
        a    = m.addVars(time, lb=params['a_lower'],    ub=params['a_upper'],    name='a')
        v    = m.addVars(time, lb=0.0,                  ub=params['v_limit'],    name='v')
        x    = m.addVars(time, lb=-GRB.INFINITY, name='x')

        # SOS2 权重（lam）：仅 ord(t)>1 定义；每个 t 的 lam 向量满足 sum=1 且 SOS2
        lam = {(t, y): m.addVar(lb=0.0, name=f"lam[{t},{y}]") for t in times1 for y in range(len(accels))}

        # 插值中的辅助量（Jmax_at_a/Jmin_at_a）——可用临时表达式，不一定要单独变量
        # 这里直接以线性表达式构建，不新建 var

        # ---- SOS2 约束 & a 的插值 & jerk 界 ----
        for t in times1:
            idxs = list(range(len(accels)))
            lam_vars = [lam[(t, y)] for y in idxs]
            # sum λ = 1
            m.addConstr(quicksum(lam_vars) == 1.0, name=f"lam_sum[{t}]")
            # a[prev(t)] = Σ accels[y] * λ
            tp = prev_of[t]
            m.addConstr(a[tp] == quicksum(accels[y] * lam[(t, y)] for y in idxs), name=f"a_interp[{t}]")
            # 声明 SOS2（顺序使用 0..Y-1）
            m.addSOS(GRB.SOS_TYPE2, lam_vars, idxs)

            # Jmax/Jmin 插值值（线性表达式）
            Jmax_at_a = quicksum(jmax_arr[y] * lam[(t, y)] for y in idxs)
            Jmin_at_a = quicksum(jmin_arr[y] * lam[(t, y)] for y in idxs)

            # jerk 的状态相关界：jerk[prev(t)] ∈ [Jmin_at_a, Jmax_at_a]
            m.addConstr(jerk[tp] <= Jmax_at_a, name=f"j_up_from_a[{t}]")
            m.addConstr(jerk[tp] >= Jmin_at_a, name=f"j_lo_from_a[{t}]")

        # ---- 运动学（Δt = 1；全部用 prev(t)）----
        for t in times1:
            tp = prev_of[t]
            m.addConstr(v[t] == v[tp] + a[tp] + 0.5 * jerk[tp],               name=f"update_v[{t}]")
            m.addConstr(a[t] == a[tp] + jerk[tp],                             name=f"update_a[{t}]")
            m.addConstr(x[t] == x[tp] + (v[tp] + 0.5*a[tp] + jerk[tp]/6.0),   name=f"update_x[{t}]")

        # ---- 跟驰安全 ----
        # 注意：preced_time/follow_time 需要是 time 的子集（键一致）
        for t in params['preced_time']:
            if t not in time or t not in params['preced_x']:
                continue  # 跳过缺数据的时刻
            m.addConstr(params['preced_x'][t] - x[t] >= params['x_safe'] + 0.1 * v[t], name=f"preced[{t}]")
        for t in params['follow_time']:
            if t not in time or t not in params['follow_x']:
                continue
            m.addConstr(x[t] - params['follow_x'][t] >= params['x_safe'] + 0.1 * params['v_max'], name=f"follow[{t}]")

        # ---- 软约束（大 M，b=1 时锁定等式；补充 v_soft_lower1）----
        M_big = params['M']
        bx = params['b_x']; bv = params['b_v']
        bx_missing = [t for t in time if t not in bx]
        bv_missing = [t for t in time if t not in bv]
        if bx_missing or bv_missing:
            raise KeyError(f"b_x/b_v 缺少键：b_x缺{len(bx_missing)}，b_v缺{len(bv_missing)}")

        for t in time:
            m.addConstr(x[t] - params['base_x'][t] <=  M_big * (1 - bx[t]), name=f"x_soft_u[{t}]")
            m.addConstr(x[t] - params['base_x'][t] >= -M_big * (1 - bx[t]), name=f"x_soft_l[{t}]")
            m.addConstr(v[t] - params['base_v'][t] <=  M_big * (1 - bv[t]), name=f"v_soft_u[{t}]")
            m.addConstr(v[t] - params['base_v'][t] >= -M_big * (1 - bv[t]), name=f"v_soft_l[{t}]")
            # AMPL 里有一个严格不等式 v[t] > -M*b_v[t]，这里用 >= 近似
            m.addConstr(v[t] >= - M_big * bv[t], name=f"v_soft_l1[{t}]")

        # ---- 目标函数（与模型一致，仅最小化 a^2；需要可扩展 w_v/w_x 再加）----
        obj = params['w_a'] * quicksum(a[t]*a[t] for t in time)
        # 若需要加入锚定项，可解开注释（建议仅对 b=0 的位置生效）：
        obj += params['w_v'] * quicksum((v[t]-params['base_v'][t])*(v[t]-params['base_v'][t]) * (1-bv[t]) for t in time)
        # obj += params['w_x'] * quicksum((x[t]-params['base_x'][t])*(x[t]-params['base_x'][t]) * (1-bx[t]) for t in time)

        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()

        status = m.Status
        solcnt = m.SolCount
        logger.info(f"[工况{cond_id}] Gurobi 状态: {status}, 可行解数: {solcnt}")

        if solcnt > 0:
            df_out = pd.DataFrame({
                'timestep_time': time,
                'vehicle_id': params['vehicle_id'],
                'maxSpeed': params['v_max'],
                'opt_a': [a[t].X for t in time],
                'opt_v': [v[t].X for t in time],
                'opt_x': [x[t].X for t in time],
                'opt_jerk': [jerk[t].X for t in time]
                # 'base_v': [params['base_v'][t] for t in time]
            })

            # plt_speed(df_out, f'{params["vehicle_id"]}_{cond_id}')

            return df_out

        # —— 无解兜底
        return pd.DataFrame({
            'timestep_time': time,
            'vehicle_id': params['vehicle_id'],
            'maxSpeed': params['v_max'],
            'opt_a': [np.nan]*len(time),
            'opt_v': [np.nan]*len(time),
            'opt_x': [np.nan]*len(time),
            'opt_jerk': [np.nan]*len(time)
            # 'base_v': [params['base_v'][t] for t in time]
        })


    except Exception as e:
        logger.exception(f"[工况{cond_id}] Gurobi 运行异常: {e}")
        return pd.DataFrame()


# =================== 任务处理函数 =============================
def process_task(task):
    veh_id, cond_id, df_grp = task
    params = prepare_ampl_param(df_grp, veh_id)
    return run_gurobi(params, cond_id)


#%%
# =================== 主程序 =============================
if __name__ == '__main__':

    df_clean = df_all.groupby('工况ID').filter(lambda g: (g['vehicle_speed'] != 0).any())
    df_clean = df_clean.sort_values(['工况ID', 'vehicle_id'], ascending=[False, False])

    # results = []
    # for (veh_id, cond_id), df_grp in tqdm(df_clean.groupby(['vehicle_id', '工况ID'], sort=False)):
    #     params = prepare_ampl_param(df_grp, veh_id)
    #     res = run_gurobi(params, cond_id)
    #     if not res.empty:
    #         results.append(res)

    # 构建任务列表
    tasks = [(veh_id, cond_id, df_grp) for (veh_id, cond_id), df_grp in df_clean.groupby(['vehicle_id', '工况ID'], sort=False)]
    # 多核并行执行
    # results = []
    # with ProcessPoolExecutor(max_workers=61) as executor:
    #     for df_res in tqdm(executor.map(process_task, tasks), total=len(tasks), desc="工况优化："):
    #         if not df_res.empty:
    #             results.append(df_res)

    TMPDIR = r"I:\qsm_file\跟驰参数默认\数据处理_revised\gurobi_parts（动力学约束）"  # 本地SSD临时目录
    os.makedirs(TMPDIR, exist_ok=True)

    BATCH = 500
    buf, part_id = [], 0
    with ProcessPoolExecutor(max_workers=56) as ex:
        for i, df_res in enumerate(tqdm(ex.map(process_task, tasks, chunksize=8),
                                        total=len(tasks), desc="工况优化：")):
            if df_res is not None and not df_res.empty:
                buf.append(df_res)
            if (i + 1) % BATCH == 0 and buf:
                pd.concat(buf, ignore_index=True).to_parquet(
                    os.path.join(TMPDIR, f"part_{part_id:05d}.parquet"))
                part_id += 1
                buf.clear()
                gc.collect()

    # 收尾
    if buf:
        pd.concat(buf, ignore_index=True).to_parquet(
            os.path.join(TMPDIR, f"part_{part_id:05d}.parquet"))
        buf.clear()
        gc.collect()

    parts = [os.path.join(TMPDIR, f) for f in os.listdir(TMPDIR) if f.endswith(".parquet")]
    df_results = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)

    # df_results = pd.concat(results)
    df_results = df_results.sort_values(['vehicle_id', 'timestep_time'])
    df = pd.merge(df_all, df_results, how='left', on=['vehicle_id', 'timestep_time'])

    use_cols = ['vehicle_id', 'timestep_time', '工况ID', 'vehicle_type', 'vehicle_odometer', 'vehicle_speed',  'vehicle_x', 'vehicle_y',
                'opt_jerk', 'opt_a', 'opt_v', 'opt_x']
    df_out = df[use_cols]
    df_out.to_csv(os.path.join(os.path.dirname(file_path), f'{file_name}_动力学约束0815.csv'), index=False, encoding='utf-8')
