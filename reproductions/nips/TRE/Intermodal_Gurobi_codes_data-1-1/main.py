# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 11:32:25 2019

@author: yimengzhang
"""

from gurobipy import *
import pandas as pd
import numpy as np
from collections import Counter
import copy
import re
import timeit
from time import process_time
# import gurobipy as gp
# from gurobipy import GRB
import sys


# sys.setrecursionlimit(1000000)
# @time_me()

def battery_constraints(has_end_depot, K, R, k, route, calculate_energy=0):
    """
    检查路线是否满足电池容量约束
    
    参数:
        has_end_depot: 是否有终点depot
        K: 车辆信息矩阵
        R: 请求信息矩阵
        k: 当前车辆索引
        route: 当前路线
        calculate_energy: 是否只计算能耗(1)或同时检查约束(0)
    
    返回:
        如果calculate_energy=0: 返回True(满足约束)或False(违反约束)
        如果calculate_energy=1: 返回(总能耗, 剩余电量)
    """
    global D

    charging_stations = [0, 4, 5, 10]
    
    # 根据车辆类型设置电池参数
    vehicle_type = K[k, 5]  
    
    # 电池容量 (kWh)
# 电池容量与能耗参数设置
    if vehicle_type == 1:  # eVTOL(电动垂直起降飞行器)
        battery_capacity = 200   # kWh(中型eVTOL常见范围 100–300kWh)
        alpha = 30               # kWh/h(垂直起降阶段典型能耗 ~1–3kWh/min)
        beta = 0.15              # kWh/km(水平飞行阶段约 0.15kWh/km)
        
    elif vehicle_type == 2:  # 电动出租车(EV Taxi)
        battery_capacity = 50    # kWh(主流出租车型号40–60kWh)
        alpha = 10               # kWh/h(按城市平均车速50km/h × 0.2kWh/km)
        beta = 0.2               # kWh/km(城市电动车平均能耗)

    elif vehicle_type == 3:  # 无人机(Drone/UAV)
        battery_capacity = 36   # kWh(中大型物流无人机典型电池容量)
        alpha = 18              # kWh/h(基于飞行速度50km/h × 0.05kWh/km)
        beta = 0.3              # kWh/km(中型载荷无人机典型航程能耗)

    
    # 初始化累计能耗
    total_energy = 0
    energy_list = [0]  # 记录每个节点后的累计能耗
    
    # 确定路线长度
    if has_end_depot == 1:
        length = len(route[4])
    else:
        length = len(route[4]) + 1
    
    # 遍历路线中的每一段
    for m in range(1, length):
        # 获取当前段的起点和终点
        node_i = route[0, m - 1]
        node_j = route[0, m]
        
        # 如果起点和终点相同,跳过
        if node_i == node_j:
            energy_list.append(total_energy)
            continue
        
        # 计算行驶距离
        distance = D[k][int(node_j), int(node_i)]
        
        # 如果距离不可达,返回False
        if distance >= 1000000000:
            if calculate_energy == 0:
                return False
            else:
                return float('inf'), -float('inf')
        
        # 计算行驶时间(到达时间 - 离开时间)
        travel_time = route[1, m] - route[3, m - 1]
        
        # 计算这一段的能耗: E = α × 时间 + β × 距离
        segment_energy = alpha * travel_time + beta * distance
        
        # 累加能耗
        total_energy += segment_energy
        
        # 检查是否到达充电站,如果是则电量充满
        if int(node_j) in charging_stations:
            total_energy = 0  # 电量充满,重置为0
        
        energy_list.append(total_energy)
        
        # 如果不是只计算能耗模式,检查是否超过电池容量
        if calculate_energy == 0:
            if total_energy > battery_capacity:
                return False
    
    # 如果是计算能耗模式,返回总能耗和剩余电量
    if calculate_energy == 1:
        remaining_battery = battery_capacity - total_energy
        return total_energy, remaining_battery
    
    # 通过所有检查,返回True
    return True


def read_data(K):
    global D_origin_All, D

    D_origin_barge = pd.read_excel("D_EGS - 10r.xlsx", 'Barge')
    D_origin_train = pd.read_excel("D_EGS - 10r.xlsx", 'Train')
    D_origin_truck = pd.read_excel("D_EGS - 10r.xlsx", 'Truck')
    D_origin_All = pd.read_excel("D_EGS - 10r.xlsx", 'All')

    no_route_barge = pd.read_excel("Barge_no_land.xlsx", 'Barge')
    no_route_truck = pd.read_excel("Barge_no_land.xlsx", 'Truck')
    D_origin_barge = add_dummy_D(D_origin_barge)
    D_origin_train = add_dummy_D(D_origin_train)
    D_origin_truck = add_dummy_D(D_origin_truck)
    D_origin_All = add_dummy_D(D_origin_All)

    D = {}
    for k in K.index:
        if K['c1'][k] == 0.8:
            D[k] = D_origin_barge.copy()
        else:
            if K['c1'][k] == 0.85:
                D[k] = D_origin_train.copy()
            else:
                D[k] = D_origin_truck.copy()
    return D, no_route_barge, no_route_truck, D_origin_All


def ok_distance(m, n, k_change, l_change, T_change):
    original_distance = D_origin_All[m][n]
    new_distance = D[k_change][m][T_change] + D[l_change][T_change][n]
    if new_distance > 1.3 * original_distance:
        return 0
    else:
        return 1


def add_dummy_D(D):
    D = D.set_index('N')
    xx = list(D.columns)
    xx.extend([s + '_dummy' for s in list(D.columns)])
    new_D = pd.DataFrame(columns=xx, index=xx)
    for i in xx:
        for j in xx:
            i_no_dummy = i
            j_no_dummy = j
            if '_dummy' in i:
                i_no_dummy = i[:len(i) - 6]
            if '_dummy' in j:
                j_no_dummy = j[:len(j) - 6]

            new_D[i][j] = D[i_no_dummy][j_no_dummy]
    return new_D


def GetA(N):
    A = pd.DataFrame(columns=['O', 'D'])
    for i in N['N']:

        N_delete = N.copy()
        N_delete = N_delete.drop(N_delete[N_delete['N'] == i].index.values)
        for h in N_delete['N']:
            A_add = pd.DataFrame([[i, h]], columns=['O', 'D'])
            A = A._append(A_add, ignore_index=True)

    return A


def instance(Data):
    K = pd.read_excel(Data, 'K')
    o = pd.read_excel(Data, 'o')
    K = K.set_index('K')
    o = o.set_index('K')
    R = pd.read_excel(Data, 'R_' + str(request_number))
    K_barge_train = K.iloc[0:82]
    K_truck = K.iloc[82:]
    # K_barge_train = K.iloc[1:]
    # K_truck = K.iloc[0:1]
    N = pd.read_excel(Data, 'N')
    T = pd.read_excel(Data, 'T')

    N_d = pd.read_excel(Data, 'N')

    # dummy depots
    o_dummy = copy.copy(o)
    for i in o_dummy.index:
        o_dummy['o'][i] = o_dummy['o'][i] + '_dummy'
        o_dummy['o2'][i] = o_dummy['o2'][i] + '_dummy'
    for i in o_dummy.index:
        o_dummy_N = pd.DataFrame([[o_dummy['o'][i]]], columns=['N'])
        o2_dummy_N = pd.DataFrame([[o_dummy['o2'][i]]], columns=['N'])
        # N = N.set_index('N')
        N_d = N_d._append(o_dummy_N)
        N_d = N_d._append(o2_dummy_N)
    N_d.drop_duplicates(subset="N", keep='first', inplace=True)
    N_d = N_d.set_index('N')

    N = pd.DataFrame(N_d.index)
    D, no_route_barge, no_route_truck, D_origin_All = read_data(K)

    # D_origin = GetD(N_d, N)
    A = GetA(N)

    # Only compute used terminals
    realTerminals = T['T']._append(R['p'])._append(R['d'])._append(o['o'])._append(o['o2'])._append(
        o_dummy['o'])._append(
        o_dummy['o2'])
    A = A[A['O'].isin(realTerminals)]
    A = A[A['D'].isin(realTerminals)]
    N = N[N['N'].isin(realTerminals)]

    return K, N, T, A, D, R, o, o_dummy, no_route_barge, no_route_truck, K_barge_train, K_truck


# @time_me()
# @profile()
# @jit
def ok_TK(i):
    all_ok_TK_i = {}
    for T_change in T.index:

        all_ok_TK_i_list = []
        if R['p'][i] == T_change or R['d'][i] == T_change:
            continue
        if D_origin_All[R['p'][i]][T_change] + D_origin_All[T_change][R['d'][i]] > 1.3 * D_origin_All[R['p'][i]][
            R['d'][i]]:
            continue

        for k_change in K.index:

            # if fixed k's fixed terminals not in the terminals of request and T, then not be considered. But this is only for fixed k with two terminals
            if k_change in fixed_vehicles[
                           int(percentage[0] * len(fixed_vehicles)):int(percentage[1] * len(fixed_vehicles))] and len(
                    Fixed[k_change]) == 2:
                if not (o['o'][k_change] == R.values[i, 0] and o['o2'][k_change] == T_change):
                    continue

            for l_change in K.index.drop(k_change):
                # if fixed k's fixed terminals not in the terminals of request and T, then not be considered. But this is only for fixed k with two terminals
                if l_change in fixed_vehicles[int(percentage[0] * len(fixed_vehicles)):int(
                        percentage[1] * len(fixed_vehicles))] and len(Fixed[l_change]) == 2:
                    if not (o['o'][l_change] == T_change and o['o2'][l_change] == R.values[i, 1]):
                        continue

                # if begin node of k_change very close to the pickup node and end node of l_change very clcose to delivery node, the vehicle may designed for this request, no matter how far it to the T
                # if end node of k_change begin node of l_change very close to the T terminal, it also considered
                if (D[k_change][o['o'][k_change]][R.iloc[i, 0]] <= 100 and D[k_change][o['o2'][l_change]][
                    R.iloc[i, 1]] <= 100) or (
                        D[k_change][o['o2'][k_change]][T_change] <= 100 and D[l_change][o['o'][l_change]][
                    T_change] <= 100):
                    all_ok_TK_i_list.append([k_change, l_change])

        all_ok_TK_i[T_change] = pd.DataFrame(all_ok_TK_i_list, columns=['k', 'l'])
    return all_ok_TK_i


def func_ok_K_canpickr(R, K, D, fixed_vehicles, Fixed, o_dummy, percentage):
    R_K = {}
    for k in K.index:
        R_k = []
        for r in R.index:
            R_k.append(r)
        R_K[k] = R_k

    ok_K_canpickr = pd.DataFrame(columns=R.index, index=range(len(K.index)))
    for r in R.index:
        n = 0
        for k in K.index:
            # capacity < load
            if K['u'][k] >= R['qr'][r]:
                arrive_time = D[k][o_dummy['o'][k]][R.values[r, 0]] / K['speed'][k]
                if k in fixed_vehicles[
                        int(percentage[0] * len(fixed_vehicles)):int(percentage[1] * len(fixed_vehicles))]:
                    arrive_time = arrive_time + Fixed[k].values[0, 1]
                if K['c1'][k] == 0.8 or K['c1'][k] == 0.85:
                    departure_time = arrive_time + service_time
                else:
                    departure_time = arrive_time
                if departure_time <= R.values[r, 3]:
                    if k in fixed_vehicles[
                            int(percentage[0] * len(fixed_vehicles)):int(percentage[1] * len(fixed_vehicles))]:
                        if departure_time <= Fixed[k]['bp'][0]:
                            ok_K_canpickr[r][n] = k
                            n = n + 1
                    else:
                        ok_K_canpickr[r][n] = k
                        n = n + 1
            else:
                R_K[k].remove(r)

    return ok_K_canpickr, R_K


def Extract(n, lst):
    return [item[n] for item in lst]


def preprocess(K, K_truck, R, N, A, D, T, no_route_barge, no_route_truck, o, o_dummy):
    x_list = []
    y_list = []
    s_list = []
    t_list = []
    t_list_n = []
    t_list_n_truck = []
    t_vehicle_list = []
    t_wait_list_n = []
    t_delay_list = []
    n_r_k_list = []
    xi_list = []
    tao_truck_list = []
    zeta_list = []
    eta_list = []

    # all x
    for k in K.index:
        for i, j in zip(A.O, A.D):
            x_list.append(tuple([k, i, j]))
    # all y
    for k in K.index:
        for r in R.index:
            for i, j in zip(A.O, A.D):
                y_list.append(tuple([k, r, i, j]))

    # all s
    for k in K.index:
        # not drop k because if same k transport a request but use the the T in the middle, it will no transshipment cost, and the distance AB + BC may < AC, which will not same as ALNS
        # so forbid T for one same k
        # for l in K.index.drop(k):
        for l in K.index:
            for i in N['N']:
                for r in R.index:
                    s_list.append(tuple([k, l, i, r]))
    # all t
    for k in K.index:
        for i in N['N']:
            for r in R.index:
                for n in [1, 2, 3]:
                    t_list.append(tuple([k, i, r, n]))

    # all t without n
    for k in K.index:
        for i in N['N']:
            for r in R.index:
                t_list_n.append(tuple([k, i, r]))
    for k in K.index:
        for i in N['N']:
            for n in [1, 2, 3]:
                t_vehicle_list.append(tuple([k, i, n]))
    for k in K.index:
        for i in N['N']:
            t_wait_list_n.append(tuple([k, i]))
    # all t_delay
    for r in R.index:
        t_delay_list.append(tuple([r]))

    # fixed
    fixed_data_path = 'Fixed_right_real.xlsx'
    Fixed_Data = pd.ExcelFile(fixed_data_path)
    Fixed = pd.read_excel(Fixed_Data, None)
    fixed_vehicles = Fixed['FixedK']['FixedK'].tolist()

    # percentage of flexible vehicles, from the first one to the percentage one
    percentage = [0, 0.3]

    for k in fixed_vehicles[int(percentage[0] * len(fixed_vehicles)):int(percentage[1] * len(fixed_vehicles))]:

        A_not_allow = copy.copy(A)
        fixed_pair = []
        for fixed_pair_i in range(len(Fixed[k]['p']) - 1):
            fixed_pair.append([Fixed[k]['p'][fixed_pair_i], Fixed[k]['p'][fixed_pair_i + 1]])
            fixed_pair.append([Fixed[k]['p'][fixed_pair_i] + '_dummy', Fixed[k]['p'][fixed_pair_i + 1] + '_dummy'])
            fixed_pair.append([Fixed[k]['p'][fixed_pair_i], Fixed[k]['p'][fixed_pair_i + 1] + '_dummy'])
            fixed_pair.append([Fixed[k]['p'][fixed_pair_i] + '_dummy', Fixed[k]['p'][fixed_pair_i + 1]])
            fixed_pair.append([Fixed[k]['p'][fixed_pair_i] + '_dummy', Fixed[k]['p'][fixed_pair_i]])
            fixed_pair.append([Fixed[k]['p'][fixed_pair_i], Fixed[k]['p'][fixed_pair_i] + '_dummy'])
            fixed_pair.append([Fixed[k]['p'][fixed_pair_i + 1], Fixed[k]['p'][fixed_pair_i + 1] + '_dummy'])
            fixed_pair.append([Fixed[k]['p'][fixed_pair_i + 1] + '_dummy', Fixed[k]['p'][fixed_pair_i + 1]])

        # fixed_pair_df = pd.DataFrame(fixed_pair, columns=['O','D'])
        for n in A_not_allow.index:
            i = A_not_allow['O'][n]
            j = A_not_allow['D'][n]
            if [i, j] in fixed_pair:
                A_not_allow.drop(n, axis=0, inplace=True)
        # A_not_allow.loc[~(A_not_allow.isin(fixed_pair_df)),:]
        for i, j in zip(A_not_allow.O, A_not_allow.D):
            try:
                x_list.remove(tuple([k, i, j]))
                for r in R.index:
                    y_list.remove(tuple([k, r, i, j]))
            except:
                continue
    # for free k, it can't go to other k's dummy depot;
    #            if dummy in x, it must be together with depot
    else:
        x_list_copy = copy.copy(x_list)
        y_list_copy = copy.copy(y_list)
        for x in x_list:
            k = x[0]
            i = x[1]
            j = x[2]
            if (i[-6:] == '_dummy' and i != o_dummy['o'][k]) or (j[-6:] == '_dummy' and j != o_dummy['o2'][k]) \
                    or (i == o_dummy['o'][k] and j != o['o'][k]) \
                    or (j == o_dummy['o2'][k] and i != o['o2'][k]):

                x_list_copy.remove(x)
                for r in R.index:
                    try:
                        y_list_copy.remove(tuple([k, r, i, j]))
                    except:
                        continue
        x_list = copy.copy(x_list_copy)
        y_list = copy.copy(y_list_copy)
    # barge can't on land
    x_list_copy = copy.copy(x_list)
    for x in x_list:
        k = x[0]
        if k not in fixed_vehicles[int(percentage[0] * len(fixed_vehicles)):int(percentage[1] * len(fixed_vehicles))]:
            if (K['c1'][k] == 0.8 and tuple([x[1], x[2]]) in zip(no_route_barge.O, no_route_barge.D)) or (
                    K['c1'][k] == 0.75 and tuple([x[1], x[2]]) in zip(no_route_truck.O, no_route_truck.D)):
                x_list_copy.remove(x)
    x_list = copy.copy(x_list_copy)

    y_list_copy = copy.copy(y_list)
    for y in y_list:
        k, r, i, j = y
        # if contains dummy, then not in y
        if i[-6:] == '_dummy' or j[-6:] == '_dummy':
            y_list_copy.remove(y)
            continue
        if k not in fixed_vehicles[int(percentage[0] * len(fixed_vehicles)):int(percentage[1] * len(fixed_vehicles))]:
            if (K['c1'][k] == 0.8 and tuple([i, j]) in zip(no_route_barge.O, no_route_barge.D)) or (
                    K['c1'][k] == 0.75 and tuple([i, j]) in zip(no_route_truck.O, no_route_truck.D)):
                y_list_copy.remove(y)
    y_list = copy.copy(y_list_copy)

    # dummy_begin can't be j, dummy_end can't be i
    # real_begin can't be j, real_end can't be i when it not with dummy
    x_list_copy = copy.copy(x_list)
    y_list_copy = copy.copy(y_list)
    for x in x_list:
        k = x[0]
        i = x[1]
        j = x[2]
        if j == o_dummy['o'][k] or i == o_dummy['o2'][k] \
                or (j == o['o'][k] and i[-6:] != '_dummy') or (i == o['o2'][k] and j[-6:] != '_dummy'):
            x_list_copy.remove(x)
            for r in R.index:
                try:
                    y_list_copy.remove(tuple([k, r, i, j]))
                except:
                    continue
    x_list = copy.copy(x_list_copy)
    y_list = copy.copy(y_list_copy)

    # capacity < load
    y_list_copy = copy.copy(y_list)
    for y in y_list:
        if K['u'][y[0]] < R['qr'][y[1]]:
            y_list_copy.remove(y)
    y_list = copy.copy(y_list_copy)

    z_list = copy.copy(x_list)

    ok_K_canpickr, R_K = func_ok_K_canpickr(R, K, D, fixed_vehicles, Fixed, o_dummy, percentage)
    # Vehicle $k$ can not pickup request $r$ when $k$ not in $K_r^{p}$ (when it is real pickup (not Tp or secondTp))
    s_list_copy = copy.copy(s_list)
    for s in s_list:
        if s[0] not in list(ok_K_canpickr[s[3]]):
            s_list_copy.remove(s)
    s_list = copy.copy(s_list_copy)

    # s not in y
    s_list_copy = copy.copy(s_list)
    for s in s_list:
        k = 0
        l = 0
        break_for = 0
        for y in y_list:
            if s[0] == y[0] and s[2] == y[3] and s[3] == y[1]:
                k = 1
            if s[1] == y[0] and s[2] == y[2] and s[3] == y[1]:
                l = 1
            if k == 1 and l == 1:
                break_for = 1
                break
        if break_for == 0:
            s_list_copy.remove(s)
    s_list = copy.copy(s_list_copy)

    # in s, k's begin depot and l's end depot can't be i

    # T which are too far away p,d
    s_list_copy = copy.copy(s_list)
    for s in s_list:
        k = s[0]
        l = s[1]
        i = s[2]
        r = s[3]
        if o['o'][k] == i or o['o'][k] + '_dummy' == i or o['o2'][l] == i or o['o2'][l] + '_dummy' == i \
                or ok_distance(R['p'][r], R['d'][r], k, l, i) == 0:
            s_list_copy.remove(s)
    s_list = copy.copy(s_list_copy)

    t_list_copy = copy.copy(t_list)
    for t in t_list:
        break_for = 0
        for x in x_list:
            if t[0] == x[0] and (t[1] == x[1] or t[1] == x[2]):
                break_for = 1
                break
        if break_for == 0:
            t_list_copy.remove(t)
    t_list = copy.copy(t_list_copy)

    t_list_n_copy = copy.copy(t_list_n)
    for t in t_list_n:
        break_for = 0
        for x in x_list:
            if t[0] == x[0] and (t[1] == x[1] or t[1] == x[2]):
                break_for = 1
                break
        if break_for == 0:
            t_list_n_copy.remove(t)
    t_list_n = copy.copy(t_list_n_copy)

    t_wait_list_n_copy = copy.copy(t_wait_list_n)
    for k, i in t_wait_list_n:
        break_or_not = 0
        for k1, i1, r in t_list_n:
            if k1 == k and i1 == i:
                break_or_not = 1
                break
        if break_or_not == 0:
            t_wait_list_n_copy.remove(tuple([k, i]))
    t_wait_list_n = copy.copy(t_wait_list_n_copy)

    t_wait_list_n_copy = copy.copy(t_wait_list_n)
    for k, i in t_wait_list_n:
        b = 0
        for k1, r, i1, j in y_list:
            if k == k1 and i == i1:
                b = 1
                break
        if b != 1:
            t_wait_list_n_copy.remove(tuple([k, i]))
    t_wait_list_n = copy.copy(t_wait_list_n_copy)

    # p and d can't be combined together, because if k transport r from i to j directly, y_list_p_d will has only one record, and the un_load_cost will be wrong
    y_list_p = []
    y_list_d = []
    for y in y_list:
        if y[2] in list(R['p']) and y[1] in R.index[R['p'] == y[2]].tolist():
            y_list_p.append(y)
        if y[3] in list(R['d']) and y[1] in R.index[R['d'] == y[3]].tolist():
            y_list_d.append(y)

    y_list_d_r = []
    for y in y_list:
        r = y[1]
        if y[3] == R['d'][r]:
            y_list_d_r.append(y)

    # delete k which not be used
    K_x = set(Extract(0, x_list))
    for k in K.index:
        if k not in K_x:
            K.drop(k, axis=0, inplace=True)

    # for each k, obtain it's A/N denpending on x
    A_K = {}
    N_K = {}
    A_K_y = {}
    for k in K.index:
        A_k = []
        N_k = []
        for x in x_list:
            if k == x[0]:
                A_k.append([x[1], x[2]])
                N_k.append(x[1])
                N_k.append(x[2])
        A_K[k] = pd.DataFrame(A_k, columns=['O', 'D'])
        N_k = set(N_k)
        N_K[k] = pd.DataFrame(N_k, columns=['N'])

    for k in K.index:
        A_k_y = copy.deepcopy(A_K[k])
        A_k_y_copy = copy.deepcopy(A_K[k])
        for index in range(len(A_k_y)):
            if A_k_y['O'][index][-6:] == '_dummy' or A_k_y['D'][index][-6:] == '_dummy':
                A_k_y_copy = A_k_y_copy.drop([index])
        A_K_y[k] = copy.copy(A_k_y_copy)

    T_K = {}
    for k in K.index:
        T_k = []
        for s in s_list:
            # doubt it
            if (k == s[0] or k == s[1]) and s[2] in list(T['T']):
                T_k.append(s[2])
        T_k = set(T_k)
        T_K[k] = pd.DataFrame(T_k, columns=['T'])
    for k, i, r in t_list_n:
        if k in K_truck.index:
            t_list_n_truck.append(tuple([k, i, r]))
    t_b, b_list, m_list = {}, [], []
    if time_dependent_traveltime == 1:
        t_b = {1: 0, 2: 5, 3: 7, 4: 9, 5: 13, 6: 13, 7: 17, 8: 19, 9: 21, 10: 24}
        # b_list is k in Wenjing's thesis
        b_list = range(1, len(t_b) + 1)
        # m is time perior, such as [b1,b2]
        m_list = range(1, len(t_b))
        for r in R.index:
            for k in K_truck.index:
                for i in N['N']:
                    n_r_k_list.append(tuple([k, i, r]))
        for k, r, i, j in y_list:
            if k in K_truck.index:
                tao_truck_list.append(tuple([k, r, i, j]))
        for b_i in b_list:
            for k, i, r in t_list_n_truck:
                zeta_list.append(tuple([b_i, k, i, r]))
        for m_i in m_list:
            for k, i, r in t_list_n_truck:
                xi_list.append(tuple([m_i, k, i, r]))

        eta_list = [1, 1, alpha, alpha, belta, belta, alpha, alpha, 1, 1]
    return x_list, y_list, y_list_p, y_list_d, z_list, s_list, t_list, t_list_n, t_vehicle_list, t_wait_list_n, t_delay_list, t_list_n_truck, y_list_d_r, K, A_K, N_K, A_K_y, T_K, ok_K_canpickr, R_K, n_r_k_list, tao_truck_list, zeta_list, xi_list, t_b, b_list, m_list, eta_list


def optimization(K, N, T, A, D, R, o, o_dummy, no_route_barge, no_route_truck, K_barge_train, K_truck):
    # 记录初始化开始时间
    initial_start_time = timeit.default_timer()
    initial_start_cpu = process_time()

    un_load_cost_table = pd.DataFrame(columns=['un_load_cost'], index=K.index)
    for k in K.index:
        if K['c1'][k] == 0.8 or K['c1'][k] == 0.85:
            un_load_cost_table['un_load_cost'][k] = un_load_cost_barge_train
        else:
            un_load_cost_table['un_load_cost'][k] = un_load_cost_truck

    emission_table = pd.DataFrame(columns=['emission'], index=K.index)
    for k in K.index:
        if K['c1'][k] == 0.8:
            emission_table['emission'][k] = 0.8  # c4
        else:
            if K['c1'][k] == 0.85:
                emission_table['emission'][k] = 0.3146
            else:
                emission_table['emission'][k] = 0.15  #

    c_delay_table = pd.DataFrame(index=R.index, columns=['c_delay'])
    for r in R.index:
        if R['bd'][r] - R['ap'][r] < 30:
            c_delay_table['c_delay'][r] = 100
        else:
            if R['bd'][r] - R['ap'][r] < 54:
                c_delay_table['c_delay'][r] = 70
            else:
                c_delay_table['c_delay'][r] = 50
    start_preprocess = timeit.default_timer()
    x_list, y_list, y_list_p, y_list_d, z_list, s_list, t_list, t_list_n, t_vehicle_list, t_wait_list_n, t_delay_list, t_list_n_truck, y_list_d_r, K, A_K, N_K, A_K_y, T_K, ok_K_canpickr, R_K, n_r_k_list, tao_truck_list, zeta_list, xi_list, t_b, b_list, m_list, eta_list = preprocess(
        K, K_truck, R, N, A, D, T, no_route_barge, no_route_truck, o, o_dummy)
    time_preprocess = timeit.default_timer() - start_preprocess
    print('time_preprocess: ', time_preprocess)
    m = Model("Intermodal")

    # varibles
    x = m.addVars(x_list, vtype=GRB.BINARY, name="x")
    y = m.addVars(y_list, vtype=GRB.BINARY, name="y")
    z = m.addVars(z_list, vtype=GRB.BINARY, name="z")
    s = m.addVars(s_list, vtype=GRB.BINARY, name="s")
    t = m.addVars(t_list, vtype=GRB.CONTINUOUS, lb=0, name="t")
    t_vehicle = m.addVars(t_vehicle_list, vtype=GRB.CONTINUOUS, lb=0, name="t_vehicle")
    t_wait_time = m.addVars(t_wait_list_n, vtype=GRB.CONTINUOUS, lb=0, name="t_wait_time")
    t_delay = m.addVars(t_delay_list, vtype=GRB.CONTINUOUS, lb=0, name="t_delay")

    if time_dependent_traveltime == 1:
        tao_truck = m.addVars(tao_truck_list, vtype=GRB.CONTINUOUS, lb=0, name="tao_truck")
        n = m.addVars(n_r_k_list, vtype=GRB.INTEGER, name="n")
        xi = m.addVars(xi_list, vtype=GRB.BINARY, name="xi")
        zeta = m.addVars(zeta_list, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="zeta")
        t_p = m.addVars(t_list_n_truck, vtype=GRB.CONTINUOUS, lb=0, name="t_p")

    m.setObjective((sum(
        (K['c1p'][k] * D[k][i][j] + K['c1'][k] * (D[k][i][j] / K['speed'][k])) * y[k, r, i, j] * R['qr'][r] for
        # 1.每公里的距离成本
        k, r, i, j in y_list) + \
                    sum(t_wait_time[k, i] for k, i in t_wait_list_n) + \
                    sum(un_load_cost_table['un_load_cost'][k] * R['qr'][r] * y[k, r, i, j] for k, r, i, j in y_list_p) + \
                    sum(un_load_cost_table['un_load_cost'][k] * R['qr'][r] * y[k, r, i, j] for k, r, i, j in y_list_d) + \
                    sum((un_load_cost_table['un_load_cost'][k] + un_load_cost_table['un_load_cost'][l]) * R['qr'][r] *
                        s[k, l, i, r] for k, l, i, r in s_list) + \
                    sum(c_storage * R['qr'][r] * (t[l, i, r, 2] - t[k, i, r, 3]) * s[k, l, i, r] for k, l, i, r in
                        s_list) + \
                    sum(c_storage * R['qr'][r] * (t[k, i, r, 2] - R['ap'][r]) * y[k, r, i, j] for k, r, i, j in
                        y_list_p) + \
                    sum(c_delay_table['c_delay'][r] * t_delay[r,] * R['qr'][r] for r in R.index) + \
                    sum(carbon_tax * D[k][i][j] * R['qr'][r] * emission_table['emission'][k] / 1000 * y[k, r, i, j] for
                        k, r, i, j in y_list)), GRB.MINIMIZE)

    # constraints
    # if vehicle is used, only choose one route begin from begin depot
    for k in K_barge_train.index:
        i = o_dummy['o'][k]
        m.addConstr(sum(x[k, i, j] for j in A_K[k][A_K[k]['O'] == i]['D']) <= 1)

        for r in R.index:
            m.addConstr(t[k, i, r, 1] == t[k, i, r, 2])
            m.addConstr(t[k, i, r, 1] == t[k, i, r, 3])

    # begin = end
    for k in K_barge_train.index:
        i = o_dummy['o'][k]
        l = o_dummy['o2'][k]
        m.addConstr(sum(x[k, i, j] for j in A_K[k][A_K[k]['O'] == i]['D']) == sum(
            x[k, j, l] for j in A_K[k][A_K[k]['D'] == l]['O']))

    # all requests must be served
    for r in R.index:
        i = R['p'][r]

        m.addConstr(
            sum(y[k, r, i, j] for k in ok_K_canpickr[r].dropna() for j in A_K_y[k][A_K_y[k]['O'] == i]['D']) == 1)

    for r in R.index:
        i = R['d'][r]

        m.addConstr(sum(y[k, r, j, i] for k in ok_K_canpickr[r].dropna() for j in A_K[k][A_K[k]['D'] == i]['O']) == 1)

    # vehilce flow conservation
    for k in K_barge_train.index:
        N_drop = copy.deepcopy(N_K[k])
        N_drop = N_drop.set_index('N')

        print(f"k: {k}")
        print(f"o_dummy['o'][k]: {o_dummy['o'][k]}")
        print(f"o_dummy['o2'][k]: {o_dummy['o2'][k]}")
        print(f"N_drop.index: {N_drop.index.tolist()}")

        N_drop = N_drop.drop([o_dummy['o'][k], o_dummy['o2'][k]], errors='ignore')
        for i in N_drop.index:
            m.addConstr(sum(x[k, i, j] for j in A_K[k][A_K[k]['O'] == i]['D']) - sum(
                x[k, j, i] for j in A_K[k][A_K[k]['D'] == i]['O']) == 0)

    # request flow conservation at transshipment terminals
    for r in R.index:
        T_drop = copy.copy(T)
        T_drop = T_drop.set_index('T')
        if R['p'][r] in T_drop.index:
            T_drop = T_drop.drop(R['p'][r])
        if R['d'][r] in T_drop.index:
            T_drop = T_drop.drop(R['d'][r])
        T_drop = T_drop.reset_index()
        for i in T_drop['T']:
            m.addConstr(
                sum(y[k, r, i, j] for k in ok_K_canpickr[r].dropna() for j in A_K_y[k][A_K_y[k]['O'] == i]['D']) - sum(
                    y[k, r, j, i] for k in ok_K_canpickr[r].dropna() for j in A_K_y[k][A_K_y[k]['D'] == i]['O']) == 0)

    # request flow conservation
    for r in R.index:
        for k in ok_K_canpickr[r].dropna():
            N_drop = copy.deepcopy(N_K[k])
            N_drop = N_drop.set_index('N')
            if R['p'][r] in N_drop.index:
                N_drop = N_drop.drop(R['p'][r])
            if R['d'][r] in N_drop.index:
                N_drop = N_drop.drop(R['d'][r])

            for h in T_K[k]['T']:
                if h in N_drop.index:
                    N_drop = N_drop.drop(h)

            for i in N_drop.index:
                m.addConstr(sum(y[k, r, i, j] for j in A_K_y[k][A_K_y[k]['O'] == i]['D']) - sum(
                    y[k, r, j, i] for j in A_K_y[k][A_K_y[k]['D'] == i]['O']) == 0)
            # only when T is used, then the request flow constraints not work;
            # if use T, but not transshipment, request flow constraints still work
            T_drop = copy.deepcopy(T_K[k])
            T_drop = T_drop.set_index('T')
            if R['p'][r] in T_drop.index:
                T_drop = T_drop.drop(R['p'][r])
            if R['d'][r] in T_drop.index:
                T_drop = T_drop.drop(R['d'][r])
            for i in T_drop.index:

                s_1, s_2 = [], []
                for s_ in s_list:
                    if s_[1] == k and s_[2] == i and s_[3] == r:
                        s_1.append(s_)
                    if s_[0] == k and s_[2] == i and s_[3] == r:
                        s_2.append(s_)
                s_1 = pd.DataFrame(s_1)
                s_2 = pd.DataFrame(s_2)
                if not s_1.empty:
                    m.addConstr(sum(y[k, r, i, j] for j in A_K_y[k][A_K_y[k]['O'] == i]['D']) - sum(
                        y[k, r, j, i] for j in A_K_y[k][A_K_y[k]['D'] == i]['O']) <= sum(s[l, k, i, r] for l in s_1[0]))
                if not s_2.empty:
                    m.addConstr(sum(y[k, r, j, i] for j in A_K_y[k][A_K_y[k]['D'] == i]['O']) - sum(
                        y[k, r, i, j] for j in A_K_y[k][A_K_y[k]['O'] == i]['D']) <= sum(s[k, l, i, r] for l in s_2[1]))
                if s_1.empty and s_2.empty:
                    m.addConstr(sum(y[k, r, i, j] for j in A_K_y[k][A_K_y[k]['O'] == i]['D']) - sum(
                        y[k, r, j, i] for j in A_K_y[k][A_K_y[k]['D'] == i]['O']) == 0)
    # vehicle flow cover request flow
    for k, r, i, j in y_list:
        m.addConstr(y[k, r, i, j] <= x[k, i, j])

    # transshipment occurs only once in the transshipment terminal
    # a request can be transfered only when transhipment happens
    for k, l, i, r in s_list:
        # if both k and l are truck, then no -> Danger! it may lose some potential solutions
        if i not in list(T['T']):
            m.addConstr(s[k, l, i, r] == 0)
        if (K['type'][k] == 1 and K['type'][l] == 1):
            m.addConstr(s[k, l, i, r] == 0)
        # forbid T for the same k
        if k == l:
            m.addConstr(s[k, l, i, r] == 0)
        try:
            if (not A_K_y[k][A_K_y[k]['D'] == i]['O'].empty) and (not A_K_y[l][A_K_y[l]['O'] == i]['D'].empty):
                m.addConstr(sum(y[k, r, j, i] for j in A_K_y[k][A_K_y[k]['D'] == i]['O']) + sum(
                    y[l, r, i, j] for j in A_K_y[l][A_K_y[l]['O'] == i]['D']) <= s[k, l, i, r] + 1)

            if not A_K_y[k][A_K_y[k]['D'] == i]['O'].empty:
                m.addConstr(s[k, l, i, r] <= sum(y[k, r, j, i] for j in A_K_y[k][A_K_y[k]['D'] == i]['O']))
            if not A_K_y[k][A_K_y[k]['O'] == i]['D'].empty:
                m.addConstr(s[k, l, i, r] <= sum(y[l, r, i, j] for j in A_K_y[k][A_K_y[k]['O'] == i]['D']))
        except:
            pass

    # t2>=t
    for k, i, r in t_list_n:
        m.addConstr(t[k, i, r, 1] <= t[k, i, r, 2])

    for k, i, r in t_list_n:
        break_ = 0
        for j in A_K[k][A_K[k]['O'] == i]['D']:
            if (k, r, i, j) in y_list:
                break_ = 1
                break
        if break_ == 0:
            continue
        if K['c1'][k] == 0.8 or K['c1'][k] == 0.85:  #
            m.addConstr(t[k, i, r, 3] >= t[k, i, r, 2] + service_time * sum(
                y[k, r, i, j] for j in A_K_y[k][A_K_y[k]['O'] == i]['D']))
        else:
            m.addConstr(t[k, i, r, 2] == t[k, i, r, 3])

    # time on arc
    for k, r, i, j in y_list:
        if k not in K_truck.index or time_dependent_traveltime == 0:
            if D[k][i][j] == 1000000000:
                m.addConstr(x[k, i, j] == 0)
            else:
                m.addConstr(t_vehicle[k, i, 3] >= t[k, i, r, 3] * y[k, r, i, j])

    for k, i, j in x_list:
        if k not in K_truck.index or time_dependent_traveltime == 0:
            m.addConstr(
                t_vehicle[k, i, 3] + D[k][i][j] / K['speed'][k] - t_vehicle[k, j, 1] <= 16000 * (1 - x[k, i, j]))
            m.addConstr(
                t_vehicle[k, i, 3] + D[k][i][j] / K['speed'][k] - t_vehicle[k, j, 1] >= - 16000 * (1 - x[k, i, j]))

    if time_dependent_traveltime == 1:
        for k, r, i, j in tao_truck_list:
            m.addConstr((t[k, j, r, 1] - t[k, i, r, 3]) * y[k, r, i, j] == tao_truck[k, r, i, j])

    # departure time in the time window of pick up terminal
    for k, r, i, j in y_list_p:
        m.addConstr(t[k, i, r, 2] >= R['ap'][r] * y[k, r, i, j])
        m.addConstr(t[k, i, r, 3] <= R['bp'][r] * (y[k, r, i, j] + 16000 * (1 - y[k, r, i, j])))

    # departure time of vehicle k is less than departure time of vehicle l when transhipment happens between k and l at terminal i
    for k, l, i, r in s_list:
        m.addConstr(t[k, i, r, 3] - t[l, i, r, 2] <= 100000 * (1 - s[k, l, i, r]))  # 转运时间窗口

    for k, i, j in x_list:
        m.addConstr(z[k, i, j] >= x[k, i, j])

    for k in K.index:
        for i, j in zip(A_K[k].O, A_K[k].D):

            try:
                m.addConstr(z[k, i, j] + z[k, j, i] == 1)
            except:
                continue

    for k in K.index:
        for i in N_K[k]['N']:
            for j in N_K[k]['N']:
                for l in N_K[k]['N']:

                    if i != j and i != l and j != l:
                        try:
                            m.addConstr(z[k, i, j] + z[k, j, l] + z[k, l, i] <= 2)
                        except:
                            continue
    # capacity constraint
    for k in K.index:
        for i, j in zip(A_K_y[k].O, A_K_y[k].D):
            m.addConstr(sum(R['qr'][r] * y[k, r, i, j] for r in R_K[k]) <= K['u'][k] * x[k, i, j])

    # battery capacity constraint (电量约束)
    # 定义充电站节点
    charging_stations = [0, 4, 5, 10]
    
    # 为每个车辆添加电池能耗变量
    battery_energy = m.addVars([(k, i) for k in K.index for i in N_K[k]['N']], 
                                vtype=GRB.CONTINUOUS, lb=0, name="battery_energy")
    
    # 为每个车辆类型设置电池参数
    battery_params = {}
    for k in K.index:
        vehicle_type = K.loc[k, 'type'] if 'type' in K.columns else 1  # 默认类型1
        
        if vehicle_type == 1:  # eVTOL
            battery_capacity = 200
            alpha = 30
            beta = 0.15
        elif vehicle_type == 2:  # 电动出租车
            battery_capacity = 50
            alpha = 10
            beta = 0.2
        elif vehicle_type == 3:  # 无人机
            battery_capacity = 36
            alpha = 18
            beta = 0.3
        else:  # 默认参数
            battery_capacity = 100
            alpha = 15
            beta = 0.2
            
        battery_params[k] = {'capacity': battery_capacity, 'alpha': alpha, 'beta': beta}
    
    # 添加电池能耗累积约束
    for k in K.index:
        params = battery_params[k]
        
        # 起始depot的能耗为0
        if o_dummy['o'][k] in N_K[k]['N'].values:
            m.addConstr(battery_energy[k, o_dummy['o'][k]] == 0)
        
        # 对于每条弧(i,j)，计算能耗累积
        for i, j in zip(A_K[k].O, A_K[k].D):
            if i in charging_stations:
                # 如果从充电站出发，能耗重置为这段弧的能耗
                if D[k][i][j] < 1000000000:  # 确保路径可达
                    travel_time = D[k][i][j] / K['speed'][k]
                    segment_energy = params['alpha'] * travel_time + params['beta'] * D[k][i][j]
                    m.addConstr(battery_energy[k, j] >= segment_energy * x[k, i, j] - 
                               10000 * (1 - x[k, i, j]))
            else:
                # 正常累积能耗
                if D[k][i][j] < 1000000000:
                    travel_time = D[k][i][j] / K['speed'][k]
                    segment_energy = params['alpha'] * travel_time + params['beta'] * D[k][i][j]
                    m.addConstr(battery_energy[k, j] >= battery_energy[k, i] + segment_energy - 
                               10000 * (1 - x[k, i, j]))
            
            # 能耗不能超过电池容量（当这条弧被使用时）
            if j not in charging_stations:
                m.addConstr(battery_energy[k, j] <= params['capacity'] + 
                           10000 * (1 - x[k, i, j]))

    fixed_data_path = 'Fixed_right_real.xlsx'
    Fixed_Data = pd.ExcelFile(fixed_data_path)
    Fixed = pd.read_excel(Fixed_Data, None)
    fixed_vehicles = Fixed['FixedK']['FixedK'].tolist()

    # percentage of flexible vehicles, from the first one to the percentage one
    percentage = [0, 0]

    # fixed routes
    for k, r, i, j in y_list:
        if k in fixed_vehicles[int(percentage[0] * len(fixed_vehicles)):int(percentage[1] * len(fixed_vehicles))] and (
                i[-6:] == '_dummy' or i in list(Fixed[k]['p'])):
            if i[-6:] == '_dummy':
                i_real = i[:-6]
            else:
                i_real = i
            index = Fixed[k].index[Fixed[k]['p'] == i_real].tolist()[0]
            m.addConstr(t[k, i, r, 1] >= Fixed[k]['ap'][index] * y[k, r, i, j])

            m.addConstr(t[k, i, r, 3] <= Fixed[k]['bp'][index] * (y[k, r, i, j] + 16000 * (1 - y[k, r, i, j])))

    for k, i, r in t_list_n:
        if i[-6:] == '_dummy':
            continue

        # truck doesn't has wait time because it can go to the request in the most approriate way
        # but if time_dependent_traveltime is not used, these constraints need to be added to calculate vehicle time, and then get travel time
        if k in K_barge_train.index or time_dependent_traveltime == 0:
            if (k, i) not in t_wait_list_n:
                continue
            m.addConstr(t_vehicle[k, i, 1] <= t[k, i, r, 1])
            m.addConstr(t_vehicle[k, i, 2] >= t[k, i, r, 2])
            m.addConstr(t_wait_time[k, i] >= (t_vehicle[k, i, 2] - t_vehicle[k, i, 1]))

    for r in R.index:

        for k in ok_K_canpickr[r].dropna():
            j = R['d'][r]
            try:
                m.addConstr(t_delay[r,] >= (t[k, j, r, 3] - R['bd'][r]) * sum(
                    y[k, r, i, j] for i in A_K_y[k][A_K_y[k]['D'] == j]['O']))
            except:
                continue

    if time_dependent_traveltime == 1:
        for k, i, r in t_list_n_truck:

            m.addConstr(t_p[k, i, r] == t[k, i, r, 3] - 24 * n[k, i, r])
            m.addConstr(t_p[k, i, r] == sum(zeta[b_i, k, i, r] * t_b[b_i] for b_i in b_list))
            m.addConstr(sum(zeta[b_i, k, i, r] for b_i in b_list) == 1)
            m.addConstr(sum(xi[m_i, k, i, r] for m_i in m_list) == 1)
            m.addConstr(zeta[1, k, i, r] <= xi[1, k, i, r])
            m.addConstr(zeta[len(t_b), k, i, r] <= xi[len(t_b) - 1, k, i, r])
            for b_i in range(2, len(t_b)):
                m.addConstr(zeta[b_i, k, i, r] <= xi[b_i - 1, k, i, r] + xi[b_i, k, i, r])
        for k, r, i, j in tao_truck_list:
            m.addConstr(tao_truck[k, r, i, j] == sum(
                zeta[b_i, k, i, r] * eta_list[b_i - 1] * D[k][i][j] / K['speed'][k] for b_i in b_list) * y[k, r, i, j])

    m.setParam('OutputFlag', True)  # silencing gurobi output
    m.Params.timeLimit = 10800  # 设置为1个小时

    m.write('workforce1.lp')

    # 记录初始化结束时间
    initial_time = timeit.default_timer() - initial_start_time
    initial_cpu = process_time() - initial_start_cpu

    start_time = process_time()
    start = timeit.default_timer()


    # Optimize model
    m.optimize()


    # 记录总运行时间
    total_time = timeit.default_timer() - start
    cpu_time = process_time() - start_time

    # 获取找到最优解的时间
    best_time = m.Runtime

    # 计算包含初始化时间的总时间
    add_initial_best_time = best_time + initial_time
    add_initial_total_time = total_time + initial_time

    print("\n时间统计:")
    print(f"初始化时间: {initial_time:.2f} 秒")
    print(f"找到最优解时间(不含初始化): {best_time:.2f} 秒")
    print(f"总运行时间(不含初始化): {total_time:.2f} 秒")
    print(f"找到最优解时间(含初始化): {add_initial_best_time:.2f} 秒")
    print(f"总运行时间(含初始化): {add_initial_total_time:.2f} 秒")
    print(f"初始目标函数值: {m.ObjVal:.2f}")

    # if find the voilated constraint, the following should be comment
    # m.write('C:/Users/yimengzhang/OneDrive/Figures/Gurobi_outputs/'+str(request_number)+'R'+K_T_number_exp+"out.mst")
    # m.write('C:/Users/yimengzhang/OneDrive/Figures/Gurobi_outputs/'+str(request_number)+'R'+K_T_number_exp+"out.sol")
    # #find the voilated constraint
    # status = m.status
    # if status == GRB.UNBOUNDED:
    #     print('The model cannot be solved because it is unbounded')
    #     sys.exit(0)
    # if status == GRB.OPTIMAL:
    #     print('The optimal objective is %g' % m.objVal)
    #     sys.exit(0)
    # if status != GRB.INF_OR_UNBD and status != GRB.INFEASIBLE:
    #     print('Optimization was stopped with status %d' % status)
    #     sys.exit(0)
    #
    # # do IIS
    # print('The model is infeasible; computing IIS')
    # m.computeIIS()
    # if m.IISMinimal:
    #     print('IIS is minimal\n')
    # else:
    #     print('IIS is not minimal\n')
    # print('\nThe following constraint(s) cannot be satisfied:')
    # for c in m.getConstrs():
    #     if c.IISConstr:
    #         print('%s' % c.constrName)

    # print request flow
    for r in R.index:
        for k in ok_K_canpickr[r].dropna():
            for i, j in zip(A_K[k].O, A_K[k].D):
                try:
                    if y[k, r, i, j].x > 0.1:
                        print(y[k, r, i, j], t_vehicle[k, i, 1], t_vehicle[k, i, 2], t_vehicle[k, i, 3], t[k, i, r, 1],
                              t[k, i, r, 2], t[k, i, r, 3], t[k, j, r, 1])
                except:
                    continue
    # print transhipment
    for r in R.index:
        if len(ok_K_canpickr[r].dropna()) > 1:
            for k in ok_K_canpickr[r].dropna():
                a = list(ok_K_canpickr[r].dropna())
                # a.remove(k)
                for l in a:
                    for i in T_K[k]['T']:
                        try:
                            if s[k, l, i, r].x > 0.1:
                                print(s[k, l, i, r])
                        except:
                            continue
    # print vehicle flow
    for k in K.index:
        for i, j in zip(A_K[k].O, A_K[k].D):
            try:
                if x[k, i, j].x > 0.1:
                    print(x[k, i, j], t[k, i, r, 1], t[k, i, r, 2], t[k, i, r, 3], t[k, j, r, 1], t_vehicle[k, i, 2],
                          t_vehicle[k, i, 1])
            except:
                continue

    # print feasible solutions in current solution pool
    for i in range(0, m.SolCount):
        m.Params.SolutionNumber = i
        print(m.PoolObjVal)

    obj = m.objVal

    request_cost = 0
    for k, r, i, j in y_list:
        request_cost = request_cost + (K['c1p'][k] * D[k][i][j] + K['c1'][k] * D[k][i][j] / K['speed'][k]) * y[
            k, r, i, j].x * R['qr'][r]

    wait_cost = 0

    for k, i in t_wait_list_n:
        wait_cost = wait_cost + t_wait_time[k, i].x

    un_load_cost = 0
    for k, r, i, j in y_list_p:
        un_load_cost = un_load_cost + un_load_cost_table['un_load_cost'][k] * R['qr'][r] * y[k, r, i, j].x
    for k, r, i, j in y_list_d:
        un_load_cost = un_load_cost + un_load_cost_table['un_load_cost'][k] * R['qr'][r] * y[k, r, i, j].x

    transsshipment_cost = 0
    for k, l, i, r in s_list:
        transsshipment_cost = transsshipment_cost + (
                un_load_cost_table['un_load_cost'][k] + un_load_cost_table['un_load_cost'][l]) * R['qr'][r] * s[
                                  k, l, i, r].x

    storage_cost = 0
    for k, l, i, r in s_list:
        storage_cost = storage_cost + c_storage * R['qr'][r] * (t[l, i, r, 2].x - t[k, i, r, 3].x) * s[k, l, i, r].x
    for k, r, i, j in y_list_p:
        storage_cost = storage_cost + c_storage * R['qr'][r] * (t[k, R['p'][r], r, 2].x - R['ap'][r]) * y[
            k, r, R['p'][r], j].x

    delay_penalty = 0
    for r in R.index:
        delay_penalty = delay_penalty + c_delay_table['c_delay'][r] * t_delay[r,].x * R['qr'][r]
    emission_cost = 0
    for k, r, i, j in y_list:
        emission_cost = emission_cost + carbon_tax * D[k][i][j] * R['qr'][r] * emission_table['emission'][k] / 1000 * y[
            k, r, i, j].x

    print('request_cost', request_cost, 'wait_cost', wait_cost, 'un_load_cost', un_load_cost, 'transsshipment_cost',
          transsshipment_cost, 'storage_cost', storage_cost, 'delay_penalty', delay_penalty, 'emission_cost',
          emission_cost)
    print('Running_Time',total_time)
    print('CPU_Time',cpu_time)
    return obj, cpu_time, initial_time, best_time, total_time, add_initial_best_time, add_initial_total_time


def main():
    # Data
    Data = pd.ExcelFile("Intermodal_EGS_data - Gurobi.xlsx")  # 读取的文件名

    K, N, T, A, D, R, o, o_dummy, no_route_barge, no_route_truck, K_barge_train, K_truck = instance(Data)
    print(K)
    obj,  CPU_Time, initial_time, best_time, total_time, add_initial_best_time, add_initial_total_time = optimization(
        K, N, T, A, D, R, o, o_dummy, no_route_barge, no_route_truck,
        K_barge_train, K_truck)

    return obj, CPU_Time, initial_time, best_time, total_time, add_initial_best_time, add_initial_total_time


K_T_number_exp = '116K5T_exp48'  # 读取的文件的sheet的名字
request_number = 1
service_time = 0
un_load_cost_barge_train = 0.75
un_load_cost_truck = 0.25

c_storage = 0.4
carbon_tax = 3.25
time_dependent_traveltime = 0
alpha, belta = 2, 1.5
obj, cpu_time, initial_time, best_time, total_time, add_initial_best_time, add_initial_total_time = main()