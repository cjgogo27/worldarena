# 电池能量约束和载具选择逻辑修改总结

## 修改概述
根据论文要求，对 `Intermodal_ALNS_0625.py` 进行了以下修改，以实现：
1. 电池能量消耗计算和约束
2. 载具租赁时间成本
3. 未服务订单惩罚
4. 电量不足时的载具切换逻辑

---

## 1. 新增全局参数（第8638行附近）

```python
# 电池能量相关参数
alpha_k = {}  # 时间相关能耗系数 (kWh/hour)
beta_k = {}   # 距离相关能耗系数 (kWh/km)
B_k = {}      # 电池容量 (kWh)
r_k = {}      # 租赁费率 (cost/hour)

# 目标函数权重
w4 = 0.5      # 租赁时间成本权重
w5 = 100      # 未服务订单惩罚权重（较大值以鼓励服务更多订单）
```

**说明**：这些参数对应论文中的 α_k, β_k, B_k, r_k 和权重 w4, w5。

---

## 2. 新增能量计算函数（第318行附近）

### 2.1 `calculate_energy_consumption(k, node_i, node_j, travel_time)`
计算载具k从节点i到节点j的能量消耗。

**公式**：
```
E_cons(k,i,j) = α_k × τ_ij + β_k × d_ij
```

**返回值**：
- 能量消耗 (kWh)
- 如果路径不可达，返回 `float('inf')`

### 2.2 `check_route_battery_feasibility(k, route)`
检查载具k是否有足够电池容量完成整条路线。

**逻辑**：
- 遍历路线所有段，累加能量消耗
- 检查总能量是否超过电池容量 B_k
- 返回 (可行性布尔值, 总能量消耗)

### 2.3 `check_segment_battery_feasibility(k, node_i, node_j, travel_time)`
检查载具k能否完成单段路径。

### 2.4 `initialize_battery_params(K)`
根据载具类型初始化电池参数。

**默认参数设置**：
- **船** (type=1): α=2.0, β=0.3, B=500 kWh, r=50 cost/hour
- **火车** (type=2): α=5.0, β=0.5, B=1000 kWh, r=100 cost/hour
- **卡车** (type=3): α=1.0, β=0.2, B=200 kWh, r=30 cost/hour
- **无人机/eVTOL** (其他): α=0.5, β=0.15, B=50 kWh, r=20 cost/hour

### 2.5 `try_alternative_vehicles_for_battery(i, K_R_key, current_k_list, obj_list)`
当当前载具电量不足时，尝试其他可用载具。

**逻辑**：
- 从 K_R 字典获取该请求的所有可用载具
- 对每个载具检查电池可行性
- 返回可行的载具选项列表

---

## 3. 修改 `objective_value_k` 函数（第5952行）

### 3.1 新增变量初始化
```python
rental_time_cost = 0
total_operation_time = 0
total_energy_consumption = 0
battery_feasible = True
```

### 3.2 在路径遍历中添加电池检查（第6005行附近）
```python
# 计算旅行时间
travel_time = new_try[1, x] - new_try[3, x - 1]

# 计算能量消耗
segment_energy = calculate_energy_consumption(k, p1, d1, travel_time)
total_energy_consumption += segment_energy

# 检查电池容量
battery_capacity = B_k.get(k, 100)
if segment_energy == float('inf') or total_energy_consumption > battery_capacity:
    battery_feasible = False
    # 返回极大惩罚值表示不可行
    return 100000000000, ...
```

**关键点**：
- 如果某段能量消耗超过电池容量，立即标记为不可行
- 返回极大值作为惩罚，促使算法选择其他方案

### 3.3 计算租赁时间成本（第6368行附近）
```python
# 计算总运营时间
if len(new_try[4]) > 2:
    for x in range(1, len(new_try[4])):
        travel_time = new_try[1, x] - new_try[3, x - 1]
        total_operation_time += travel_time

# 计算租赁时间成本
rental_rate = r_k.get(k, 20)
rental_time_cost = rental_rate * total_operation_time
```

**对应论文公式**：
```
C_rent_time = Σ(r_k × T_k) for all k ∈ K
```

### 3.4 修改目标函数（第6390行附近）
```python
# Demir模式
cost = w1 × (request_cost + wait_cost + transshipment_cost + un_load_cost) 
     + w2 × delay_penalty 
     + w3 × emission_cost 
     + w4 × rental_time_cost

# 非Demir模式
cost = vehicle_cost + request_cost + wait_cost + transshipment_cost 
     + un_load_cost + emission_cost + storage_cost + delay_penalty 
     + rental_time_cost
```

---

## 4. 修改 `overall_obj` 函数（第6520行）

在函数末尾添加未服务订单惩罚计算：

```python
# 计算未服务订单数量
total_requests = len(R)
unserved_requests = total_requests - served_requests

# 未服务订单惩罚
p_unserved = 100  # 每个未服务订单的基础惩罚成本
unserved_cost = w5 × p_unserved × unserved_requests

# 加入总成本
overall_cost = overall_cost + unserved_cost
```

**对应论文公式**：
```
C_unserved = Σ(p_r^un × (1 - z_r)) for all r ∈ R
```

---

## 5. 初始化调用（第8243行）

在主函数中数据读取后添加：
```python
# 初始化电池参数
initialize_battery_params(K)
```

---

## 6. 全局变量更新（第2974行）

在 `global` 声明中添加新参数：
```python
global ..., alpha_k, beta_k, B_k, r_k, w4, w5
```

---

## 实现逻辑流程

### 电量不足时的处理流程

1. **路径规划时**：
   - `objective_value_k` 计算路径能量消耗
   - 如果超过电池容量 → 返回极大惩罚值
   
2. **载具选择时**：
   - 算法尝试插入请求到载具k
   - 如果 `objective_value_k` 返回极大值 → 该方案被拒绝
   - 算法自动尝试 K_R['1k'][request_id] 中的其他载具
   
3. **无可用载具时**：
   - 如果所有载具都电量不足
   - 请求保留在 R_pool 中（未服务）
   - 在 `overall_obj` 中被罚款 `w5 × p_unserved`

### 示例场景

**场景1：单载具可行**
```
请求R1: 从节点A到节点B
- 尝试无人机k1: 能量消耗 45 kWh < 电池容量 50 kWh ✓
- 选择k1运送
```

**场景2：切换载具**
```
请求R2: 从节点C到节点D（距离较远）
- 尝试无人机k1: 能量消耗 55 kWh > 电池容量 50 kWh ✗
- 尝试卡车k3: 能量消耗 180 kWh < 电池容量 200 kWh ✓
- 选择k3运送
```

**场景3：放弃订单**
```
请求R3: 从节点E到节点F（极远距离）
- 尝试所有载具: 全部能量不足 ✗
- 订单保留在R_pool中
- 惩罚: w5 × 100 = 10000 元
```

---

## 关键约束条件

### 能量守恒约束（对应论文式7）
```
e_j^k = e_i^k - E_cons(k,i,j) × y_{ij}^k + ch_j^k
```

**实现方式**：
- 本代码简化为直接检查总能量消耗
- 不显式建模剩余电量，而是在每段计算时累加
- 如果累加值超过电池容量，立即拒绝

### 电池容量约束（对应论文式8）
```
0 ≤ e_i^k ≤ B_k
```

**实现方式**：
```python
if total_energy_consumption > battery_capacity:
    return 惩罚值
```

### 起飞可行性约束（对应论文式9）
```
e_i^k ≥ E_cons(k,i,j) → y_{ij}^k = 1
e_i^k < E_cons(k,i,j) → y_{ij}^k = 0
```

**实现方式**：
- 在 `check_segment_battery_feasibility` 中检查单段能量
- 如果不可行，该弧不会被添加到路径中

---

## 参数调优建议

### 权重参数
- **w1 (运营成本)**: 1.0 (基准)
- **w2 (延误惩罚)**: 1.0 - 5.0（取决于延误敏感度）
- **w3 (排放成本)**: 0.5 - 2.0（取决于环保要求）
- **w4 (租赁时间)**: 0.3 - 1.0（取决于时间成本重要性）
- **w5 (未服务订单)**: 50 - 200（需要非常大以保证服务率）

### 电池参数
建议从实际载具数据或文献中获取，或根据以下规则调整：
- α_k: 根据载具功率（通常 0.3-5 kWh/hour）
- β_k: 根据单位距离能耗（通常 0.1-0.8 kWh/km）
- B_k: 实际电池容量（20-1000 kWh）
- r_k: 市场租赁价格（10-200 cost/hour）

---

## 注意事项

1. **Excel数据文件**：
   - 建议在K表中添加列：alpha, beta, B, rental_rate
   - 如果没有，代码会使用默认值

2. **性能考虑**：
   - 电池检查在每次路径评估时都会执行
   - 对于大规模实例，可能影响计算速度
   - 建议预先筛选明显不可行的载具-请求配对

3. **充电站**：
   - 当前实现假设所有节点都可充电
   - 但未显式建模充电时间和功率
   - 如需精确建模，需扩展 `check_route_battery_feasibility`

4. **多段路径**：
   - 对于2k和3k（两个或三个载具）的情况
   - 每段都会独立检查电池约束
   - 确保每个载具都能完成其负责的段

---

## 测试建议

1. **单元测试**：
   ```python
   # 测试能量计算
   energy = calculate_energy_consumption(k=0, node_i=0, node_j=5, travel_time=2.0)
   
   # 测试电池可行性
   feasible, total = check_route_battery_feasibility(k=0, route=test_route)
   ```

2. **集成测试**：
   - 运行小规模实例（5-10个请求）
   - 检查是否有请求因电量不足而未服务
   - 验证成本计算是否包含租赁时间和未服务惩罚

3. **参数敏感性分析**：
   - 改变 w4, w5 观察服务率变化
   - 改变 B_k 观察载具选择变化

---

## 修改文件清单

- **主文件**: `Intermodal_ALNS_0625.py`
- **修改行数**: ~200行新增代码
- **主要修改函数**:
  - `objective_value_k` (新增电池检查和租赁成本)
  - `overall_obj` (新增未服务惩罚)
  - 新增6个辅助函数
- **全局变量**: 新增8个参数

---

## 对应论文章节

本修改实现了论文以下部分：

- **Section 2.2**: 能量消耗计算 (式5)
- **Section 2.3**: 电池约束 (式7-9)
- **Section 2.4**: 租赁时间成本 (式11)
- **Section 2.5**: 服务率约束 (式12-13)
- **Section 2.1**: 修改后的目标函数 (式1)

---

## 后续扩展建议

1. **充电站建模**：
   - 添加充电站节点类型
   - 建模充电时间和成本
   - 动态规划充电站访问

2. **多目标优化**：
   - 实现Pareto前沿搜索
   - 平衡成本、排放、服务率

3. **鲁棒性优化**：
   - 考虑能量消耗的不确定性
   - 预留电量安全边际

4. **实时调度**：
   - 动态更新电池状态
   - 在线决策充电时机
