import json
import openai
import time
import os
from datetime import datetime

# ==================== 配置区域 ====================
# 运行模式选择：
# "num_users" - 按指定数量生成（原模式）
# "household_file" - 从household_id.json文件读取指定家庭ID
GENERATION_MODE = "household_file"  # 可选: "num_users" 或 "household_file"

# num_users模式配置（仅当GENERATION_MODE="num_users"时有效）
NUM_USERS = 10

# household_file模式配置（仅当GENERATION_MODE="household_file"时有效）
HOUSEHOLD_ID_FILE = "E:\mayue\FinalTraj\Trajectory_Generation_multi_agent\household_id\household_id_20251211_163143.json"
# ==================================================

# 设置OpenAI客户端
def create_openai_client():
    """创建OpenAI客户端"""
    client = openai.OpenAI(
        api_key="YOUR_API_KEY_HERE",
        base_url="https://api.openai.com/v1",
        timeout=30.0
    )
    return client

# 读取household_id文件
def read_household_ids(file_path):
    """从JSON文件读取household_id列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            household_ids = json.load(f)
        if not isinstance(household_ids, list):
            print(f"错误: household_id文件格式不正确，应该是数组")
            return []
        print(f"成功读取 {len(household_ids)} 个家庭ID")
        return household_ids
    except Exception as e:
        print(f"读取household_id文件失败: {e}")
        return []

# 读取个人静态信息
def read_person_static_info(file_path):
    """读取个人静态信息JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取 {len(data)} 条个人静态信息")
        return data
    except Exception as e:
        print(f"读取个人信息文件失败: {e}")
        return []

# 读取家庭静态信息
def read_household_static_info(file_path):
    """读取家庭静态信息JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 创建字典，以household_id为键
        household_dict = {item['household_id']: item for item in data}
        print(f"成功读取 {len(household_dict)} 条家庭静态信息")
        return household_dict
    except Exception as e:
        print(f"读取家庭信息文件失败: {e}")
        return {}

# 从user_id提取household_id
def extract_household_id(user_id):
    """从user_id中提取household_id（去除下划线后的部分）"""
    # 例如: "30002756_2" -> "30002756"
    return user_id.split('_')[0] if '_' in user_id else user_id

# 提取个人信息
def extract_person_info(person_data):
    """从个人静态数据中提取所有关键信息（16个维度）"""
    person_info = {
        'user_id': person_data.get('user_id', 'Unknown'),
        'age_range': person_data.get('age_range', 'Unknown'),
        'hispanic': person_data.get('hispanic', 'Unknown'),
        'relationship': person_data.get('relationship', 'Unknown'),
        'gender': person_data.get('gender', 'Unknown'),
        'race': person_data.get('race', 'Unknown'),
        'education': person_data.get('education', 'Unknown'),
        'employment_status': person_data.get('employment_status', 'Unknown'),
        'traveled_abroad': person_data.get('traveled_abroad', 'Unknown'),
        'distance_to_work_miles': person_data.get('distance_to_work_miles', 0),
        'work_state': person_data.get('work_state', 'Unknown'),
        'driver_on_travel_day': person_data.get('driver_on_travel_day', 'Unknown'),
        'work_from_home': person_data.get('work_from_home', 'Unknown'),
        'work_schedule': person_data.get('work_schedule', 'Unknown'),
        'occupation': person_data.get('occupation', 'Unknown'),
        'primary_activity': person_data.get('primary_activity', 'Unknown')
    }
    return person_info

# 提取家庭信息
def extract_household_info(household_data):
    """从家庭静态数据中提取所有关键信息（13个维度）"""
    if not household_data:
        return {
            'household_id': 'Unknown',
            'home_ownership': 'Unknown',
            'household_size': 0,
            'vehicle_count': 0,
            'household_income': 'Unknown',
            'driver_count': 0,
            'adult_count': 0,
            'young_children_count': 0,
            'msa_size': 'Unknown',
            'urban_area': 'Unknown',
            'household_race': 'Unknown',
            'household_hispanic': 'Unknown',
            'state': 'Unknown'
        }
    
    household_info = {
        'household_id': household_data.get('household_id', 'Unknown'),
        'home_ownership': household_data.get('home_ownership', 'Unknown'),
        'household_size': household_data.get('household_size', 0),
        'vehicle_count': household_data.get('vehicle_count', 0),
        'household_income': household_data.get('household_income', 'Unknown'),
        'driver_count': household_data.get('driver_count', 0),
        'adult_count': household_data.get('adult_count', 0),
        'young_children_count': household_data.get('young_children_count', 0),
        'msa_size': household_data.get('msa_size', 'Unknown'),
        'urban_area': household_data.get('urban_area', 'Unknown'),
        'household_race': household_data.get('household_race', 'Unknown'),
        'household_hispanic': household_data.get('household_hispanic', 'Unknown'),
        'state': household_data.get('state', 'Unknown')
    }
    return household_info

# 生成用户轨迹（结合个人和家庭信息）
def generate_trajectory_with_household(client, person_info, household_info, max_retries=3):
    """使用个人和家庭信息生成24小时活动轨迹"""
    retries = 0
    
    while retries < max_retries:
        try:
            # 提取个人信息
            user_id = person_info['user_id']
            age_range = person_info['age_range']
            hispanic = person_info['hispanic']
            relationship = person_info['relationship']
            gender = person_info['gender']
            race = person_info['race']
            education = person_info['education']
            employment = person_info['employment_status']
            traveled_abroad = person_info['traveled_abroad']
            distance_to_work = person_info['distance_to_work_miles']
            work_state = person_info['work_state']
            is_driver = person_info['driver_on_travel_day']
            work_from_home = person_info['work_from_home']
            work_schedule = person_info['work_schedule']
            occupation = person_info['occupation']
            primary_activity = person_info['primary_activity']
            
            # 提取家庭信息
            household_id = household_info['household_id']
            home_ownership = household_info['home_ownership']
            household_size = household_info['household_size']
            vehicle_count = household_info['vehicle_count']
            household_income = household_info['household_income']
            driver_count = household_info['driver_count']
            adult_count = household_info['adult_count']
            young_children = household_info['young_children_count']
            msa_size = household_info['msa_size']
            urban_area = household_info['urban_area']
            household_race = household_info['household_race']
            household_hispanic = household_info['household_hispanic']
            state = household_info['state']
            
            # 构建系统提示
            system_prompt = """You are a human behavior analysis expert specializing in generating realistic daily activity patterns. 
Your task is to generate a COMPLETE 24-hour activity trajectory with ABSOLUTELY NO MISSING TIME PERIODS.
Times do NOT need to be on the hour - use realistic times like 7:23, 8:47, etc.
You MUST consider BOTH individual characteristics AND household context."""
            
            # 构建用户提示 - 使用个人和家庭的所有信息
            user_prompt = f"""
User Information (California NHTS 2017 Data - Complete Profile):

PERSONAL INFORMATION (16 dimensions):
- User ID: {user_id}
- Age Range: {age_range}
- Hispanic: {hispanic}
- Relationship: {relationship}
- Gender: {gender}
- Race: {race}
- Education: {education}
- Employment Status: {employment}
- Traveled Abroad: {traveled_abroad}
- Distance to Work: {distance_to_work} miles
- Work State: {work_state}
- Driver on Travel Day: {is_driver}
- Work From Home: {work_from_home}
- Work Schedule: {work_schedule}
- Occupation: {occupation}
- Primary Activity: {primary_activity}

HOUSEHOLD INFORMATION (13 dimensions):
- Household ID: {household_id}
- Home Ownership: {home_ownership}
- Household Size: {household_size} people
- Vehicle Count: {vehicle_count} vehicles
- Household Income: {household_income}
- Driver Count: {driver_count} drivers
- Adult Count: {adult_count} adults
- Young Children Count: {young_children} children
- MSA Size: {msa_size}
- Urban Area: {urban_area}
- Household Race: {household_race}
- Household Hispanic: {household_hispanic}
- State: {state}

CRITICAL REQUIREMENTS:
1. Generate a COMPLETE 24-hour activity trajectory (00:00 to 24:00)
2. NO MISSING TIME PERIODS - every minute must be accounted for
3. Use ONLY these 10 activity types (NO OTHER ACTIVITIES ALLOWED):
   • home - at home (sleeping, resting, meals at home, leisure at home)
   • work - at workplace (includes commuting time - use home before and after)
   • education - at school/university
   • shopping - shopping activities
   • service - personal services (bank, government, etc.)
   • medical - medical appointments, healthcare
   • dine_out - eating out at restaurants
   • socialize - social activities, visiting friends/family
   • exercise - physical exercise, gym, sports
   • dropoff_pickup - dropping off or picking up people
   
   ⚠️ DO NOT USE: commute, lunch, breakfast, dinner, sleep, rest, leisure, etc.
   ⚠️ For commuting: use "home" before work and after work
   ⚠️ For meals at home: use "home"
   ⚠️ For meals outside: use "dine_out"

4. Consider ALL 16 personal characteristics AND 13 household characteristics:
   
   PERSONAL FACTORS:
   - Age, Hispanic status, relationship, gender, race
   - Education level, employment status, work schedule (full-time/part-time)
   - Travel abroad experience, distance to work, work location state
   - Driver status, work from home status, occupation, primary activity
   
   HOUSEHOLD FACTORS:
   - Home ownership (Own vs Rent) affects stability and activity patterns
   - Household size and composition (adults, young children) affects responsibilities
   - Vehicle availability relative to household size affects mobility
   - Household income affects activity choices and locations
   - Urban vs rural location affects travel times and activity accessibility
   - MSA size affects available activities and commute patterns

5. Times do NOT need to be on the hour - use realistic times like 7:15, 8:45, 12:30, etc.

Output Format:
First, provide a brief reasoning (max 100 words) about why this trajectory fits BOTH the individual AND household profile.
Then on a new line, output ONLY the trajectory in this EXACT format:
0:00-7:23 home, 7:23-8:15 exercise, 8:15-17:30 work, 17:30-18:45 shopping, 18:45-24:00 home

Important Rules:
- Start time MUST be 0:00 (or 00:00)
- End time MUST be 24:00
- Time intervals must NOT overlap
- Time intervals must be continuous with NO GAPS
- Activity names must be EXACTLY one of the 10 allowed types (lowercase, single word)
- NO activities like: commute, lunch, breakfast, dinner, sleep, etc.
- Times can be realistic (not just whole hours): 7:15, 8:47, 12:30, etc.
- Format: H:MM or HH:MM (both acceptable)
"""
            
            # 调用OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=600,
                temperature=0.7
            )
            
            result = response.choices[0].message.content.strip()
            
            # 返回成功结果
            return True, result
            
        except Exception as e:
            error_msg = str(e)
            
            # 检查是否是配额不足错误
            if "insufficient_user_quota" in error_msg.lower() or "quota" in error_msg.lower():
                return False, f"API配额不足: {error_msg}\n建议: 1) 减少批处理数量 2) 等待配额重置 3) 使用其他API密钥"
            
            # 检查是否是网络错误
            if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                retries += 1
                if retries < max_retries:
                    print(f"  网络错误，正在重试 ({retries}/{max_retries})...")
                    time.sleep(5)
                    continue
            
            # 其他错误
            retries += 1
            if retries < max_retries:
                print(f"  生成出错，正在重试 ({retries}/{max_retries}): {error_msg}")
                time.sleep(3)
            else:
                return False, f"生成失败 (已重试{max_retries}次): {error_msg}"
    
    return False, "达到最大重试次数"

# 解析生成的轨迹
def parse_trajectory_result(result):
    """解析API返回的轨迹结果，分离推理和轨迹部分"""
    lines = result.split('\n')
    
    reasoning_lines = []
    trajectory_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 更精确的轨迹识别：包含时间模式 (数字:数字-数字:数字)
        import re
        time_pattern = r'\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}'
        
        if re.search(time_pattern, line):
            # 这是轨迹行
            trajectory_lines.append(line)
        else:
            # 这是推理行
            reasoning_lines.append(line)
    
    reasoning = ' '.join(reasoning_lines).strip()
    trajectory = ', '.join(trajectory_lines).strip() if trajectory_lines else ''
    
    # 如果没有找到明确的轨迹，尝试从整个结果中提取
    if not trajectory:
        import re
        # 匹配格式: 数字:数字-数字:数字 活动名
        pattern = r'\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\s+\w+'
        matches = re.findall(pattern, result)
        if matches:
            trajectory = ', '.join(matches)
    
    # 验证并清理轨迹中的活动类型
    trajectory = validate_and_clean_trajectory(trajectory)
    
    return reasoning, trajectory

# 将轨迹转换为schedule格式
def convert_trajectory_to_schedule(trajectory):
    """将轨迹字符串转换为schedule格式的列表"""
    import re
    
    if not trajectory:
        return []
    
    schedule = []
    
    # 分割轨迹段
    segments = trajectory.split(',')
    
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        
        # 提取时间和活动
        match = re.match(r'(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s+(.+)', segment)
        if match:
            start_hour = match.group(1).zfill(2)
            start_min = match.group(2)
            end_hour = match.group(3).zfill(2)
            end_min = match.group(4)
            activity = match.group(5).strip()
            
            schedule.append({
                "activity": activity,
                "start_time": f"{start_hour}:{start_min}",
                "end_time": f"{end_hour}:{end_min}"
            })
    
    return schedule

# 验证和清理轨迹中的活动类型
def validate_and_clean_trajectory(trajectory):
    """验证轨迹只包含允许的活动类型，并自动修正常见错误"""
    if not trajectory:
        return trajectory
    
    # 允许的活动类型
    allowed_activities = {
        'home', 'work', 'education', 'shopping', 'service', 
        'medical', 'dine_out', 'socialize', 'exercise', 'dropoff_pickup'
    }
    
    # 常见的错误映射（自动修正）
    activity_mapping = {
        'commute': 'home',
        'commute to work': 'home',
        'commute home': 'home',
        'commuting': 'home',
        'travel': 'home',
        'lunch': 'dine_out',
        'breakfast': 'home',
        'dinner': 'home',
        'meal': 'home',
        'sleep': 'home',
        'rest': 'home',
        'leisure': 'home',
        'relax': 'home',
        'school': 'education',
        'university': 'education',
        'college': 'education',
        'gym': 'exercise',
        'workout': 'exercise',
        'hospital': 'medical',
        'doctor': 'medical',
        'clinic': 'medical',
        'restaurant': 'dine_out',
        'eating out': 'dine_out',
        'visit': 'socialize',
        'meeting': 'socialize',
        'grocery': 'shopping',
        'store': 'shopping'
    }
    
    import re
    
    # 分割轨迹段
    segments = trajectory.split(',')
    cleaned_segments = []
    
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        
        # 提取时间和活动
        match = re.match(r'(\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2})\s+(.+)', segment)
        if match:
            time_range = match.group(1)
            activity = match.group(2).strip().lower()
            
            # 移除可能的额外描述
            activity = activity.split('(')[0].strip()
            activity = activity.split('[')[0].strip()
            
            # 检查是否需要映射
            if activity in activity_mapping:
                activity = activity_mapping[activity]
                print(f"    ⚠ 自动修正活动: '{match.group(2)}' -> '{activity}'")
            
            # 检查是否在允许列表中
            if activity in allowed_activities:
                cleaned_segments.append(f"{time_range} {activity}")
            else:
                # 未知活动，默认改为 home
                print(f"    ⚠ 未知活动 '{activity}' 已替换为 'home'")
                cleaned_segments.append(f"{time_range} home")
        else:
            # 无法解析的段，保留原样
            cleaned_segments.append(segment)
    
    return ', '.join(cleaned_segments)

# 保存生成的轨迹
def save_trajectories(trajectories, output_dir):
    """保存轨迹到JSON文件（完整版和Schedule格式版）"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 保存完整版本（包含所有信息）
    output_file = os.path.join(output_dir, f"california_trajectories_with_household_{timestamp}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(trajectories, f, ensure_ascii=False, indent=2)
    print(f"\n完整版轨迹已保存至: {output_file}")
    
    # 创建schedule格式版本（新格式，类似multi_agent的输出）
    schedule_trajectories = []
    for item in trajectories:
        schedule = convert_trajectory_to_schedule(item["trajectory"])
        if schedule:  # 只添加成功转换的轨迹
            schedule_trajectories.append({
                "user_id": item["user_id"],
                "schedule": schedule
            })
    
    # 保存schedule格式版本
    schedule_file = os.path.join(output_dir, f"all_trajectories_{timestamp}.json")
    with open(schedule_file, 'w', encoding='utf-8') as f:
        json.dump(schedule_trajectories, f, ensure_ascii=False, indent=2)
    print(f"Schedule格式轨迹已保存至: {schedule_file}")
    
    return output_file, schedule_file

# 保存统计报告
def save_statistics_report(statistics, output_dir):
    """保存统计报告"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"generation_report_{timestamp}.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("California Trajectory Generation Report (With Household Info)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Statistics:\n")
        f.write("-"*80 + "\n")
        for key, value in statistics.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"统计报告已保存至: {report_file}")
    return report_file

# 主函数
def main():
    print("="*80)
    print("California Trajectory Generation System (With Household Information)")
    print("Using Person + Household Static Information")
    print("="*80)
    print(f"\n运行模式: {GENERATION_MODE}")
    if GENERATION_MODE == "household_file":
        print(f"Household ID文件: {HOUSEHOLD_ID_FILE}")
    elif GENERATION_MODE == "num_users":
        print(f"生成用户数: {NUM_USERS}")
    print("="*80 + "\n")
    
    # 文件路径配置
    person_file_path = "E:/mayue/FinalTraj/Oklahoma/processed_data/oklahoma_person_static.json"
    household_file_path = "E:/mayue/FinalTraj/Oklahoma/processed_data/oklahoma_household_static.json"
    output_dir = "E:/mayue/FinalTraj/Trajectory_Generation_Household/output"
    
    # 读取个人静态信息
    print(f"正在读取个人静态信息: {person_file_path}")
    person_data = read_person_static_info(person_file_path)
    
    if not person_data:
        print("错误: 没有找到有效的个人静态信息，程序退出。")
        return
    
    # 读取家庭静态信息
    print(f"正在读取家庭静态信息: {household_file_path}")
    household_dict = read_household_static_info(household_file_path)
    
    if not household_dict:
        print("错误: 没有找到有效的家庭静态信息，程序退出。")
        return
    
    # 根据模式选择用户
    total_users = len(person_data)
    print(f"\n可用用户总数: {total_users}")
    print(f"可用家庭总数: {len(household_dict)}")
    print(f"\n当前运行模式: {GENERATION_MODE}")
    
    selected_users = []
    
    if GENERATION_MODE == "household_file":
        # 模式1: 从household_id文件读取
        print(f"\n正在从文件读取household_id: {HOUSEHOLD_ID_FILE}")
        target_household_ids = read_household_ids(HOUSEHOLD_ID_FILE)
        
        if not target_household_ids:
            print("错误: 未能读取有效的household_id列表，程序退出。")
            return
        
        print(f"目标家庭数量: {len(target_household_ids)}")
        print(f"目标家庭ID: {target_household_ids}")
        
        # 筛选出属于这些家庭的所有用户
        for person in person_data:
            person_info = extract_person_info(person)
            household_id = extract_household_id(person_info['user_id'])
            if household_id in target_household_ids:
                selected_users.append(person)
        
        print(f"\n找到 {len(selected_users)} 个用户属于这些家庭")
        
        if not selected_users:
            print("警告: 没有找到任何用户属于指定的家庭ID")
            return
        
        # 按家庭分组显示
        household_user_count = {}
        for person in selected_users:
            person_info = extract_person_info(person)
            household_id = extract_household_id(person_info['user_id'])
            household_user_count[household_id] = household_user_count.get(household_id, 0) + 1
        
        print("\n各家庭成员数量:")
        for hh_id, count in sorted(household_user_count.items()):
            print(f"  家庭 {hh_id}: {count} 个成员")
        
    elif GENERATION_MODE == "num_users":
        # 模式2: 按数量生成（原模式）
        print("\n建议生成数量:")
        print("  - 测试: 5-10 个用户")
        print("  - 小批量: 50-100 个用户")
        print("  - 中等批量: 500-1000 个用户")
        print("  - 警告: 大批量生成可能消耗大量API配额\n")
        
        num_users = NUM_USERS
        print(f"当前设置: 生成前 {num_users} 个用户的轨迹")
        print("(如需修改，请编辑脚本顶部的 NUM_USERS 变量)\n")
        
        # 选择用户
        selected_users = person_data[:num_users]
    
    else:
        print(f"错误: 未知的运行模式 '{GENERATION_MODE}'")
        print("请设置 GENERATION_MODE 为 'num_users' 或 'household_file'")
        return
    
    num_users = len(selected_users)
    
    # 创建OpenAI客户端
    print("正在初始化OpenAI客户端...")
    client = create_openai_client()
    print("客户端初始化成功\n")
    
    # 生成轨迹
    all_trajectories = []
    success_count = 0
    failure_count = 0
    matched_household_count = 0
    unmatched_household_count = 0
    
    print("="*80)
    print("开始生成轨迹（结合个人和家庭信息）")
    print("="*80 + "\n")
    
    for i, person_raw_data in enumerate(selected_users):
        # 提取个人信息
        person_info = extract_person_info(person_raw_data)
        user_id = person_info['user_id']
        
        # 从user_id提取household_id
        household_id = extract_household_id(user_id)
        
        # 查找对应的家庭信息
        household_raw_data = household_dict.get(household_id)
        household_info = extract_household_info(household_raw_data)
        
        if household_raw_data:
            matched_household_count += 1
            household_status = "✓ 找到家庭信息"
        else:
            unmatched_household_count += 1
            household_status = "⚠ 未找到家庭信息（使用默认值）"
        
        print(f"\n[{i+1}/{num_users}] 正在处理用户: {user_id}")
        print(f"  家庭ID: {household_id} - {household_status}")
        print(f"  个人: 年龄={person_info['age_range']}, 性别={person_info['gender']}, 职业={person_info['occupation']}")
        print(f"  家庭: 规模={household_info['household_size']}人, 车辆={household_info['vehicle_count']}辆, 收入={household_info['household_income']}")
        print(f"  家庭: 儿童={household_info['young_children_count']}人, 地区={household_info['urban_area']}")
        
        # 生成轨迹
        success, result = generate_trajectory_with_household(client, person_info, household_info)
        
        if success:
            print(f"  ✓ 轨迹生成成功")
            success_count += 1
            
            # 解析结果
            reasoning, trajectory = parse_trajectory_result(result)
            
            # 存储结果
            all_trajectories.append({
                "user_id": user_id,
                "household_id": household_id,
                "person_info": person_info,
                "household_info": household_info,
                "reasoning": reasoning,
                "trajectory": trajectory,
                "raw_result": result,
                "generation_time": datetime.now().isoformat()
            })
            
            print(f"  推理: {reasoning[:80]}..." if len(reasoning) > 80 else f"  推理: {reasoning}")
            print(f"  轨迹: {trajectory[:80]}..." if len(trajectory) > 80 else f"  轨迹: {trajectory}")
        else:
            print(f"  ✗ 轨迹生成失败: {result}")
            failure_count += 1
            
            # 如果是配额错误，提前终止
            if "quota" in result.lower():
                print("\n警告: API配额不足，停止生成。")
                break
        
        # 添加延迟，避免API请求过于频繁
        if i < num_users - 1:
            delay = 3
            print(f"  等待 {delay} 秒...")
            time.sleep(delay)
    
    # 保存结果
    print("\n" + "="*80)
    print("生成完成，正在保存结果")
    print("="*80 + "\n")
    
    if all_trajectories:
        output_file, schedule_file = save_trajectories(all_trajectories, output_dir)
        
        # 统计信息
        statistics = {
            "运行模式": GENERATION_MODE,
            "总用户数": total_users,
            "总家庭数": len(household_dict),
            "选择生成数": num_users,
            "成功生成": success_count,
            "失败数量": failure_count,
            "成功率": f"{success_count/num_users*100:.2f}%",
            "匹配到家庭信息": matched_household_count,
            "未匹配家庭信息": unmatched_household_count,
            "家庭匹配率": f"{matched_household_count/num_users*100:.2f}%",
            "完整版输出文件": output_file,
            "Schedule格式输出文件": schedule_file
        }
        
        report_file = save_statistics_report(statistics, output_dir)
        
        # 打印统计信息
        print("\n" + "="*80)
        print("生成统计")
        print("="*80)
        for key, value in statistics.items():
            print(f"{key}: {value}")
        print("="*80 + "\n")
        
    else:
        print("\n错误: 没有成功生成任何轨迹。")

if __name__ == "__main__":
    main()
