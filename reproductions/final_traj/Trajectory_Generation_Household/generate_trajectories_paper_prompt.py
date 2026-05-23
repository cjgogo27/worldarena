"""
论文复现：基于论文中的完整Prompt设计
Human Mobility Modeling with Household Coordination Activities under Limited Information
via Retrieval-Augmented LLMs

Core mechanisms from paper:
1. Retrieval-Augmented LLM: Retrieve generated household members' activities
2. Feedback Loop: Statistical consistency (duration, frequency, timing)
3. Household Coordination: Ensure temporal consistency of coordinated activities
4. Structured Prompt: As per paper's Fig.2 system prompt

Prompts used directly from paper (Figure 2)
"""

import json
import openai
import time
import os
from datetime import datetime
from collections import defaultdict
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==================== Configuration ====================
GENERATION_MODE = "household_file"  # "num_users" or "household_file"
NUM_USERS = 5
HOUSEHOLD_ID_FILE = "E:\mayue\FinalTraj\Trajectory_Generation_multi_agent\household_id\household_id_20251211_163143.json"

# 🎯 Model Selection: "openai" or "local_llama"
USE_LOCAL_LLAMA = False  # True: 使用本地Llama模型, False: 使用OpenAI API
LOCAL_LLAMA_MODEL_PATH = "/data/mayue/cjy/Other_method/FinalTraj/Trajectory_Generation_multi_agent/llama3_1_trajectory_lora_rank32_20251209_184649/final"
LOCAL_LLAMA_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# RAG and Feedback Loop
USE_RETRIEVAL = True
USE_FEEDBACK_LOOP = True
MAX_FEEDBACK_ITERATIONS = 2
FEEDBACK_THRESHOLD = 0.15

# Activity types (10 from NHTS - California data)
ACTIVITY_CODES = {
    '1': 'home',
    '2': 'work',
    '3': 'education',
    '4': 'shopping',
    '5': 'service',
    '6': 'medical',
    '7': 'dine_out',
    '8': 'socialize',
    '9': 'exercise',
    '10': 'dropoff_pickup'
}

ALLOWED_ACTIVITIES = {
    'home', 'work', 'education', 'shopping', 'service',
    'medical', 'dine_out', 'socialize', 'exercise', 'dropoff_pickup'
}

# ====================  Model Client Creation ====================
def create_openai_client():
    """Create OpenAI client"""
    client = openai.OpenAI(
        api_key="YOUR_API_KEY_HERE",
        base_url="https://api.openai.com/v1",
        timeout=30.0
    )
    return client

def create_local_llama_client():
    """Create local Llama model client with LoRA adapter"""
    print("\n🚀 Loading local Llama model with LoRA adapter...")
    print(f"  Base model: {LOCAL_LLAMA_BASE_MODEL}")
    print(f"  LoRA adapter: {LOCAL_LLAMA_MODEL_PATH}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_LLAMA_BASE_MODEL,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_LLAMA_BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(
            base_model,
            LOCAL_LLAMA_MODEL_PATH,
            torch_dtype=torch.bfloat16
        )
        model.eval()
        
        print("✓ Model loaded successfully\n")
        return {"model": model, "tokenizer": tokenizer, "type": "local_llama"}
        
    except Exception as e:
        print(f"✗ Failed to load local model: {e}")
        raise

def create_client():
    """Create client based on configuration"""
    if USE_LOCAL_LLAMA:
        return create_local_llama_client()
    else:
        return create_openai_client()

# ==================== Data Loading ====================
def read_person_static_info(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data)} person records")
        return data
    except Exception as e:
        print(f"✗ Failed to read person file: {e}")
        return []

def read_household_static_info(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        household_dict = {item['household_id']: item for item in data}
        print(f"✓ Loaded {len(household_dict)} household records")
        return household_dict
    except Exception as e:
        print(f"✗ Failed to read household file: {e}")
        return {}

def read_household_ids(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            household_ids = json.load(f)
        if not isinstance(household_ids, list):
            return []
        print(f"✓ Loaded {len(household_ids)} target household IDs")
        return household_ids
    except Exception as e:
        print(f"✗ Failed to read household_id file: {e}")
        return []

def extract_household_id(user_id):
    return user_id.split('_')[0] if '_' in user_id else user_id

def extract_person_info(person_data):
    return {
        'user_id': person_data.get('user_id', 'Unknown'),
        'age_range': person_data.get('age_range', 'Unknown'),
        'gender': person_data.get('gender', 'Unknown'),
        'education': person_data.get('education', 'Unknown'),
        'employment_status': person_data.get('employment_status', 'Unknown'),
        'relationship': person_data.get('relationship', 'Unknown'),
        'occupation': person_data.get('occupation', 'Unknown'),
        'driver_on_travel_day': person_data.get('driver_on_travel_day', 'Unknown'),
        'work_schedule': person_data.get('work_schedule', 'Unknown'),
        'distance_to_work_miles': person_data.get('distance_to_work_miles', 0),
        'work_from_home': person_data.get('work_from_home', 'Unknown'),
        'primary_activity': person_data.get('primary_activity', 'Unknown'),
    }

def extract_household_info(household_data):
    if not household_data:
        return {
            'household_id': 'Unknown', 'home_ownership': 'Unknown',
            'household_size': 0, 'vehicle_count': 0,
            'household_income': 'Unknown', 'driver_count': 0,
            'adult_count': 0, 'young_children_count': 0,
            'msa_size': 'Unknown', 'urban_area': 'Unknown',
            'household_race': 'Unknown', 'household_hispanic': 'Unknown',
            'state': 'Unknown'
        }
    return {
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

# ==================== RAG Module ====================
class RetrievalAugmentedLLM:
    """RAG module for household coordination - stores and retrieves household activities"""
    
    def __init__(self):
        self.household_activities = {}  # {household_id: {user_id: trajectory}}
    
    def store_generated_activity(self, household_id, user_id, trajectory):
        """Store generated activity in database"""
        if household_id not in self.household_activities:
            self.household_activities[household_id] = {}
        self.household_activities[household_id][user_id] = trajectory
    
    def retrieve_household_activities(self, household_id, exclude_user_id=None):
        """Retrieve other household members' activities"""
        if household_id not in self.household_activities:
            return {}
        
        activities = {}
        for user_id, trajectory in self.household_activities[household_id].items():
            if exclude_user_id and user_id == exclude_user_id:
                continue
            activities[user_id] = trajectory
        
        return activities

# ==================== Prompt from Paper (Figure 2) ====================
def build_paper_system_prompt():
    """System prompt - Part 1: Task Description from paper"""
    return """You are a human behavior analysis expert specializing in generating realistic daily activity patterns.

# Task Description
Generate a realistic one-day activity chain for a person based on their demographic information, 
matching empirical patterns from NHTS survey data. 
Also include information about household members participating in each activity.

OUTPUT FORMAT:
[activity_type, start_time-end_time, household_members_participating]

Example:
[home, 0:00-7:15]
[work, 8:00-17:30]
[dine_out, 18:00-19:00]
[home, 19:00-24:00]

CRITICAL REQUIREMENTS:
1. Generate COMPLETE 24-hour coverage (00:00 to 24:00)
2. NO overlapping time intervals
3. NO missing time periods
4. Use realistic, variable times (not just whole hours)
5. Consider household members in coordination activities"""

def build_paper_user_prompt(person_info, household_info, other_activities=None, 
                           statistical_feedback=None, iteration=0):
    """User prompt following paper's structure (Figure 2)"""
    
    prompt = f"""# Activity Type Codes (15 types from NHTS):
1. **Home activities**: Sleep, household chores, remote work
2. **Work activities**: Professional or volunteer work
3. **School attendance**: Education activities
4. **Shopping**: Shopping activities
5. **Personal service**: Personal services, banking, government
6. **Medical**: Medical/healthcare activities
7. **Eating out**: Dining at restaurants
8. **Socializing**: Social activities, visiting friends/family
9. **Exercising**: Physical exercise, gym, sports
10. **Transport assistance**: Driving others (dropoff/pickup)
... (15 types total)

---

# Statistical Data
## Activity Type Frequencies (California NHTS 2017):
| Code | Activity | Statistic % | Accompany by Household % | Notes |
|------|----------|------------|--------------------------|-------|
| 1 | Home | 35-45% | 27.3% | Always start/end here |
| 2 | Work | 20-25% | 9.2% | For workers only |
| 10 | Transport | 4-8% | 58.8% | Common activity |

## Statistical Patterns to Match:
- **Activity Chain Length**: Vary from 3 to 14 activities, with natural mix of shorter and longer chains
- **Activity Duration Distribution**: Use VARIABLE and realistic durations based on activity type, NOT fixed slots
  - Short activities (shopping): 30-60 minutes
  - Medium activities (dining): 45-90 minutes  
  - Long activities (work): 8-10 hours
- **Activity Timing**: Create MORE CONTINUOUS distribution of activity starts/ends throughout the day
- **Household Coordination**: For each activity, indicate how many household members participate
  - Some activities are done together: dining, shopping, recreation
  - Some are done alone: work, education, personal services

---

# Guidelines
## Create natural variation
- Avoid fixed patterns or identical durations
- Natural circadian rhythm patterns

## Respect time constraints
- Activities should flow logically
- Employment status strongly affects daily patterns (work hours)
- Age influences activity types and timing
- Income level affects activity choices and locations

## Household coordination
- Consider which household members would logically participate together
- Some activities are likely shared (meals, shopping) vs alone (work, education)
- Account for school schedules, work hours in coordination

## Output Format Specification:
[activity_code, start_quarter, end_quarter, household_members_count]

Where:
- activity_code: Code number (1-15)
- start_quarter: HH:MM format
- end_quarter: HH:MM format
- household_members_count: Number of people participating (1-N)

---

# Agent Demographic Info and In Context Feedback

## Current Agent Demographic Info:
- Licensed Driver: {person_info['driver_on_travel_day']}
- Educational Attainment: {person_info['education']}
- Gender: {person_info['gender']}
- Has Job: {person_info['employment_status']}
- Occupation: {person_info['occupation']}
- Work Schedule: {person_info['work_schedule']}
- Age: {person_info['age_range']}

## Current Agent Household Info:
- Household Size: {household_info['household_size']} people
- Vehicle Count: {household_info['vehicle_count']} vehicles
- Household Income: {household_info['household_income']}
- Home Ownership: {household_info['home_ownership']}
- Young Children: {household_info['young_children_count']}
- Adult Count: {household_info['adult_count']}
- Urban Area: {household_info['urban_area']}
- MSA Size: {household_info['msa_size']}
- Location: {household_info['state']}

"""
    
    # Retrieval: Other household members' activity chains
    if other_activities:
        prompt += """## Other household members' activity chains (Retrieved from database):
"""
        for member_id, trajectory in other_activities.items():
            prompt += f"- {member_id}: {trajectory}\n"
        
        prompt += """
## Unfulfilled coordination activities that need to be addressed:
Consider activities that multiple household members should participate in together.
Examples:
- Joint meals at typical family meal times (breakfast 7-8am, dinner 6-7pm)
- Shopping together when vehicles/drivers are shared
- School dropoff/pickup coordination

"""
    
    # Feedback: Statistical deviations
    if statistical_feedback and iteration > 0:
        prompt += """## Statistic Feedback based on generated Chains Dataset:
Current distribution deviations from NHTS target - please adjust:
"""
        for activity, stats in statistical_feedback.items():
            if stats['recommendation'] == 'increase':
                prompt += f"- MORE {activity} activities: currently {stats['current']:.0f}, target {stats['target']:.0f}\n"
            else:
                prompt += f"- FEWER {activity} activities: currently {stats['current']:.0f}, target {stats['target']:.0f}\n"
        
        prompt += "\n"
    
    prompt += f"""---

# Generation Instructions

Generate a complete 24-hour activity chain for {person_info['user_id']} based on:

1. **Demographic Profile**: 
   - {person_info['age_range']}, {person_info['gender']}, {person_info['occupation']}
   - Employment: {person_info['employment_status']}, Work Schedule: {person_info['work_schedule']}
   - Driver: {person_info['driver_on_travel_day']}, Education: {person_info['education']}

2. **Household Context**:
   - Size: {household_info['household_size']}, Children: {household_info['young_children_count']}
   - Vehicles: {household_info['vehicle_count']}, Income: {household_info['household_income']}
   - Urban: {household_info['urban_area']}

3. **Constraints**:
   - Match NHTS statistical distributions
   - Ensure realistic, variable durations
   - Maintain logical activity sequences
   - Coordinate with other household members where appropriate
   - Create natural variation, avoid identical patterns

OUTPUT INSTRUCTIONS:
First, briefly explain (max 50 words) how this chain reflects the person's profile and household context.
Then output ONLY the activity chain in format:
[home, 0:00-7:15], [work, 8:00-17:30], [dine_out, 18:00-19:00], [home, 19:00-24:00]

Make sure:
- No gaps in 24-hour coverage
- No overlapping times
- Use realistic times (7:15, 8:47, 12:30 - not just whole hours)
- Include household member count when applicable
- Follow NHTS statistical patterns
"""
    
    return prompt

# ==================== Generate with Paper's Prompt ====================
def generate_trajectory_paper_prompt(client, person_info, household_info, 
                                   rag_module, iteration=0, max_retries=3):
    """Generate using paper's exact prompt structure"""
    retries = 0
    
    while retries < max_retries:
        try:
            household_id = household_info['household_id']
            user_id = person_info['user_id']
            
            # RAG: Retrieve other household members' activities
            other_activities = {}
            if USE_RETRIEVAL:
                other_activities = rag_module.retrieve_household_activities(household_id, user_id)
            
            # Build prompts following paper's structure
            system_prompt = build_paper_system_prompt()
            
            # Simple statistical feedback for demo (would need real NHTS data)
            stat_feedback = None
            if USE_FEEDBACK_LOOP and iteration > 0:
                stat_feedback = {
                    'work': {'current': 20, 'target': 25, 'recommendation': 'increase'},
                    'home': {'current': 50, 'target': 40, 'recommendation': 'decrease'},
                }
            
            user_prompt = build_paper_user_prompt(person_info, household_info, 
                                                 other_activities, stat_feedback, iteration)
            
            # Call LLM based on client type
            if isinstance(client, dict) and client.get("type") == "local_llama":
                # Local Llama model
                model = client["model"]
                tokenizer = client["tokenizer"]
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=600,
                        temperature=0.85,  # 使用训练时的温度
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            else:
                # OpenAI API
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Paper used GPT-4o-mini
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=600,
                    temperature=0.7,
                    top_p=0.95
                )
                result = response.choices[0].message.content.strip()
            
            return True, result
            
        except Exception as e:
            error_msg = str(e)
            
            if "quota" in error_msg.lower():
                return False, f"API quota exceeded: {error_msg}"
            
            if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                retries += 1
                if retries < max_retries:
                    print(f"    Network error, retrying ({retries}/{max_retries})...")
                    time.sleep(5)
                    continue
            
            retries += 1
            if retries < max_retries:
                print(f"    Generation error, retrying ({retries}/{max_retries}): {error_msg}")
                time.sleep(3)
            else:
                return False, f"Generation failed after {max_retries} retries: {error_msg}"
    
    return False, "Max retries exceeded"

# ==================== Parse & Validate ====================
def parse_trajectory(result):
    """Parse LLM output following paper's format: [activity, HH:MM-HH:MM]"""
    # Split reasoning from trajectory
    # LLM first outputs reasoning, then trajectory
    parts = result.split('\n\n')  # Try double newline first
    if len(parts) == 1:
        parts = result.split('\n')
    
    reasoning_text = ""
    trajectory_text = ""
    in_trajectory = False
    
    for line in parts:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # Detect trajectory section (contains [ and ])
        if '[' in line_stripped and ']' in line_stripped and '-' in line_stripped:
            in_trajectory = True
        
        if in_trajectory:
            trajectory_text += line_stripped + " "
        else:
            reasoning_text += line_stripped + " "
    
    # Parse trajectory from trajectory_text
    # Handle both formats: 
    # 1. [home, 0:00-7:15], [work, 8:00-17:30], ...
    # 2. [home, 0:00-7:15]
    #    [work, 8:00-17:30]
    
    trajectory_lines = []
    
    # Method 1: Find all [activity, time] patterns
    import re as regex_module
    pattern = r'\[([^\]]+)\]'  # Match content inside brackets
    matches = regex_module.findall(pattern, trajectory_text)
    
    for match in matches:
        if ',' in match and '-' in match:
            # This looks like [activity, time-time]
            trajectory_lines.append('[' + match + ']')
    
    trajectory = ", ".join(trajectory_lines) if trajectory_lines else trajectory_text.strip()
    
    return reasoning_text.strip(), trajectory

def validate_trajectory(trajectory):
    """Validate and clean trajectory - handles [activity, HH:MM-HH:MM] format"""
    if not trajectory:
        return trajectory
    
    cleaned = []
    
    # Handle multiple formats:
    # 1. [home, 0:00-7:15], [work, 8:00-17:30], [dine_out, 18:00-19:00]
    # 2. [home, 0:00-7:15]
    #    [work, 8:00-17:30]
    
    import re as regex_module
    
    # Find all [activity, time-time] patterns
    pattern = r'\[([^\]]+)\]'
    matches = regex_module.findall(pattern, trajectory)
    
    for match in matches:
        parts = [p.strip() for p in match.split(',')]
        
        if len(parts) >= 2:
            activity = parts[0].strip()
            time_range = parts[1].strip()
            
            # Validate time format (HH:MM-HH:MM or with extra data)
            if '-' in time_range:
                # Extract just the time part
                time_part = regex_module.match(r'(\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2})', time_range)
                if time_part:
                    time_range = time_part.group(1)
                
                # Map code to activity name if needed
                if activity.isdigit():
                    activity = ACTIVITY_CODES.get(activity, 'home')
                
                # Validate activity is in allowed list
                if activity.lower() not in ALLOWED_ACTIVITIES:
                    activity = 'home'  # Default
                
                cleaned.append(f"[{activity.lower()}, {time_range}]")
    
    return ", ".join(cleaned)

def convert_to_schedule(trajectory):
    """Convert to schedule format"""
    if not trajectory:
        return []
    
    schedule = []
    
    # Parse format: [activity, HH:MM-HH:MM], [activity, HH:MM-HH:MM]
    import re as regex_module
    
    # Find all [activity, time-time] patterns
    pattern = r'\[([^\]]+)\]'
    matches = regex_module.findall(pattern, trajectory)
    
    for match in matches:
        parts = [p.strip() for p in match.split(',')]
        
        if len(parts) >= 2:
            activity = parts[0].strip().lower()
            time_str = parts[1].strip()
            
            # Extract time from potential extra data
            time_match = regex_module.match(r'(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})', time_str)
            if time_match:
                start_h, start_m, end_h, end_m = time_match.groups()
                schedule.append({
                    "activity": activity,
                    "start_time": f"{start_h.zfill(2)}:{start_m}",
                    "end_time": f"{end_h.zfill(2)}:{end_m}"
                })
    
    # Sort by start time
    schedule = sorted(schedule, key=lambda x: x['start_time'])
    
    # Merge consecutive activities with same type
    schedule = merge_consecutive_activities(schedule)
    
    # Fill gaps to ensure 24-hour continuity
    schedule = fill_gaps_in_schedule(schedule)
    
    return schedule

def merge_consecutive_activities(schedule):
    """Merge consecutive activities with the same type"""
    if not schedule or len(schedule) <= 1:
        return schedule
    
    merged = []
    current = schedule[0].copy()
    
    for i in range(1, len(schedule)):
        next_item = schedule[i]
        
        # Check if activities are the same and consecutive
        if (current['activity'] == next_item['activity'] and 
            current['end_time'] == next_item['start_time']):
            # Merge: extend current activity's end time
            current['end_time'] = next_item['end_time']
        else:
            # Different activity or gap, save current and start new
            merged.append(current)
            current = next_item.copy()
    
    # Don't forget the last activity
    merged.append(current)
    
    return merged

def time_to_minutes(time_str):
    """Convert HH:MM to minutes since 00:00"""
    parts = time_str.split(':')
    return int(parts[0]) * 60 + int(parts[1])

def minutes_to_time(minutes):
    """Convert minutes since 00:00 to HH:MM"""
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"

def fill_gaps_in_schedule(schedule):
    """Fill gaps in schedule to ensure 24-hour continuity by extending previous activity"""
    if not schedule:
        return [{"activity": "home", "start_time": "00:00", "end_time": "24:00"}]
    
    filled = []
    
    # Handle start of day (should start at 00:00)
    if schedule[0]['start_time'] != '00:00':
        filled.append({
            "activity": schedule[0]['activity'],  # Use same activity as next one
            "start_time": "00:00",
            "end_time": schedule[0]['start_time']
        })
    
    # Process each activity and fill gaps
    for i in range(len(schedule)):
        current = schedule[i].copy()
        
        # Check if there's a gap to the next activity
        if i < len(schedule) - 1:
            current_end = time_to_minutes(schedule[i]['end_time'])
            next_start = time_to_minutes(schedule[i + 1]['start_time'])
            
            if current_end < next_start:
                # Extend current activity to fill the gap
                current['end_time'] = schedule[i + 1]['start_time']
        
        filled.append(current)
    
    # Handle end of day (should end at 24:00)
    if filled[-1]['end_time'] != '24:00':
        filled[-1]['end_time'] = '24:00'
    
    # Final merge: combine adjacent activities with same type
    filled = merge_adjacent_same_activities(filled)
    
    return filled

def merge_adjacent_same_activities(schedule):
    """Merge adjacent activities with the same type"""
    if not schedule or len(schedule) <= 1:
        return schedule
    
    merged = []
    current = schedule[0].copy()
    
    for i in range(1, len(schedule)):
        next_item = schedule[i]
        
        # If same activity type and consecutive, merge them
        if (current['activity'] == next_item['activity'] and
            current['end_time'] == next_item['start_time']):
            # Extend current activity
            current['end_time'] = next_item['end_time']
        else:
            # Different activity, save current and move to next
            merged.append(current)
            current = next_item.copy()
    
    # Don't forget the last activity
    merged.append(current)
    
    return merged

# ==================== Output ====================
def save_results(trajectories, output_dir):
    """Save trajectories"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Full format
    full_file = os.path.join(output_dir, f"rag_trajectories_{timestamp}.json")
    with open(full_file, 'w', encoding='utf-8') as f:
        json.dump(trajectories, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Full trajectories: {full_file}")
    
    # Schedule format
    schedule_data = []
    for item in trajectories:
        schedule = convert_to_schedule(item["trajectory"])
        if schedule:
            schedule_data.append({
                "user_id": item["user_id"],
                "schedule": schedule
            })
    
    schedule_file = os.path.join(output_dir, f"rag_trajectories_{timestamp}.json")
    with open(schedule_file, 'w', encoding='utf-8') as f:
        json.dump(schedule_data, f, ensure_ascii=False, indent=2)
    print(f"✓ Schedule format: {schedule_file}")
    
    return full_file, schedule_file

# ==================== Main ====================
def main():
    print("\n" + "="*80)
    print("论文复现：使用论文Prompt的轨迹生成")
    print("Reproduction: Trajectory Generation with Paper's Prompts")
    print("="*80 + "\n")
    
    # Load data
    print("Loading data...")
    person_data = read_person_static_info("E:\\FrankYcj\\FinalTraj\\California\\processed_data\\california_person_static.json")
    household_dict = read_household_static_info("E:\\FrankYcj\\FinalTraj\\California\\processed_data\\california_household_static.json")
    
    if not person_data or not household_dict:
        print("✗ Failed to load data")
        return
    
    # Select users
    if GENERATION_MODE == "num_users":
        selected_users = person_data[:NUM_USERS]
    else:
        target_ids = read_household_ids(HOUSEHOLD_ID_FILE)
        selected_users = [p for p in person_data 
                         if extract_household_id(p['user_id']) in target_ids]
    
    print(f"✓ Selected {len(selected_users)} users\n")
    
    # Initialize
    rag_module = RetrievalAugmentedLLM()
    client = create_client()
    
    print("="*80)
    model_name = "Local Llama (LoRA fine-tuned)" if USE_LOCAL_LLAMA else "OpenAI GPT-4o-mini"
    print(f"Starting generation with paper's prompt using {model_name}")
    print("="*80 + "\n")
    
    all_trajectories = []
    success_count = 0
    
    for i, person_raw in enumerate(selected_users):
        person_info = extract_person_info(person_raw)
        hh_id = extract_household_id(person_info['user_id'])
        household_raw = household_dict.get(hh_id)
        household_info = extract_household_info(household_raw)
        
        print(f"[{i+1}/{len(selected_users)}] Processing {person_info['user_id']}")
        print(f"  Household: {hh_id} (Size: {household_info['household_size']})")
        
        success, result = generate_trajectory_paper_prompt(client, person_info, household_info, rag_module)
        
        if success:
            print(f"  ✓ Success")
            success_count += 1
            
            reasoning, trajectory = parse_trajectory(result)
            trajectory = validate_trajectory(trajectory)
            
            rag_module.store_generated_activity(hh_id, person_info['user_id'], trajectory)
            
            all_trajectories.append({
                "user_id": person_info['user_id'],
                "household_id": hh_id,
                "person_info": person_info,
                "household_info": household_info,
                "reasoning": reasoning,
                "trajectory": trajectory,
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"  Trajectory: {trajectory[:60]}...")
        else:
            print(f"  ✗ Failed: {result[:60]}")
            if "quota" in result.lower():
                break
        
        if i < len(selected_users) - 1:
            time.sleep(3)
    
    # Save results
    print("\n" + "="*80)
    print("Saving results...")
    print("="*80 + "\n")
    
    if all_trajectories:
        output_dir = "E:\\FrankYcj\\FinalTraj\\Trajectory_Generation_Household\\output"
        save_results(all_trajectories, output_dir)
        
        print(f"\nStatistics:")
        print(f"  Total: {len(selected_users)}")
        print(f"  Success: {success_count}")
        print(f"  Success Rate: {success_count/len(selected_users)*100:.1f}%")

if __name__ == "__main__":
    main()
