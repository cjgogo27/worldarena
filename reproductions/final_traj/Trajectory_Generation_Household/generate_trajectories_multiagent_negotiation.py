
import json
import time
import os
from openai import OpenAI
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Any

API_KEY = "YOUR_API_KEY_HERE"
BASE_URL = "https://api.openai.com/v1"
MODEL = "gpt-4o"
TIMEOUT = 30

PERSON_FILE = r"E:\FrankYcj\FinalTraj\California\processed_data\california_person_static.json"
HOUSEHOLD_FILE = r"E:\FrankYcj\FinalTraj\California\processed_data\california_household_static.json"
OUTPUT_DIR = r"E:\FrankYcj\FinalTraj\Trajectory_Generation_Household\output"
OUTPUT_TRAJECTORIES_DIR = r"E:\FrankYcj\FinalTraj\Trajectory_Generation_Household\output_trajectories"

ALLOWED_ACTIVITIES = {
    "home", "work", "education", "shopping", "service", 
    "medical", "dine_out", "socialize", "exercise", "dropoff_pickup"
}

TASK_TO_ACTIVITY_MAPPING = {
    "shopping": "shopping",
    "grocery_shopping": "shopping",
    "vehicle_maintenance": "service",
    "car_maintenance": "service",
    "house_cleaning": "home",
    "cleaning": "home",
    "laundry": "home",
    "cooking": "home",
    "medical_appointment": "medical",
    "doctor_visit": "medical",
    "dropoff_pickup": "dropoff_pickup",
    "school_pickup": "dropoff_pickup",
    "childcare": "dropoff_pickup",
}

def map_task_to_activity(task_id: str) -> str:
    task_id_lower = task_id.lower().strip()
    
    if task_id_lower in TASK_TO_ACTIVITY_MAPPING:
        return TASK_TO_ACTIVITY_MAPPING[task_id_lower]
    
    for key, value in TASK_TO_ACTIVITY_MAPPING.items():
        if key in task_id_lower or task_id_lower in key:
            return value
    
    return "service"


class Agent:
    def __init__(
        self, 
        agent_id: str, 
        person_info: Dict[str, Any],
        household_info: Dict[str, Any],
        other_members: Dict[str, Dict[str, Any]],
        openai_client: OpenAI
    ):
        self.agent_id = agent_id
        self.person_info = person_info
        self.household_info = household_info
        self.other_members = other_members 
        self.client = openai_client

        self.mandatory_activities = []  
        self.conversation_history = []  
        self.allocated_tasks = []  
        self.personal_activities = []  
        self.final_schedule = []  
    
    def set_mandatory_activities(self, activities: List[Dict[str, Any]]):
        self.mandatory_activities = activities
    
    def update_conversation_history(self, conversation: List[Dict[str, str]]):
        self.conversation_history = conversation
    
    def set_allocated_tasks(self, tasks: List[Dict[str, Any]]):
        self.allocated_tasks = tasks
    
    def set_personal_activities(self, activities: List[Dict[str, Any]]):
        self.personal_activities = activities
    
    def set_final_schedule(self, schedule: List[Dict[str, Any]]):
        self.final_schedule = schedule
    
    def get_mandatory_summary(self) -> str:
        if not self.mandatory_activities:
            return "Free all day"
        return ", ".join([
            f"{a['activity']} {a['start_time']}-{a['end_time']}"
            for a in self.mandatory_activities
        ])
    
    def get_my_profile_summary(self) -> str:
        p = self.person_info
        return f"""- User ID: {p['user_id']}
- Relationship: {p['relationship']}
- Age: {p['age_range']}, Gender: {p['gender']}
- Race: {p['race']}, Hispanic: {p['hispanic']}
- Education: {p['education']}
- Employment: {p['employment_status']}, Schedule: {p['work_schedule']}
- Occupation: {p['occupation']}
- Primary activity: {p['primary_activity']}
- Work from home: {p['work_from_home']}
- Distance to work: {p['distance_to_work_miles']} miles
- Work state: {p['work_state']}
- Can drive: {p['driver_on_travel_day']}
- Today's schedule: {self.get_mandatory_summary()}"""
    
    def get_household_summary(self) -> str:
        h = self.household_info
        return f"""- Household ID: {h['household_id']}
- Size: {h['household_size']} people
- Young children: {h['young_children_count']}, Adults: {h['adult_count']}
- Vehicles: {h['vehicle_count']}, Drivers: {h['driver_count']}
- Income: {h['household_income']}
- Home ownership: {h['home_ownership']}
- MSA size: {h['msa_size']}
- Urban area: {h['urban_area']}
- Household race: {h['household_race']}
- Household hispanic: {h['household_hispanic']}
- State: {h['state']}"""
    
    def get_other_member_summary(self, other_agents: Dict[str, 'Agent']) -> str:
        summary_lines = []
        for uid, agent in other_agents.items():
            if uid == self.agent_id:
                continue
            info = agent.person_info
            mand_str = agent.get_mandatory_summary()
            
            summary_lines.append(
                f"- {uid} ({info['relationship']}, {info['gender']}, {info['age_range']}): "
                f"{info['employment_status']}, works {info['work_schedule']}, "
                f"schedule today: {mand_str}, can drive: {info['driver_on_travel_day']}"
            )
        return "\n".join(summary_lines)
    
    def propose_mandatory_activities(self, max_retries: int = 3) -> bool:
        """Phase 1"""
        has_young_children = self.household_info.get('young_children_count', 0) > 0
        
        prompt = f"""You are Agent {self.agent_id} proposing your MANDATORY activities.

YOUR COMPLETE PROFILE:
{self.get_my_profile_summary()}

HOUSEHOLD INFO:
- Young children in household: {self.household_info.get('young_children_count', 0)}

Task: Propose your MANDATORY activities (work/education) ONLY.

CRITICAL RULES:
1. Include ONLY work or education activities
2. DO NOT include commuting or travel activities
3. {'Include dropoff_pickup ONLY if you are responsible for childcare' if has_young_children else 'DO NOT include dropoff_pickup (no young children in household)'}
4. Use ONLY these activity types: work, education{', dropoff_pickup' if has_young_children else ''}

Output JSON format:
```json
{{
  "mandatory_activities": [
    {{
      "activity": "work",
      "start_time": "08:30",
      "end_time": "17:15",
      "uses_vehicle": true,
      "reasoning": "Full-time worker"
    }}
  ]
}}
```

If no mandatory activities, return: {{"mandatory_activities": []}}

Generate your mandatory activities:"""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a household member proposing mandatory activities."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                
                result_text = response.choices[0].message.content.strip()
                
                if "```json" in result_text:
                    json_start = result_text.find("```json") + 7
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()
                elif "```" in result_text:
                    json_start = result_text.find("```") + 3
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()
                
                result = json.loads(result_text)
                self.mandatory_activities = result.get("mandatory_activities", [])
                print(f"  → Agent {self.agent_id}: Generated {len(self.mandatory_activities)} mandatory activities")
                return True
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"   Agent {self.agent_id}: Failed to generate mandatory activities")
                    self.mandatory_activities = []
                    return False
        
        return False
    
    def propose_initial(
        self, 
        tasks_list: List[Dict[str, Any]], 
        all_agents: Dict[str, 'Agent'],
        max_retries: int = 3
    ) -> str:
        """Phase 2"""
        tasks_desc = "\n".join([
            f"- {t['task_id']} ({t['activity']}, ~{t['duration_minutes']}min)"
            for t in tasks_list
        ])
        
        prompt = f"""You are household member {self.agent_id} in a family meeting discussing today's tasks.

YOUR PROFILE:
{self.get_my_profile_summary()}

HOUSEHOLD INFO:
{self.get_household_summary()}

OTHER FAMILY MEMBERS:
{self.get_other_member_summary(all_agents)}

TASKS to allocate today:
{tasks_desc}

Based on YOUR schedule and what you know about OTHER family members' schedules, propose which task(s) you can handle. Be natural and reference others if relevant.
Example: "I can do the shopping around 6pm after work. But for the morning pickup, maybe [other_member] can handle it since they start work later?"

Your proposal:"""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                    max_tokens=150
                )
                statement = response.choices[0].message.content.strip().strip('"')
                return statement
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    return f"I can help with some tasks if needed."
        
        return "I can help with some tasks if needed."
    
    def respond(
        self, 
        tasks_list: List[Dict[str, Any]], 
        all_agents: Dict[str, 'Agent'],
        max_retries: int = 3
    ) -> Optional[str]:
        """Phase 2"""
        recent_conv = self.conversation_history[-6:] if len(self.conversation_history) > 6 else self.conversation_history
        conv_text = "\n".join([f"{h['speaker']}: {h['statement']}" for h in recent_conv])
        
        prompt = f"""You are {self.agent_id} continuing the family discussion.

YOUR PROFILE:
{self.get_my_profile_summary()}

OTHER FAMILY MEMBERS:
{self.get_other_member_summary(all_agents)}

RECENT CONVERSATION:
{conv_text}

Based on what others said and what you know about everyone's schedules, respond naturally. 
Be brief (1-2 sentences) and natural. Reference other family members by their relationship if it makes sense.
Response rules:
1. ONLY discuss unassigned tasks (ignore already allocated ones)
2. Do NOT repeat information from previous messages
3. Be brief (1-2 sentences) and focus on moving the discussion forward
Your response:"""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                    max_tokens=100
                )
                statement = response.choices[0].message.content.strip().strip('"')
                return statement if len(statement) >= 10 else None
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    return None
        
        return None
    
    def propose_personal_activities(self, max_retries: int = 3) -> bool:
        """Phase 3"""
        mand_summary = "\n".join([
            f"- {a['activity']}: {a['start_time']}-{a['end_time']}"
            for a in self.mandatory_activities
        ]) if self.mandatory_activities else "None"
        
        task_lines = []
        if self.allocated_tasks:
            for t in self.allocated_tasks:
                if isinstance(t, dict):
                    task_id = t.get('task_id', 'household_task')
                    sched = t.get('scheduled_time', {})
                    if isinstance(sched, dict):
                        start = sched.get('start', '12:00')
                        end = sched.get('end', '13:00')
                        task_lines.append(f"- {task_id}: {start}-{end}")
                    elif isinstance(sched, str):
                        task_lines.append(f"- {task_id}: {sched}")
        task_summary = "\n".join(task_lines) if task_lines else "None"
        
        prompt = f"""You are Agent {self.agent_id} proposing PERSONAL activities.

Your Already Scheduled:
Mandatory: {mand_summary}
Household Tasks: {task_summary}

Propose personal activities (exercise, socialize, dine_out):
```json
{{
  "personal_activities": [
    {{
      "activity": "exercise",
      "start_time": "19:00",
      "end_time": "20:00",
      "uses_vehicle": false,
      "reasoning": "Evening gym session"
    }}
  ]
}}
```

Generate personal activities:"""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are proposing personal activities for remaining time."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    max_tokens=400
                )
                
                result_text = response.choices[0].message.content.strip()
                
                if "```json" in result_text:
                    json_start = result_text.find("```json") + 7
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()
                elif "```" in result_text:
                    json_start = result_text.find("```") + 3
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()
                
                result = json.loads(result_text)
                self.personal_activities = result.get("personal_activities", [])
                print(f"   Agent {self.agent_id}: Generated {len(self.personal_activities)} personal activities")
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"   Agent {self.agent_id}: Failed to generate personal activities")
                    self.personal_activities = []
                    return False
        
        return False
    
    def schedule_full_day(self, max_retries: int = 3) -> bool:
        all_activities = []
        all_activities.extend([{"source": "mandatory", **a} for a in self.mandatory_activities])
        
        for t in self.allocated_tasks:
            if isinstance(t, dict):
                task_id = t.get("task_id", "household_task")
                mapped_activity = map_task_to_activity(task_id)
                
                sched_time = t.get("scheduled_time", {})
                if isinstance(sched_time, dict):
                    all_activities.append({
                        "source": "household",
                        "activity": mapped_activity,  
                        "start_time": sched_time.get("start", "12:00"),
                        "end_time": sched_time.get("end", "13:00")
                    })
                elif isinstance(sched_time, str):
                    all_activities.append({
                        "source": "household",
                        "activity": mapped_activity,  
                        "start_time": "12:00",
                        "end_time": "13:00"
                    })
        
        all_activities.extend([{"source": "personal", **a} for a in self.personal_activities])
        
        activities_desc = "\n".join([
            f"- {a['activity']}: {a['start_time']}-{a['end_time']} ({a['source']})"
            for a in all_activities
        ])
        
        has_young_children = self.household_info.get('young_children_count', 0) > 0
        allowed_for_agent = list(ALLOWED_ACTIVITIES)
        if not has_young_children and 'dropoff_pickup' in allowed_for_agent:
            allowed_for_agent.remove('dropoff_pickup')
        
        prompt = f"""Create complete 24-hour schedule for {self.agent_id}.

Activities to include:
{activities_desc}

HOUSEHOLD CONTEXT:
- Young children: {self.household_info.get('young_children_count', 0)}

Requirements:
1. Cover 00:00-24:00 continuously
2. No overlaps or gaps
3. Sort chronologically
4. Use ONLY these activities: {', '.join(allowed_for_agent)}
5. {'DO NOT use dropoff_pickup (no young children in household)' if not has_young_children else 'Use dropoff_pickup ONLY for childcare/school pickup'}

Output:
```json
{{
  "full_schedule": [
    {{"activity": "home", "start_time": "00:00", "end_time": "07:30"}},
    {{"activity": "work", "start_time": "07:30", "end_time": "17:00"}},
    {{"activity": "home", "start_time": "17:00", "end_time": "24:00"}}
  ]
}}
```

Generate full schedule:"""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are scheduling a complete 24-hour day with no gaps."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=800
                )
                
                result_text = response.choices[0].message.content.strip()
                
                if "```json" in result_text:
                    json_start = result_text.find("```json") + 7
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()
                elif "```" in result_text:
                    json_start = result_text.find("```") + 3
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()
                
                result = json.loads(result_text)
                self.final_schedule = result.get("full_schedule", [])
                print(f"  → Agent {self.agent_id}: Generated full day schedule with {len(self.final_schedule)} time slots")
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"   Agent {self.agent_id}: Failed to generate full day schedule")
                    self.final_schedule = []
                    return False
        
        return False

# ===== Utility Functions =====
def create_openai_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=TIMEOUT)


def load_json(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_household_id(user_id: str) -> str:
    return user_id.split('_')[0] if '_' in user_id else user_id


def group_by_household(persons_list: List[Dict]) -> Dict[str, List[str]]:
    households = defaultdict(list)
    for p in persons_list:
        uid = p.get("user_id")
        if uid:
            hid = extract_household_id(uid)
            households[hid].append(uid)
    return households


def extract_person_info(person_data: Dict) -> Dict[str, Any]:
    return {
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


def extract_household_info(household_data: Optional[Dict]) -> Dict[str, Any]:
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

# ===== Phase 2a: Propose Household Tasks =====
def propose_household_tasks(client: OpenAI, household_info: Dict, member_infos: Dict, max_retries: int = 3) -> tuple:
    members_summary = "\n".join([
        f"- {uid}: {info['age_range']}, {info['employment_status']}, Driver: {info['driver_on_travel_day']}"
        for uid, info in member_infos.items()
    ])
    
    has_young_children = household_info.get('young_children_count', 0) > 0
    
    if has_young_children:
        allowed_activities_str = "shopping, service, medical, dropoff_pickup"
    else:
        allowed_activities_str = "shopping, service, medical"
    
    prompt = f"""Generate household tasks for this family:

Household:
- Size: {household_info['household_size']} people
- Young children: {household_info['young_children_count']}
- Vehicles: {household_info['vehicle_count']}
- Income: {household_info['household_income']}

Members:
{members_summary}

CRITICAL RULES: 
1. Use ONLY these activity types: {allowed_activities_str}
2. {'Include dropoff_pickup for childcare/school pickup' if has_young_children else 'DO NOT use dropoff_pickup (no young children in household)'}
3. DO NOT use dropoff_pickup for adult commuting
4. Use 'service' for vehicle maintenance, repairs, etc.
5. Use 'shopping' for grocery shopping, errands, etc.

Generate 2-3 household tasks:
```json
{{
  "household_tasks": [
    {{
      "task_id": "shopping",
      "activity": "shopping",
      "duration_minutes": 60,
      "priority": "high"
    }}
  ]
}}
```

Generate tasks:"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=400
            )
            
            result_text = response.choices[0].message.content.strip()
            
            if "```json" in result_text:
                json_start = result_text.find("```json") + 7
                json_end = result_text.find("```", json_start)
                result_text = result_text[json_start:json_end].strip()
            
            result = json.loads(result_text)
            
            tasks = result.get("household_tasks", [])
            valid_tasks = []
            
            for task in tasks:
                activity = task.get("activity", "")
                task_id = task.get("task_id", "")
                
                mapped_activity = map_task_to_activity(task_id)
                
                if mapped_activity == "dropoff_pickup" or activity == "dropoff_pickup":
                    if not has_young_children:
                        print(f"  ⚠ Filtered out dropoff_pickup task (no young children)")
                        continue
                
                task["activity"] = mapped_activity
                valid_tasks.append(task)
            
            result["household_tasks"] = valid_tasks
            return True, result
            
        except:
            if attempt < max_retries - 1:
                time.sleep(2)
    
    return False, {"household_tasks": []}


# ===== Phase 2b: Multi-Agent Negotiation =====
def negotiate_task_allocation_multiagent(
    agents: Dict[str, Agent],
    household_tasks: Dict,
    all_agents: Dict[str, Agent],
    max_rounds: int = 4 ##可以在这里设置最大对话轮数，我目前设置的是4
) -> tuple:
    tasks_list = household_tasks.get("household_tasks", [])
    if not tasks_list:
        return True, {
            "family_discussion": [],
            "negotiation_summary": "No tasks to allocate.",
            "task_allocations": []
        }
    
    member_ids = list(agents.keys())
    conversation = []
    
    print(f"      Multi-agent negotiation using Agent instances...")
    
    # Round 1: Initial proposals (每个Agent生成初始提案)
    for uid in member_ids:
        agent = agents[uid]
        statement = agent.propose_initial(tasks_list, all_agents)
        conversation.append({"speaker": uid, "statement": statement})
        print(f"         {uid}: {statement[:60]}...")
        
        # 更新对话历史
        for a in agents.values():
            a.update_conversation_history(conversation)
        
        time.sleep(0.5)
    
    # Rounds 2-N: Discussion (多轮回应)
    for round_num in range(2, max_rounds):
        for uid in member_ids:
            agent = agents[uid]
            statement = agent.respond(tasks_list, all_agents)
            
            if statement:
                conversation.append({"speaker": uid, "statement": statement})
                print(f"         {uid}: {statement[:60]}...")
                
                # 更新对话历史
                for a in agents.values():
                    a.update_conversation_history(conversation)
                
                time.sleep(0.5)
                
                ## 这个判断条件我在考虑要不要添加，目前先注释掉
                # if any(word in statement.lower() for word in ["deal", "sounds good", "agreed"]):
                #     if round_num >= 4:
                #         break
        
        # recent = [h["statement"] for h in conversation[-len(member_ids):]]
        # if all(any(w in s.lower() for w in ["agree", "deal", "good", "okay"]) for s in recent):
        #     print(f"      Consensus reached after round {round_num}!")
        #     break
    
    allocations = extract_allocations_from_conversation(
        agents[member_ids[0]].client,  
        member_ids, 
        tasks_list, 
        conversation
    )
    
    allocations_by_member = defaultdict(list)
    for alloc in allocations:
        assigned_to = alloc.get("assigned_to")
        if isinstance(assigned_to, list):
            assigned_to = assigned_to[0] if assigned_to else None
        if assigned_to and assigned_to in agents:
            allocations_by_member[assigned_to].append(alloc)
            agents[assigned_to].set_allocated_tasks(allocations_by_member[assigned_to])
            print(f"      → Agent {assigned_to}: Allocated task {alloc.get('task_id')}")
    
    summary = f"After {len(conversation)} exchanges, family reached consensus through multi-agent negotiation using Agent instances."
    
    return True, {
        "family_discussion": conversation,
        "negotiation_summary": summary,
        "task_allocations": allocations
    }


def extract_allocations_from_conversation(client: OpenAI, member_ids: List[str], tasks_list: List[Dict], conversation: List[Dict]) -> List[Dict]:
    conv_text = "\n".join([f"{h['speaker']}: {h['statement']}" for h in conversation])
    tasks_desc = "\n".join([f"- {t['task_id']}" for t in tasks_list])
    
    prompt = f"""Analyze this family conversation and extract who will do which tasks:

CONVERSATION:
{conv_text}

TASKS:
{tasks_desc}

MEMBERS: {', '.join(member_ids)}

Output JSON:
```json
{{
  "allocations": [
    {{
      "task_id": "shopping",
      "assigned_to": "30007884_1",
      "scheduled_time": {{"start": "18:00", "end": "19:00"}},
      "reasoning": "Volunteered to handle after work"
    }}
  ]
}}
```

Extract allocations:"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        result_text = response.choices[0].message.content.strip()
        
        if "```json" in result_text:
            json_start = result_text.find("```json") + 7
            json_end = result_text.find("```", json_start)
            result_text = result_text[json_start:json_end].strip()
        
        result = json.loads(result_text)
        return result.get("allocations", [])
    except:
        return [{
            "task_id": task["task_id"],
            "assigned_to": member_ids[i % len(member_ids)],
            "scheduled_time": {"start": "12:00", "end": "13:00"},
            "reasoning": "Default allocation"
        } for i, task in enumerate(tasks_list)]

# ===== Main Processing Function=====
def process_household_with_agent_based_negotiation(
    client: OpenAI,
    household_id: str,
    member_user_ids: List[str],
    persons_dict: Dict,
    households_dict: Dict
) -> Dict:
    print(f"\n{'='*70}")
    print(f" Household: {household_id}")
    print(f" Members: {len(member_user_ids)}")
    print(f"{'='*70}")
    
    household_info = extract_household_info(households_dict.get(household_id))
    member_infos = {uid: extract_person_info(persons_dict[uid]) for uid in member_user_ids}
    
    print(f"\n PHASE 0: Creating Agent instances for all household members...")
    print(f"{'─'*70}")
    agents = {}
    for uid in member_user_ids:
        other_members = {
            other_uid: member_infos[other_uid]
            for other_uid in member_user_ids
            if other_uid != uid
        }
        
        agent = Agent(
            agent_id=uid,
            person_info=member_infos[uid],
            household_info=household_info,
            other_members=other_members,
            openai_client=client
        )
        
        agents[uid] = agent
        print(f"  Agent {uid} ({member_infos[uid]['relationship']}, {member_infos[uid]['age_range']}) instantiated")
    
    print(f"\n All {len(agents)} Agent instances created and ready!")
    
    print(f"\n PHASE 1: Mandatory Activities")
    print(f"{'─'*70}")
    for idx, uid in enumerate(member_user_ids, 1):
        agent = agents[uid]
        print(f"  [{idx}/{len(member_user_ids)}] Agent {uid}...")
        success = agent.propose_mandatory_activities()
        if success:
            print(f"       {len(agent.mandatory_activities)} mandatory activity(ies) - State updated")
        else:
            print(f"       Failed to generate mandatory activities")
        time.sleep(1)
    
    print(f"\n PHASE 2: Household Maintenance & Agent-Based Negotiation")
    print(f"{'─'*70}")
    
    print(f"  Step 2a: Generating household tasks")
    success, household_tasks = propose_household_tasks(client, household_info, member_infos)
    if success:
        print(f"       {len(household_tasks.get('household_tasks', []))} task(s) generated")
    else:
        household_tasks = {"household_tasks": []}
    time.sleep(2)
    
    print(f"  Step 2b: Agent-based multi-agent negotiation")
    success, allocation_result = negotiate_task_allocation_multiagent(
        agents, household_tasks, agents  
    )
    if success:
        print(f"       {len(allocation_result.get('task_allocations', []))} task(s) allocated - States updated")
        print(f"      Dialogue exchanges: {len(allocation_result.get('family_discussion', []))}")
    else:
        allocation_result = {"family_discussion": [], "task_allocations": []}
    
    print(f"\n PHASE 3: Personal Activities")
    print(f"{'─'*70}")
    for idx, uid in enumerate(member_user_ids, 1):
        agent = agents[uid]
        print(f"  [{idx}/{len(member_user_ids)}] Agent {uid}...")
        try:
            success = agent.propose_personal_activities()
            if success:
                print(f"       {len(agent.personal_activities)} personal activity(ies) - State updated")
            else:
                print(f"       Failed to generate personal activities")
        except Exception as e:
            print(f"       ERROR in Phase 3 for {uid}: {e}")
        time.sleep(1)
    
    print(f"\n PHASE 4: Full Day Scheduling ")
    print(f"{'─'*70}")
    for idx, uid in enumerate(member_user_ids, 1):
        agent = agents[uid]
        print(f"  [{idx}/{len(member_user_ids)}] Agent {uid}...")
        success = agent.schedule_full_day()
        if success:
            print(f"       {len(agent.final_schedule)} activity segments - State updated")
        else:
            print(f"       Failed to generate full schedule")
        time.sleep(1)
    
    mandatory_by_member = {uid: {"mandatory_activities": agents[uid].mandatory_activities} for uid in member_user_ids}
    personal_by_member = {uid: {"personal_activities": agents[uid].personal_activities} for uid in member_user_ids}
    schedules = {uid: {"full_schedule": agents[uid].final_schedule} for uid in member_user_ids}
    print(f"  All results collected from {len(agents)} Agent instances")
    
    return {
        "household_id": household_id,
        "household_info": household_info,
        "members": member_infos,
        "phase1_mandatory": mandatory_by_member,
        "phase2_household_tasks": household_tasks,
        "phase2_allocation": allocation_result,
        "phase3_personal": personal_by_member,
        "phase4_schedules": schedules,
        "generation_time": datetime.now().isoformat(),
        "agent_based": True,  
        "agents_count": len(agents)
    }

def save_individual_trajectories(results: List[Dict], timestamp: str):
    os.makedirs(OUTPUT_TRAJECTORIES_DIR, exist_ok=True)
    
    all_trajectories = []
    
    for household_result in results:
        schedules = household_result.get("phase4_schedules", {})
        
        for user_id, schedule_data in schedules.items():
            full_schedule = schedule_data.get("full_schedule", [])
            
            if full_schedule:
                simplified_schedule = []
                for act in full_schedule:
                    simplified_schedule.append({
                        "activity": act.get("activity"),
                        "start_time": act.get("start_time"),
                        "end_time": act.get("end_time")
                    })
                
                trajectory = {
                    "user_id": user_id,
                    "schedule": simplified_schedule
                }
                
                all_trajectories.append(trajectory)
                print(f"    ✓ Collected trajectory for {user_id}")
    
    if all_trajectories:
        trajectory_filename = f"all_trajectories_{timestamp}.json"
        trajectory_filepath = os.path.join(OUTPUT_TRAJECTORIES_DIR, trajectory_filename)
        
        with open(trajectory_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_trajectories, f, indent=2, ensure_ascii=False)
        
        print(f"\n    ✓ Saved all trajectories → {trajectory_filename}")
    
    return len(all_trajectories)


def main():
    print("="*70)
    print(" MULTI-AGENT NEGOTIATION - REFACTORED WITH AGENT CLASS")
    print(" Each household member = Agent instance with encapsulated state")
    print("="*70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TRAJECTORIES_DIR, exist_ok=True)
    
    persons_list = load_json(PERSON_FILE)
    households_list = load_json(HOUSEHOLD_FILE)
    
    persons_dict = {p["user_id"]: p for p in persons_list if p.get("user_id")}
    households_dict = {h["household_id"]: h for h in households_list if h.get("household_id")}
    
    households_grouped = group_by_household(persons_list)
    multi_person = {hid: members for hid, members in households_grouped.items() 
                    if len(members) >= 2 and hid in households_dict}
    
    sample_n = 1  # 处理1个家庭
    selected = list(multi_person.items())[:sample_n]
    
    print(f"\n Processing {sample_n} household(s) with Agent-based approach...")
    
    client = create_openai_client()
    
    results = []
    for idx, (hid, members) in enumerate(selected, 1):
        try:
            result = process_household_with_agent_based_negotiation(
                client, hid, members, persons_dict, households_dict
            )
            results.append(result)
            print(f"\n Household {idx}/{len(selected)} completed")
            
            if idx < len(selected):
                time.sleep(3)
        except Exception as e:
            print(f"\n Error processing household {hid}: {e}")
            import traceback
            traceback.print_exc()
    
    if results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        output_file = os.path.join(OUTPUT_DIR, f"agent_based_negotiation_{timestamp}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        trajectory_count = save_individual_trajectories(results, timestamp)
        
        print(f"\n{'='*70}")
        print(f" GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"  Households processed: {len(results)}")
        print(f"  Full output: {output_file}")
        print(f"  Total trajectories: {trajectory_count} users in {OUTPUT_TRAJECTORIES_DIR}")
        print(f"  Method: Agent-based (each member = Agent instance)")
    else:
        print("\n No results to save.")


if __name__ == "__main__":
    main()
