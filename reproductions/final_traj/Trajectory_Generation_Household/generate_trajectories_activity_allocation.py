
import json
import openai
import time
import os
from datetime import datetime
from collections import defaultdict

API_KEY = "YOUR_API_KEY_HERE" 
BASE_URL = "https://api.openai.com/v1"  
MODEL = "gpt-4o"  

PERSON_FILE = r"E:\FrankYcj\FinalTraj\California\processed_data\california_person_static.json"
HOUSEHOLD_FILE = r"E:\FrankYcj\FinalTraj\California\processed_data\california_household_static.json"
OUTPUT_DIR = r"E:\FrankYcj\FinalTraj\Trajectory_Generation_Household\output"

ALLOWED_ACTIVITIES = [
    "home", "work", "education", "shopping", "service",
    "medical", "dine_out", "socialize", "exercise", "dropoff_pickup"
]


def create_openai_client():
    return openai.OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=30.0)


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_household_id(user_id):
    return user_id.split('_')[0] if '_' in user_id else user_id


def group_by_household(person_list):
    households = defaultdict(list)
    for p in person_list:
        uid = p.get("user_id")
        if uid:
            hid = extract_household_id(uid)
            households[hid].append(uid)
    return households


def extract_person_info(person_data):
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


def extract_household_info(household_data):
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


# ===== Phase 1: Mandatory Activities =====
def propose_mandatory_activities(client, person_info, household_info, max_retries=3):
    prompt = f"""You are Agent {person_info['user_id']} proposing your MANDATORY activities.

Personal Profile:
- Age: {person_info['age_range']}
- Gender: {person_info['gender']}
- Employment: {person_info['employment_status']}
- Work Schedule: {person_info['work_schedule']}
- Occupation: {person_info['occupation']}
- Primary Activity: {person_info['primary_activity']}
- Work From Home: {person_info['work_from_home']}
- Distance to Work: {person_info['distance_to_work_miles']} miles
- Driver: {person_info['driver_on_travel_day']}

Household Context:
- Household Size: {household_info['household_size']} people
- Young Children: {household_info['young_children_count']}
- Vehicles: {household_info['vehicle_count']}
- Income: {household_info['household_income']}

Task: Propose your MANDATORY activities (work/education) ONLY.

Output JSON format:
```json
{{
  "mandatory_activities": [
    {{
      "activity": "work",
      "start_time": "08:30",
      "end_time": "17:15",
      "uses_vehicle": true,
      "reasoning": "Full-time worker, commute required"
    }}
  ]
}}
```

If no mandatory activities, return: {{"mandatory_activities": []}}

Generate your mandatory activities:"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
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
            return True, result
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return False, {"error": str(e), "mandatory_activities": []}
    
    return False, {"mandatory_activities": []}


# ===== Phase 2: Household Maintenance Activities =====
## ==== Phase 2a: Propose Household Tasks =====
def propose_household_tasks(client, household_info, member_infos, max_retries=3):
    members_summary = "\n".join([
        f"- {uid}: Age {info['age_range']}, {info['employment_status']}, Driver: {info['driver_on_travel_day']}"
        for uid, info in member_infos.items()
    ])
    
    prompt = f"""You are the Household Manager for household {household_info['household_id']}.

Household Profile:
- Size: {household_info['household_size']} people
- Adults: {household_info['adult_count']}
- Young Children: {household_info['young_children_count']}
- Vehicles: {household_info['vehicle_count']}
- Income: {household_info['household_income']}
- Urban Area: {household_info['urban_area']}

Members:
{members_summary}

Task: Generate household maintenance activities that need to be done today.

Activity Types to Consider:
1. dropoff_pickup: ONLY if young_children_count > 0 - school dropoff/pickup for children
2. shopping: Daily groceries and errands (every household)
3. service: Bank, post office, government services (occasional)
4. medical: Doctor appointments (occasional)

Rules:
1. **CRITICAL**: "dropoff_pickup" ONLY if household has young children ({household_info['young_children_count']} children)
   - If young_children_count = 0, DO NOT generate dropoff_pickup tasks
   - If young_children_count > 0, generate morning dropoff (07:30-08:30) and/or afternoon pickup (15:00-16:00)
2. Shopping is MEDIUM priority (every household needs)
3. Consider household income and urban area for activity frequency
4. Each task should have: activity type, preferred time window, duration, priority

Output JSON format:
```json
{{
  "household_tasks": [
    {{
      "task_id": "pickup_morning",
      "activity": "dropoff_pickup",
      "time_window": {{"start": "07:30", "end": "08:30"}},
      "duration_minutes": 30,
      "uses_vehicle": true,
      "priority": "high",
      "reasoning": "School dropoff for young children"
    }},
    {{
      "task_id": "shopping_daily",
      "activity": "shopping",
      "time_window": {{"start": "09:00", "end": "20:00"}},
      "duration_minutes": 60,
      "uses_vehicle": true,
      "priority": "medium",
      "reasoning": "Daily grocery shopping"
    }}
  ]
}}
```

Generate household tasks:"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a household manager generating daily tasks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
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
            return True, result
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return False, {"error": str(e), "household_tasks": []}
    
    return False, {"household_tasks": []}


## ===== Phase 2b: Task Allocation Negotiation =====
def negotiate_task_allocation(client, household_info, member_infos, mandatory_activities, household_tasks, max_retries=3):
    members_status = []
    for uid, info in member_infos.items():
        mand = mandatory_activities.get(uid, {}).get("mandatory_activities", [])
        mand_summary = f"Work {mand[0]['start_time']}-{mand[0]['end_time']}" if mand else "No mandatory activities"
        members_status.append(f"- {uid}: {info['age_range']}, {info['employment_status']}, Driver: {info['driver_on_travel_day']}, {mand_summary}")
    
    tasks_summary = "\n".join([
        f"- {t['task_id']}: {t['activity']} ({t['duration_minutes']}min, priority: {t['priority']})"
        for t in household_tasks.get("household_tasks", [])
    ])
    
    members_details = "\n\n".join([
        f"**{uid} ({info['relationship']}):**\n"
        f"  - Gender: {info['gender']}, Age: {info['age_range']}\n"
        f"  - Employment: {info['employment_status']}, Schedule: {info['work_schedule']}\n"
        f"  - Occupation: {info['occupation']}\n"
        f"  - Driver: {info['driver_on_travel_day']}\n"
        f"  - Work from home: {info['work_from_home']}\n"
        f"  - Mandatory activities: {', '.join([a['activity'] + ' ' + a['start_time'] + '-' + a['end_time'] for a in mandatory_activities.get(uid, {}).get('mandatory_activities', [])]) if mandatory_activities.get(uid, {}).get('mandatory_activities') else 'None'}"
        for uid, info in member_infos.items()
    ])
    
    prompt = f"""You are simulating a REAL family discussion for household {household_info['household_id']}.

Household Context:
- Household Size: {household_info['household_size']} people
- Vehicles: {household_info['vehicle_count']} (shared resource)
- Children: {household_info['young_children_count']}
- Income: {household_info['household_income']}

Family Members:
{members_details}

Household Tasks That Need To Be Done Today:
{tasks_summary}

Your Task: Simulate a REALISTIC family conversation where members discuss, negotiate, and decide who does what.

Conversation Guidelines:
1. Each member speaks from their perspective (use first person)
2. Show negotiation dynamics:
   - Initial proposals ("I can handle the shopping after work")
   - Concerns raised ("But I have to work until 5pm, that's too late")
   - Compromises offered ("How about you do pickup and I'll do shopping?")
   - Trade-offs ("If you do the morning pickup, I'll cook dinner")
   - Final agreement reached

3. Consider real constraints:
   - Work schedules (who's more flexible?)
   - Driver status (who can use the car?)
   - Childcare responsibilities
   - Fairness (not putting everything on one person)

4. Realistic dialogue flow:
   - Someone suggests initial plan
   - Others respond with concerns/alternatives
   - Discussion of timing conflicts
   - Negotiation and compromise
   - Final consensus

Example conversation format:
```
Member A: "I think I can handle the shopping after work around 6pm since I finish at 5."
Member B: "That works, but what about the morning school pickup? I have an early meeting."
Member A: "I can't do the pickup, I start work at 8:30 and need to leave by 7:45."
Member B: "Okay, how about this - I'll do the morning pickup if you handle both shopping and dinner prep?"
Member A: "That sounds fair. I'll do shopping at 6pm and then cook. You do the 8am pickup."
Member B: "Deal. And I'll help with evening cleanup to balance things out."
```

Output JSON format:
```json
{{
  "family_discussion": [
    {{"speaker": "30007884_1", "statement": "I can do the shopping after I get off work at 5pm, maybe around 6?"}},
    {{"speaker": "30007884_2", "statement": "That works for me. But what about the morning school dropoff at 8am?"}},
    {{"speaker": "30007884_1", "statement": "I can't do mornings - I need to leave for work by 7:30. Can you handle it?"}},
    {{"speaker": "30007884_2", "statement": "Sure, I'll do the morning dropoff since I have a more flexible morning. You take the shopping then."}},
    {{"speaker": "30007884_1", "statement": "Deal! Thanks for being flexible with the morning."}}
  ],
  "negotiation_summary": "Family discussed the tasks. Member 1 has rigid morning schedule due to long commute, so Member 2 agreed to handle morning pickup. In exchange, Member 1 takes on the evening shopping task. Both feel this is a fair distribution.",
  "task_allocations": [
    {{
      "task_id": "pickup_morning",
      "assigned_to": "30007884_2",
      "scheduled_time": {{"start": "07:45", "end": "08:15"}},
      "reasoning": "Member 2 has more flexible morning schedule and volunteered after discussion"
    }},
    {{
      "task_id": "shopping_daily",
      "assigned_to": "30007884_1",
      "scheduled_time": {{"start": "18:00", "end": "19:00"}},
      "reasoning": "Member 1 agreed to handle evening shopping as a trade-off for not doing morning pickup"
    }}
  ]
}}
```

Now simulate the REAL family discussion and negotiation:"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are coordinating household task allocation through negotiation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=1000
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
            return True, result
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return False, {"error": str(e), "negotiation_reasoning": "", "task_allocations": []}
    
    return False, {"negotiation_reasoning": "", "task_allocations": []}


# ===== Phase 3: Personal Activities =====
def propose_personal_activities(client, person_info, household_info, mandatory_activities, allocated_tasks, max_retries=3):
    mand_summary = "\n".join([
        f"- {a['activity']}: {a['start_time']}-{a['end_time']}"
        for a in mandatory_activities.get("mandatory_activities", [])
    ])
    
    task_summary = "\n".join([
        f"- {t['task_id']}: {t['scheduled_time']['start']}-{t['scheduled_time']['end']}"
        for t in allocated_tasks
    ]) if allocated_tasks else "None"
    
    prompt = f"""You are Agent {person_info['user_id']} proposing your PERSONAL activities.

Your Profile:
- Age: {person_info['age_range']}
- Gender: {person_info['gender']}
- Employment: {person_info['employment_status']}
- Education: {person_info['education']}
- Income Context: {household_info['household_income']}

Your Already Scheduled Activities:

Mandatory Activities:
{mand_summary if mand_summary else "None"}

Allocated Household Tasks:
{task_summary}

Task: Propose personal activities to fill remaining time.

Personal Activity Types:
- exercise: Gym, sports, physical activity
- socialize: Visiting friends, social events
- dine_out: Eating at restaurants
- medical: Doctor visits (if needed)
- service: Personal errands (if needed)

Rules:
1. Consider your age, employment, and household income
2. Activities should be realistic (not too many)
3. Times should NOT overlap with mandatory or household tasks

Output JSON format:
```json
{{
  "personal_activities": [
    {{
      "activity": "exercise",
      "start_time": "18:30",
      "end_time": "19:30",
      "uses_vehicle": false,
      "reasoning": "Evening workout after work"
    }},
    {{
      "activity": "socialize",
      "start_time": "20:00",
      "end_time": "21:30",
      "uses_vehicle": false,
      "reasoning": "Evening social time with friends"
    }}
  ]
}}
```

Generate your personal activities:"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are proposing personal activities for remaining time."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=600
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
            return True, result
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return False, {"error": str(e), "personal_activities": []}
    
    return False, {"personal_activities": []}


# ===== Phase 4: Activity Scheduling =====
def schedule_full_day(client, person_info, mandatory_activities, allocated_tasks, personal_activities, max_retries=3):
    mand = mandatory_activities.get("mandatory_activities", [])
    personal = personal_activities.get("personal_activities", [])
    
    all_activities = {
        "mandatory": mand,
        "household_tasks": allocated_tasks,
        "personal": personal
    }
    
    prompt = f"""You are the Activity Scheduler for Agent {person_info['user_id']}.

All Proposed Activities:
```json
{json.dumps(all_activities, indent=2)}
```

Your Task: Create a COMPLETE 24-hour schedule (00:00 to 24:00) with NO GAPS.

Rules:
1. Start at 00:00, end at 24:00
2. NO time gaps - every minute must be accounted for
3. Fill gaps with "home" activity
4. Keep mandatory activities intact (work/education)
5. Keep household task times intact
6. Fit personal activities in remaining time
7. If conflicts exist, adjust personal activity times slightly
8. Use ONLY these activities: {', '.join(ALLOWED_ACTIVITIES)}

CRITICAL ACTIVITY RULES:
- "dropoff_pickup": ONLY for dropping off/picking up children (if household has young_children > 0)
- "home": Use for time at home, including preparing for work, commuting time, resting, sleeping
- DO NOT use "dropoff_pickup" for personal commuting to work 

Output JSON format:
```json
{{
  "scheduling_reasoning": "Brief explanation of how you organized the day",
  "full_schedule": [
    {{"start_time": "00:00", "end_time": "08:00", "activity": "home"}},
    {{"start_time": "08:00", "end_time": "17:00", "activity": "work"}},
    {{"start_time": "17:00", "end_time": "18:00", "activity": "shopping"}},
    {{"start_time": "18:00", "end_time": "19:00", "activity": "exercise"}},
    {{"start_time": "19:00", "end_time": "24:00", "activity": "home"}}
  ]
}}
```
Create the full schedule:"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are scheduling a complete 24-hour day with no gaps."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1200
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
            
            # Validate schedule
            schedule = result.get("full_schedule", [])
            if schedule and schedule[0]["start_time"] in ["0:00", "00:00"] and schedule[-1]["end_time"] == "24:00":
                return True, result
            else:
                if attempt < max_retries - 1:
                    continue
                else:
                    return False, {"error": "Invalid schedule format", "full_schedule": []}
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return False, {"error": str(e), "scheduling_reasoning": "", "full_schedule": []}
    
    return False, {"scheduling_reasoning": "", "full_schedule": []}


# ===== Main Processing Function =====
def process_household_with_phases(client, household_id, member_ids, persons_dict, households_dict):
    print(f"\n{'='*70}")
    print(f" Household: {household_id}")
    print(f" Members: {len(member_ids)}")
    print(f"{'='*70}")
    
    household_info = extract_household_info(households_dict.get(household_id))
    member_infos = {uid: extract_person_info(persons_dict[uid]) for uid in member_ids}
    
    # ===== PHASE 1: Mandatory Activities =====
    print(f"\n PHASE 1: Propose Mandatory Activities")
    print(f"{'─'*70}")
    
    mandatory_by_member = {}
    for idx, uid in enumerate(member_ids, 1):
        print(f"  [{idx}/{len(member_ids)}] Agent {uid} proposing mandatory activities...")
        success, result = propose_mandatory_activities(client, member_infos[uid], household_info)
        if success:
            mandatory_by_member[uid] = result
            mand_count = len(result.get("mandatory_activities", []))
            print(f"       {mand_count} mandatory activity(ies)")
        else:
            print(f"       Failed: {result.get('error', 'Unknown')}")
            mandatory_by_member[uid] = {"mandatory_activities": []}
        time.sleep(1)
    
    # ===== PHASE 2: Household Tasks & Allocation =====
    print(f"\n PHASE 2: Household Maintenance & Allocation")
    print(f"{'─'*70}")
    
    print(f"  Step 2a: Generating household tasks...")
    success, household_tasks = propose_household_tasks(client, household_info, member_infos)
    if success:
        task_count = len(household_tasks.get("household_tasks", []))
        print(f"       {task_count} household task(s) generated")
    else:
        print(f"       Failed: {household_tasks.get('error', 'Unknown')}")
        household_tasks = {"household_tasks": []}
    time.sleep(2)
    
    print(f"  Step 2b: Simulating family discussion and negotiation...")
    success, allocation_result = negotiate_task_allocation(
        client, household_info, member_infos, mandatory_by_member, household_tasks
    )
    if success:
        alloc_count = len(allocation_result.get("task_allocations", []))
        discussion = allocation_result.get("family_discussion", [])
        print(f"       {alloc_count} task(s) allocated after discussion")
        print(f"      Family discussion ({len(discussion)} exchanges):")
        for exchange in discussion[:3]:  
            speaker_id = exchange.get("speaker", "Unknown")
            statement = exchange.get("statement", "")
            print(f"          {speaker_id}: {statement[:60]}...")
        if len(discussion) > 3:
            print(f"         ... and {len(discussion) - 3} more exchanges")
        print(f"      Summary: {allocation_result.get('negotiation_summary', '')[:100]}...")
    else:
        print(f"       Failed: {allocation_result.get('error', 'Unknown')}")
        allocation_result = {"task_allocations": [], "family_discussion": [], "negotiation_summary": ""}
    time.sleep(2)
    
    # Group allocations by member
    allocations_by_member = defaultdict(list)
    for alloc in allocation_result.get("task_allocations", []):
        assigned_to = alloc.get("assigned_to")
        if assigned_to:
            allocations_by_member[assigned_to].append(alloc)
    
    # ===== PHASE 3: Propose Personal Activities =====
    print(f"\n PHASE 3: Propose Personal Activities")
    print(f"{'─'*70}")
    
    personal_by_member = {}
    for idx, uid in enumerate(member_ids, 1):
        print(f"  [{idx}/{len(member_ids)}] Agent {uid} proposing personal activities...")
        allocated = allocations_by_member.get(uid, [])
        success, result = propose_personal_activities(
            client, member_infos[uid], household_info, 
            mandatory_by_member[uid], allocated
        )
        if success:
            personal_count = len(result.get("personal_activities", []))
            personal_by_member[uid] = result
            print(f"       {personal_count} personal activity(ies)")
        else:
            print(f"       Failed: {result.get('error', 'Unknown')}")
            personal_by_member[uid] = {"personal_activities": []}
        time.sleep(1)
    
    # ===== PHASE 4: Activity Scheduling =====
    print(f"\n PHASE 4: Activity Scheduling (Full Day)")
    print(f"{'─'*70}")
    
    schedules = {}
    for idx, uid in enumerate(member_ids, 1):
        print(f"  [{idx}/{len(member_ids)}] Scheduling full day for {uid}...")
        allocated = allocations_by_member.get(uid, [])
        success, result = schedule_full_day(
            client, member_infos[uid],
            mandatory_by_member[uid], allocated, personal_by_member[uid]
        )
        if success:
            schedule_count = len(result.get("full_schedule", []))
            schedules[uid] = result
            print(f"       {schedule_count} activity segments")
        else:
            print(f"       Failed: {result.get('error', 'Unknown')}")
            schedules[uid] = {"full_schedule": []}
        time.sleep(1)

    return {
        "household_id": household_id,
        "household_info": household_info,
        "members": member_infos,
        "phase1_mandatory": mandatory_by_member,
        "phase2_household_tasks": household_tasks,
        "phase2_allocation": allocation_result,
        "phase3_personal": personal_by_member,
        "phase4_schedules": schedules,
        "generation_time": datetime.now().isoformat()
    }

def main():
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    persons_list = load_json(PERSON_FILE)
    households_list = load_json(HOUSEHOLD_FILE)
    
    persons_dict = {p["user_id"]: p for p in persons_list if p.get("user_id")}
    households_dict = {h["household_id"]: h for h in households_list if h.get("household_id")}
    
    households_grouped = group_by_household(persons_list)
    multi_person = {hid: members for hid, members in households_grouped.items() 
                    if len(members) >= 2 and hid in households_dict}

    sample_n = 1  # Process 1 household
    selected = list(multi_person.items())[:sample_n]
    
    print(f"\n Processing {sample_n} households...")
    
    client = create_openai_client()
    
    results = []
    for idx, (hid, members) in enumerate(selected, 1):
        
        try:
            result = process_household_with_phases(client, hid, members, persons_dict, households_dict)
            results.append(result)
            
            print(f"\n Household {idx}/{len(selected)} completed")
            
            if idx < len(selected):
                time.sleep(3)
                
        except Exception as e:
            print(f"\n Error processing household {hid}: {str(e)}")
            continue
    
    # Save final results (only once)
    if results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_DIR, f"activity_allocation_{timestamp}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n\n{'='*70}")
        print(f" GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"  Households processed: {len(results)}")
        print(f"  Total members: {sum(len(r['members']) for r in results)}")
        print(f"  Output file: {output_file}")
    else:
        print(f"\n\n{'='*70}")
        print(f"  NO RESULTS GENERATED")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
