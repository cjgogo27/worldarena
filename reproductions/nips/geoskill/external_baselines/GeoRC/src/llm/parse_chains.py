from typing import List
import re
def parse_reasoning_chains(file_path: str) -> List[str]:
    with open(file_path, 'r') as file:
        content = file.read()

    parts = re.split(r'\bRound \d+\b', content)
    
    if parts and not parts[0].strip():
        parts = parts[1:] # gets rid of empty elements
    res = []
    for part in parts:
        lines= part.split('\n') 
        filtered_lines = []
        for line in lines:
            if 'reasoning' in line.strip().lower():
                continue
            if 'conclusion' in line.strip().lower():
                break  # stop reading after finding conclusion
            filtered_lines.append(line.strip())
        
        res.append('\n'.join(filtered_lines))
    # get rid of reasoning and conclusion keywords in Joshua/Tejas chains
    return res

def parse_reasoning_chains_with_conclusion(file_path: str) -> List[str]:
    with open(file_path, 'r') as file:
        content = file.read()

    res = []
    chains = [paragraph.split("\n", 1)[-1] for paragraph in content.split("\nRound") if paragraph.strip()]
    for chain in chains:
        lines = chain.split("\n")
        valid_lines = []
        for line in lines:
            annotation = line.strip().lower()
            if annotation == 'reasoning' or annotation=='reasoning:' or annotation == 'conclusion' or annotation=='conclusion:':
                continue
            if line.strip():
                valid_lines.append(line.strip())
        res.append('\n'.join(valid_lines))
        # print(f"filtered chain is: \n {res[-1]}")
    # print(f"after removing annotations: {res}")
    return res