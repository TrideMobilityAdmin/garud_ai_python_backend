import re
from functools import lru_cache
from app.core.maintenance_config import ABBREVIATIONS_DICT, DIRECTIONAL_SENSITIVE
from app.utils.maintenance_utils import get_nested_task_mhs_lookup

# PRESERVED: Old code abbreviation logic with directional sensitivity
ABBR_PATTERNS = {}
for abbr, full in ABBREVIATIONS_DICT.items():
    if abbr not in DIRECTIONAL_SENSITIVE:
        ABBR_PATTERNS[abbr] = re.compile(r'\b' + re.escape(abbr) + r'\b', re.IGNORECASE)

# PRESERVED: Old code component extraction patterns
COMPILED_PATTERNS = [
    re.compile(r'^(.+?)\s*REPLACEMENT$', re.IGNORECASE),
    re.compile(r'^(.+?)\s*REM\s*&\s*INST(?:L|ALL|ATION)?$', re.IGNORECASE),
    re.compile(r'^(.+?)\s*R\s*&\s*I$', re.IGNORECASE),
    re.compile(r'^(.+?)\s*REMOVAL\s*(?:&|AND)\s*INSTALLATION$', re.IGNORECASE),
    re.compile(r'^(.+?)\s*REMOVE\s*(?:&|AND)\s*INSTALL(?:ATION)?$', re.IGNORECASE),
]

def normalize_task_description(text):
    if not text:
        return ""
    text = str(text).strip()
    return re.sub(r'\s+', ' ', text)

@lru_cache(maxsize=20000)
def expand_abbreviations_cached(text):
    """PRESERVED: Conservative abbreviation expansion from old code"""
    if not text:
        return ""
    
    text = normalize_task_description(text)
    task_norm = text.lower()
    
    # Only expand non-directional abbreviations (preserved from old code)
    for abbr, pattern in ABBR_PATTERNS.items():
        task_norm = pattern.sub(ABBREVIATIONS_DICT[abbr], task_norm)
    
    return re.sub(r'\s+', ' ', task_norm).strip()

@lru_cache(maxsize=1000)
def map_special_task_aircraft_type(user_type):
    """PRESERVED: Old code aircraft type mapping"""
    if 'A320' in str(user_type) or 'A321' in str(user_type):
        return 'A320 & B737'
    return user_type

@lru_cache(maxsize=5000)
def extract_component_cached(task_desc):
    """PRESERVED: Old code component extraction with fixes"""
    if not task_desc:
        return None
    
    task_upper = task_desc.upper().strip()
    
    # Try compiled patterns first (from old code)
    for pattern in COMPILED_PATTERNS:
        match = pattern.match(task_upper)
        if match:
            component = match.group(1).strip()
            # PRESERVED: Avoid generic ENGINE matches for specific components
            if component == 'ENGINE' and any(specific in task_upper for specific in [
                'FIRE BOTTLE', 'INLET COWL', 'FAN COWL', 'EXHAUST', 'FAN BLADES', 'THRUST REVERSER'
            ]):
                continue
            return component
    
    return task_desc.strip()

@lru_cache(maxsize=5000)
def find_removal_installation_cached(component, aircraft_type):
    """PRESERVED: Old code removal/installation sum logic"""
    nested_task_mhs_lookup = get_nested_task_mhs_lookup()
    
    if aircraft_type not in nested_task_mhs_lookup:
        return [], [], 0.0
    
    type_lookup = nested_task_mhs_lookup[aircraft_type]
    component_upper = component.upper()
    
    removal_tasks = []
    installation_tasks = []
    removal_mhs = []
    installation_mhs = []
    
    for task_key, mhs in type_lookup.items():
        task_upper = task_key.upper()
        
        if component_upper in task_upper:
            if any(rem_word in task_upper for rem_word in ['REMOVAL', 'REMOVE', 'REM']):
                if not any(inst_word in task_upper for inst_word in ['INSTALLATION', 'INSTALL', 'INST']):
                    removal_tasks.append(task_key)
                    removal_mhs.append(mhs)
            
            if any(inst_word in task_upper for inst_word in ['INSTALLATION', 'INSTALL', 'INST']):
                if not any(rem_word in task_upper for rem_word in ['REMOVAL', 'REMOVE', 'REM']):
                    installation_tasks.append(task_key)
                    installation_mhs.append(mhs)
    
    total_removal = sum(removal_mhs) if removal_mhs else 0
    total_installation = sum(installation_mhs) if installation_mhs else 0
    total_mhs = total_removal + total_installation
    
    return [removal_tasks, installation_tasks], [total_removal, total_installation], total_mhs