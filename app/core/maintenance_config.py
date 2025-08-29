import os

# --- Configuration ---
MODEL_DIR = 'model'  # Use the same as model building code
MASTER_DATA_MAP_FILE = os.path.join(MODEL_DIR, 'master_data_map.pkl')
MAX_WORKERS = min(32, (os.cpu_count() or 1) * 4)

# --- Enhanced component keywords from new code ---
COMPONENT_KEYWORDS = {
    'cockpit_seat': ['cockpit seat', 'pilot seat', 'flight deck seat', 'crew seat', 'observer seat'],
    'passenger_seat': ['passenger seat', 'pax seat', 'cabin seat'],
    'external_wash': ['external wash', 'exterior wash', 'aircraft wash', 'exterior cleaning'],
    'technical_external_wash': ['technical external wash', 'technical wash', 'technical exterior'],
    'cockpit_cleaning': ['cockpit cleaning', 'flight deck cleaning', 'pilot cabin cleaning'],
    'cabin_cleaning': ['cabin cleaning', 'passenger cabin cleaning'],
    'cabin_deep_cleaning': ['cabin deep cleaning', 'deep cabin cleaning', 'intensive cabin'],
    'cargo_cleaning': ['cargo cleaning', 'cargo hold cleaning'],
    'fwd_cargo_cleaning': ['fwd cargo cleaning', 'forward cargo cleaning'],
    'aft_cargo_cleaning': ['aft cargo cleaning', 'rear cargo cleaning'],
    'oven': ['oven', 'cooking oven', 'galley oven'],
    'coffee_maker': ['coffee maker', 'coffee machine', 'coffee brewing', 'coffee dispenser'],
    'water_boiler': ['water boiler', 'boiler'],
    'water_tank': ['water tank', 'potable water tank'],
    'cabin_panels': ['cabin panel', 'interior panel', 'cabin trim', 'sidewall panel'],
    'cargo_panels': ['cargo panel', 'cargo lining'],
    'engine_chemical_wash': ['engine chemical wash', 'chemical flush wash', 'internal & external chemical wash', 
                             'dual engine wash', 'both engine wash', 'engine wash', 'chemical wash'],
    # Add more from the new code...
}

# --- Component exclusion rules from new code ---
COMPONENT_EXCLUSION_RULES = {
    'cockpit_seat': {
        'exclude_if_target_has': ['passenger', 'pax', 'cabin seat'],
        'must_have_one_of': ['cockpit', 'pilot', 'crew', 'observer', 'flight deck'],
        'boost_if_has': ['cockpit', 'pilot', 'crew', 'observer']
    },
    'passenger_seat': {
        'exclude_if_target_has': ['cockpit', 'pilot', 'crew', 'observer', 'flight deck'],
        'must_have_one_of': ['passenger', 'pax', 'cabin seat'],
        'boost_if_has': ['passenger', 'pax', 'cabin']
    },
    # Add more exclusion rules...
}

# --- Abbreviations dictionary with directional sensitivity ---
ABBREVIATIONS_DICT = {
    'r&i': 'remove and install',
    'rem': 'remove',
    'instl': 'install',
    'instsl': 'install',
    'insp': 'inspection',
    'rep': 'repair',
    'chk': 'check',
    'tst': 'test',
    'eng': 'engine',
    'a/c': 'aircraft',
    'comp': 'component',
    'assy': 'assembly',
    'maint': 'maintenance',
    'fwd': 'forward',
    'aft': 'aft',
    'lh': 'left hand',
    'rh': 'right hand',
    'pax': 'passenger',
    'seat': 'seat',
    'galley': 'galley',
    'cargo': 'cargo',
    'panel': 'panel',
    'closet': 'closet',
    'ntf': 'ntf',
    'pcu': 'passenger control unit',
    'oxg': 'oxygen generator',
    'mlg': 'main landing gear',
    'nlg': 'nose landing gear',
    'apu': 'auxiliary power unit',
    't/rev': 'thrust reverser',
    'cfm': 'cfm',
    'v2500': 'v2500',
    'cna': 'cna',
    'de-installation': 'removal',
    'dismantling': 'removal',
    'take-off': 'removal',
    'takeoff': 'removal',
    'take off': 'removal'
}

# Directional sensitivity
DIRECTIONAL_SENSITIVE = ['lh', 'rh', 'fwd', 'aft']