import os
import re

class MaintenanceConfig:
    """Configuration for maintenance prediction service"""
    
    # Directories
    MODEL_DIR = 'model'
    MASTER_DATA_MAP_FILE = os.path.join(MODEL_DIR, 'master_data_map.pkl')
    
    # Performance settings
    MAX_WORKERS = min(32, (os.cpu_count() or 1) * 4)
    
    # --- FIXED: More specific component extraction patterns ---
    COMPILED_PATTERNS = [
        # Only match exact patterns, avoid generic ENGINE matches
        re.compile(r'^(.+?)\s*REPLACEMENT$', re.IGNORECASE),
        re.compile(r'^(.+?)\s*REM\s*&\s*INST(?:L|ALL|ATION)?$', re.IGNORECASE),
        re.compile(r'^(.+?)\s*R\s*&\s*I$', re.IGNORECASE),
        re.compile(r'^(.+?)\s*REMOVAL\s*(?:&|AND)\s*INSTALLATION$', re.IGNORECASE),
        re.compile(r'^(.+?)\s*REMOVE\s*(?:&|AND)\s*INSTALL(?:ATION)?$', re.IGNORECASE),
    ]
    
    # Enhanced technical ID patterns for exact matching
    TECHNICAL_ID_PATTERNS = {
        'EO': re.compile(r'EO[_\-]?([A-Z0-9_\-]+)', re.IGNORECASE),
        'EWI': re.compile(r'EWI[_\-]?([A-Z0-9_\-]+)', re.IGNORECASE),
        'REF_TASK': re.compile(r'REF\s+TASK:\s*(\d+)', re.IGNORECASE),
        'AD': re.compile(r'(?:EASA\s+)?AD\s+([0-9\-]+(?:\s+CORRECTION)?)', re.IGNORECASE),
        'SB': re.compile(r'SB\s+([0-9\-]+)', re.IGNORECASE)
    }
    
    # FIXED: Enhanced abbreviations with better directional handling
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
        'ohb': 'overhead bin',
        'lav': 'lavatory'
    }
    
    # FIXED: Don't expand directional abbreviations when they might be significant
    DIRECTIONAL_SENSITIVE = ['lh', 'rh', 'fwd', 'aft']
    
    # Create abbreviation patterns
    ABBR_PATTERNS = {}
    
    @classmethod
    def initialize_patterns(cls):
        """Initialize abbreviation patterns"""
        for abbr, full in cls.ABBREVIATIONS_DICT.items():
            # Skip expanding directional terms that might be significant for matching
            if abbr not in cls.DIRECTIONAL_SENSITIVE:
                cls.ABBR_PATTERNS[abbr] = re.compile(r'\b' + re.escape(abbr) + r'\b', re.IGNORECASE)