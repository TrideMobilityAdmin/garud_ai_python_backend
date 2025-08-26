import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import lru_cache
from typing import List, Dict, Any, Tuple
from app.core.maintenance_config import MaintenanceConfig
from app.utils.maintenance_utils import MaintenanceUtils

class MaintenanceHelperService:
    """Helper service for maintenance prediction tasks"""
    
    def __init__(self):
        self.config = MaintenanceConfig()
        self.config.initialize_patterns()
        self.utils = MaintenanceUtils()
    
    @lru_cache(maxsize=20000)
    def expand_abbreviations_cached(self, text):
        """FIXED: More conservative abbreviation expansion to preserve directional info"""
        if not text:
            return ""
        
        text = self.utils.normalize_task_description(text)
        task_norm = text.lower()
        
        # Only expand non-directional abbreviations
        for abbr, pattern in self.config.ABBR_PATTERNS.items():
            task_norm = pattern.sub(self.config.ABBREVIATIONS_DICT[abbr], task_norm)
        
        return re.sub(r'\s+', ' ', task_norm).strip()
    
    @lru_cache(maxsize=1000)
    def map_special_task_aircraft_type(self, user_type):
        """Map user aircraft type to special task aircraft type"""
        if 'A320' in str(user_type) or 'A321' in str(user_type):
            return 'A320 & B737'
        return user_type
    
    @lru_cache(maxsize=5000)
    def extract_technical_identifiers(self, description):
        """Extract technical identifiers from description"""
        identifiers = {}
        description_str = str(description)
        
        for id_type, pattern in self.config.TECHNICAL_ID_PATTERNS.items():
            matches = pattern.findall(description_str)
            if matches:
                identifiers[id_type] = matches[0] if len(matches) == 1 else matches
        
        return identifiers
    
    @lru_cache(maxsize=5000)
    def has_technical_numbers(self, description):
        """Check if description has technical numbers"""
        description_str = str(description).upper()
        return bool(re.search(r'(?:EO|EWI|REF\s+TASK|AD|SB)[_\-\s]*[0-9A-Z]', description_str))
    
    @lru_cache(maxsize=5000)
    def extract_component_cached(self, task_desc):
        """FIXED: More precise component extraction - EXACT MATCH TO ORIGINAL"""
        if not task_desc:
            return None
        
        task_upper = task_desc.upper().strip()
        
        # Try compiled patterns first
        for pattern in self.config.COMPILED_PATTERNS:
            match = pattern.match(task_upper)
            if match:
                component = match.group(1).strip()
                # FIXED: Avoid generic ENGINE matches for specific components
                if component == 'ENGINE' and any(specific in task_upper for specific in [
                    'FIRE BOTTLE', 'INLET COWL', 'FAN COWL', 'EXHAUST', 'FAN BLADES', 'THRUST REVERSER'
                ]):
                    continue
                return component
        
        return task_desc.strip()
    
    @lru_cache(maxsize=5000)
    def find_removal_installation_cached(self, component, aircraft_type, nested_task_mhs_lookup):
        """Find removal and installation tasks for a component"""
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
    
    def find_replacement_tasks_with_sum(self, query_desc, aircraft_type, special_tasks_data_map):
        """
        ORIGINAL REPLACEMENT LOGIC: Find tasks containing REPLACEMENT keyword and sum their MHS
        This matches the exact logic from the original api.py
        """
        mapped_type = self.map_special_task_aircraft_type(aircraft_type)
        if mapped_type not in special_tasks_data_map:
            return None
        
        query_upper = query_desc.upper().strip()
        
        # Check if query contains REPLACEMENT keyword
        if 'REPLACEMENT' not in query_upper:
            return None
        
        # Extract component from query
        component = self.extract_component_cached(query_desc)
        if not component:
            return None
        
        component_upper = component.upper()
        matching_tasks = []
        total_mhs = 0.0
        
        # Find all tasks that match the component and contain replacement-related keywords
        for task_data in special_tasks_data_map[mapped_type]:
            task_desc_upper = task_data['task_description_upper']
            
            # Check if task contains the component and replacement keywords
            if (component_upper in task_desc_upper and 
                any(keyword in task_desc_upper for keyword in ['REPLACEMENT', 'REPLACE', 'REM', 'INST'])):
                matching_tasks.append(task_data)
                total_mhs += task_data['total_mhs']
        
        if matching_tasks and total_mhs > 0:
            # Create description of what was combined
            task_descriptions = [task['task_description'] for task in matching_tasks]
            combined_description = f"Sum of {len(matching_tasks)} replacement tasks for {component}"
            
            return {
                'source': 'special_tasks',
                'task_description': combined_description,
                'total_mhs': total_mhs,
                'similarity': 1.0,
                'match_type': 'replacement_sum',
                'combined_tasks': task_descriptions,
                'component': component
            }
        
        return None
    
    @staticmethod
    def ultra_fast_similarity_search(query_embedding, matrix_key, precomputed_embeddings, embedding_indices, threshold=0.75, top_k=5):
        """PERFORMANCE: Ultra-fast vectorized similarity search"""
        if matrix_key not in precomputed_embeddings:
            return []
        
        try:
            matrix = precomputed_embeddings[matrix_key]
            indices = embedding_indices[matrix_key]
            
            # Vectorized cosine similarity - much faster than sklearn for single query
            query_norm = np.linalg.norm(query_embedding)
            matrix_norms = np.linalg.norm(matrix, axis=1)
            
            # Avoid division by zero
            valid_mask = (query_norm > 0) & (matrix_norms > 0)
            if not np.any(valid_mask):
                return []
            
            similarities = np.zeros(len(matrix))
            similarities[valid_mask] = np.dot(matrix[valid_mask], query_embedding) / (matrix_norms[valid_mask] * query_norm)
            
            # Find top matches above threshold
            valid_indices = np.where(similarities >= threshold)[0]
            if len(valid_indices) == 0:
                return []
            
            # Get top k results
            top_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1][:top_k]]
            
            results = []
            for idx in top_indices:
                task = indices[idx]
                similarity = similarities[idx]
                results.append((task, float(similarity)))
            
            return results
        
        except Exception as e:
            print(f"Error in similarity search for {matrix_key}: {e}")
            return []
    
    def clear_all_caches(self):
        """Clear all LRU caches"""
        try:
            self.expand_abbreviations_cached.cache_clear()
            self.extract_technical_identifiers.cache_clear()
            self.extract_component_cached.cache_clear()
            self.find_removal_installation_cached.cache_clear()
            self.map_special_task_aircraft_type.cache_clear()
            self.has_technical_numbers.cache_clear()
            return True
        except Exception as e:
            print(f"Error clearing caches: {e}")
            return False
    
    def get_cache_info(self):
        """Get cache information"""
        return {
            'expand_abbreviations': self.expand_abbreviations_cached.cache_info()._asdict() if hasattr(self.expand_abbreviations_cached, 'cache_info') else {},
            'technical_identifiers': self.extract_technical_identifiers.cache_info()._asdict() if hasattr(self.extract_technical_identifiers, 'cache_info') else {},
            'extract_component': self.extract_component_cached.cache_info()._asdict() if hasattr(self.extract_component_cached, 'cache_info') else {},
            'find_removal_installation': self.find_removal_installation_cached.cache_info()._asdict() if hasattr(self.find_removal_installation_cached, 'cache_info') else {},
            'map_special_task': self.map_special_task_aircraft_type.cache_info()._asdict() if hasattr(self.map_special_task_aircraft_type, 'cache_info') else {},
            'has_technical_numbers': self.has_technical_numbers.cache_info()._asdict() if hasattr(self.has_technical_numbers, 'cache_info') else {}
        }