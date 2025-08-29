import numpy as np
import re
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple
from app.core.maintenance_config import COMPONENT_KEYWORDS, COMPONENT_EXCLUSION_RULES
from app.utils.maintenance_utils import (
    get_special_tasks_data_map, get_ad_sb_data_map,
    get_combination_recipes, get_precomputed_embeddings, get_embedding_indices
)
from app.utils.text_processing import (
    expand_abbreviations_cached, map_special_task_aircraft_type,
    extract_component_cached, find_removal_installation_cached
)

# --- PRESERVED: Old code ultra-fast similarity search ---
def ultra_fast_similarity_search(query_embedding, matrix_key, threshold=0.75, top_k=5):
    """PRESERVED: Ultra-fast vectorized similarity search from old code"""
    precomputed_embeddings = get_precomputed_embeddings()
    embedding_indices = get_embedding_indices()
    
    if matrix_key not in precomputed_embeddings:
        return []
    
    try:
        matrix = precomputed_embeddings[matrix_key]
        indices = embedding_indices[matrix_key]
        
        # Vectorized cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        matrix_norms = np.linalg.norm(matrix, axis=1)
        
        valid_mask = (query_norm > 0) & (matrix_norms > 0)
        if not np.any(valid_mask):
            return []
        
        similarities = np.zeros(len(matrix))
        similarities[valid_mask] = np.dot(matrix[valid_mask], query_embedding) / (matrix_norms[valid_mask] * query_norm)
        
        valid_indices = np.where(similarities >= threshold)[0]
        if len(valid_indices) == 0:
            return []
        
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

# --- NEW: Enhanced similarity boost from new code ---
@lru_cache(maxsize=5000)
def should_exclude_match(query_desc, target_desc):
    """Enhanced exclusion check from new code"""
    query_lower = query_desc.lower()
    target_lower = target_desc.lower()
    
    # Extract component types
    query_component = extract_component_type(query_desc)
    target_component = extract_component_type(target_desc)
    
    if query_component != 'unknown' and target_component != 'unknown':
        if query_component != target_component:
            # Check if they're in compatible groups
            compatible_groups = [
                ['cockpit_seat'], ['passenger_seat'],
                ['external_wash'], ['technical_external_wash'],
                ['cockpit_cleaning'], ['cabin_cleaning', 'cabin_deep_cleaning'],
                ['cargo_cleaning', 'fwd_cargo_cleaning', 'aft_cargo_cleaning'],
                ['oven'], ['coffee_maker'], ['water_boiler'], ['water_tank'],
            ]
            
            in_same_group = any(query_component in group and target_component in group for group in compatible_groups)
            if not in_same_group:
                return True
    
    # Apply exclusion rules
    if query_component in COMPONENT_EXCLUSION_RULES:
        rules = COMPONENT_EXCLUSION_RULES[query_component]
        
        exclude_terms = rules.get('exclude_if_target_has', [])
        if any(term.lower() in target_lower for term in exclude_terms):
            return True
        
        must_have = rules.get('must_have_one_of', [])
        if must_have and not any(term.lower() in target_lower for term in must_have):
            return True
    
    return False

@lru_cache(maxsize=5000)
def extract_component_type(description):
    """Enhanced component type extraction from new code"""
    desc_lower = description.lower()
    
    priority_order = [
        'engine_chemical_wash', 'cockpit_seat', 'passenger_seat',
        'technical_external_wash', 'external_wash',
        'cockpit_cleaning', 'cabin_deep_cleaning', 'cabin_cleaning',
        'fwd_cargo_cleaning', 'aft_cargo_cleaning', 'cargo_cleaning',
        'oven', 'coffee_maker', 'water_boiler', 'water_tank',
    ]
    
    for comp_type in priority_order:
        if comp_type in COMPONENT_KEYWORDS:
            keywords = COMPONENT_KEYWORDS[comp_type]
            for keyword in keywords:
                if keyword.lower() in desc_lower:
                    return comp_type
    
    return 'unknown'

def exact_match_search_special(query_desc, aircraft_type):
    """PRESERVED: Enhanced exact matching with directional awareness from old code"""
    special_tasks_data_map = get_special_tasks_data_map()
    
    mapped_type = map_special_task_aircraft_type(aircraft_type)
    if mapped_type not in special_tasks_data_map:
        return None
    
    query_upper = query_desc.upper().strip()
    query_expanded = expand_abbreviations_cached(query_desc).upper().strip()
    
    # Check for exact matches
    for task_data in special_tasks_data_map[mapped_type]:
        task_upper = task_data.get('task_description_upper', task_data['task_description'].upper())
        task_expanded = task_data.get('cleaned_description', expand_abbreviations_cached(task_data['task_description'])).upper()
        
        # Exact match found
        if query_upper == task_upper or query_expanded == task_expanded:
            return {
                'source': 'special_tasks',
                'task_description': task_data['task_description'],
                'total_mhs': task_data['total_mhs'],
                'similarity': 1.0,
                'match_type': 'exact'
            }
    
    return None

def search_special_tasks_enhanced(query_embedding, aircraft_type, query_desc="", threshold=0.70, top_k=5):
    """ENHANCED: Combine old code performance with new code enhancements"""
    try:
        results = []
        mapped_type = map_special_task_aircraft_type(aircraft_type)
        
        # Use old code's ultra-fast similarity search
        orig_results = ultra_fast_similarity_search(query_embedding, f"special_{mapped_type}_original", threshold, top_k)
        clean_results = ultra_fast_similarity_search(query_embedding, f"special_{mapped_type}_clean", threshold, top_k)
        
        # Also check variations if available
        var_results = ultra_fast_similarity_search(query_embedding, f"special_{mapped_type}_variations", threshold, top_k)
        
        all_results = orig_results + clean_results + var_results
        seen_tasks = set()
        
        for task_data, similarity in all_results:
            task_key = task_data['task_description']
            if task_key not in seen_tasks:
                seen_tasks.add(task_key)
                
                # Apply new code's exclusion logic
                if query_desc and should_exclude_match(query_desc, task_data['task_description']):
                    continue
                    
                results.append({
                    'source': 'special_tasks',
                    'task_description': task_data['task_description'],
                    'total_mhs': task_data['total_mhs'],
                    'similarity': similarity,
                    'match_type': 'semantic'
                })
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]
    
    except Exception as e:
        print(f"ERROR in search_special_tasks_enhanced: {e}")
        return []

def search_ad_sb_enhanced(query_embedding, aircraft_type, aircraft_reg, query_desc="", threshold=0.80, top_k=5):
    """PRESERVED: Old code AD/SB search logic with enhancements"""
    ad_sb_data_map = get_ad_sb_data_map()
    
    results = []
    
    # PRESERVED: Old code's AD/SB matching logic
    adsb_key = (aircraft_type, aircraft_reg)
    key_variants = [
        adsb_key,
        f"{aircraft_type}_{aircraft_reg}",
        (aircraft_type, aircraft_reg) if aircraft_reg else (aircraft_type, "")
    ]
    
    for key_variant in key_variants:
        key_str = f"{key_variant[0]}_{key_variant[1]}" if isinstance(key_variant, tuple) else str(key_variant)
        
        search_keys = [
            f"adsb_{key_str}_orig_desc1",
            f"adsb_{key_str}_orig_desc2", 
            f"adsb_{key_str}_clean_desc1",
            f"adsb_{key_str}_clean_desc2"
        ]
        
        for search_key in search_keys:
            results.extend(ultra_fast_similarity_search(query_embedding, search_key, threshold, top_k))
        
        # If we found good matches, use them
        if results and max([r[1] for r in results]) > 0.85:
            break
    
    # Fallback to other aircraft types
    if not results or max([r[1] for r in results]) < 0.8:
        for key in ad_sb_data_map.keys():
            if key != adsb_key:
                key_str = f"{key[0]}_{key[1]}" if isinstance(key, tuple) else str(key)
                search_keys = [
                    f"adsb_{key_str}_orig_desc1",
                    f"adsb_{key_str}_orig_desc2",
                    f"adsb_{key_str}_clean_desc1", 
                    f"adsb_{key_str}_clean_desc2"
                ]
                
                for search_key in search_keys:
                    results.extend(ultra_fast_similarity_search(query_embedding, search_key, threshold, 3))
    
    # Process results with exclusion logic
    seen_tasks = set()
    unique_results = []
    
    for task_data, similarity in results:
        task_key = task_data.get('task_number', task_data.get('task_description', ''))
        if task_key not in seen_tasks:
            seen_tasks.add(task_key)
            
            desc1 = task_data.get('task_description1', '')
            desc2 = task_data.get('task_description2', '')
            
            # Apply exclusion logic if query_desc provided
            best_desc = desc1 or desc2
            if query_desc and should_exclude_match(query_desc, best_desc):
                continue
            
            unique_results.append({
                'source': 'ad_sb',
                'task_number': task_data.get('task_number', ''),
                'task_description1': desc1,
                'task_description2': desc2,
                'best_description': best_desc,
                'total_mhs': task_data.get('total_mhs', 0),
                'similarity': similarity,
                'match_type': 'semantic',
                'aircraft_type': task_data.get('aircraft_type', ''),
                'aircraft_reg': task_data.get('aircraft_reg', '')
            })
    
    return sorted(unique_results, key=lambda x: x['similarity'], reverse=True)[:top_k]

def search_combination_recipes(query_desc, aircraft_type, threshold=0.8):
    """PRESERVED: Old code combination recipe logic"""
    combination_recipes = get_combination_recipes()
    
    try:
        query_upper = query_desc.upper()
        
        # Direct recipe lookup
        for recipe_key, recipe_data in combination_recipes.items():
            if recipe_key.upper() in query_upper or any(task.upper() in query_upper for task in recipe_data.get('tasks', [])):
                return {
                    'source': 'combination_recipe',
                    'task_description': recipe_key,
                    'total_mhs': recipe_data.get('total_mhs', 0),
                    'similarity': 0.9,
                    'match_type': 'recipe',
                    'combined_from': recipe_data.get('tasks', []),
                    'removal_mhs': recipe_data.get('removal_mhs', 0),
                    'installation_mhs': recipe_data.get('installation_mhs', 0)
                }
        
        # Component-based matching
        component_patterns = [
            r'^(.+?)\s+(?:REPLACEMENT|R&I|REM\s*&\s*INSTL?|REMOVE\s+AND\s+INSTALL)',
            r'^(.+?)\s+(?:REMOVAL\s+AND\s+INSTALLATION)'
        ]
        
        for pattern in component_patterns:
            match = re.search(pattern, query_upper)
            if match:
                component = match.group(1).strip()
                for recipe_key, recipe_data in combination_recipes.items():
                    if component in recipe_key.upper():
                        return {
                            'source': 'combination_recipe',
                            'task_description': recipe_key,
                            'total_mhs': recipe_data.get('total_mhs', 0),
                            'similarity': 0.85,
                            'match_type': 'component_recipe',
                            'combined_from': recipe_data.get('tasks', []),
                            'removal_mhs': recipe_data.get('removal_mhs', 0),
                            'installation_mhs': recipe_data.get('installation_mhs', 0)
                        }
        
        return None
    
    except Exception as e:
        print(f"Error in search_combination_recipes: {e}")
        return None