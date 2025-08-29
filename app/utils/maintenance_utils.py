import os
import pickle
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.maintenance_config import MASTER_DATA_MAP_FILE
import logging

logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)

# Global Variables
embedding_model = None
master_data_map = {}
special_tasks_data_map = {}
ad_sb_data_map = {}
combination_recipes = {}
nested_task_mhs_lookup = {}

# PRESERVED: Old code precomputed matrices for performance
PRECOMPUTED_EMBEDDINGS = {}
EMBEDDING_INDICES = {}
_assets_initialized = False

def initialize_precomputed_matrices():
    """PRESERVED: Pre-compute embedding matrices for instant similarity calculation"""
    global PRECOMPUTED_EMBEDDINGS, EMBEDDING_INDICES
    
    print("Pre-computing embedding matrices...")
    start_time = time.time()
    
    # AD/SB matrices with proper keys (preserved from old code)
    for key, tasks in ad_sb_data_map.items():
        if tasks:
            try:
                key_str = f"{key[0]}_{key[1]}" if isinstance(key, tuple) else str(key)
                
                orig_emb1 = np.array([task['original_embedding1'] for task in tasks])
                orig_emb2 = np.array([task['original_embedding2'] for task in tasks])
                clean_emb1 = np.array([task['embedding1'] for task in tasks])
                clean_emb2 = np.array([task['embedding2'] for task in tasks])
                
                PRECOMPUTED_EMBEDDINGS[f"adsb_{key_str}_orig_desc1"] = orig_emb1
                PRECOMPUTED_EMBEDDINGS[f"adsb_{key_str}_orig_desc2"] = orig_emb2
                PRECOMPUTED_EMBEDDINGS[f"adsb_{key_str}_clean_desc1"] = clean_emb1
                PRECOMPUTED_EMBEDDINGS[f"adsb_{key_str}_clean_desc2"] = clean_emb2
                
                for suffix in ['_orig_desc1', '_orig_desc2', '_clean_desc1', '_clean_desc2']:
                    EMBEDDING_INDICES[f"adsb_{key_str}{suffix}"] = tasks
                    
            except Exception as e:
                print(f"Warning: Failed to process AD/SB key {key}: {e}")
    
    # Special Tasks matrices (enhanced with new structure)
    for key, tasks in special_tasks_data_map.items():
        if tasks:
            try:
                original_embeddings = np.array([task.get('original_embedding', task.get('embedding', np.zeros(384))) for task in tasks])
                clean_embeddings = np.array([task.get('embedding', np.zeros(384)) for task in tasks])
                
                PRECOMPUTED_EMBEDDINGS[f"special_{key}_original"] = original_embeddings
                PRECOMPUTED_EMBEDDINGS[f"special_{key}_clean"] = clean_embeddings
                EMBEDDING_INDICES[f"special_{key}_original"] = tasks
                EMBEDDING_INDICES[f"special_{key}_clean"] = tasks
                
                # Add variation embeddings if available
                if any('variation_embeddings' in task and task['variation_embeddings'] for task in tasks):
                    variation_embeddings = []
                    variation_tasks = []
                    for task in tasks:
                        if 'variation_embeddings' in task and task['variation_embeddings']:
                            for var_emb in task['variation_embeddings']:
                                variation_embeddings.append(var_emb)
                                variation_tasks.append(task)
                    
                    if variation_embeddings:
                        PRECOMPUTED_EMBEDDINGS[f"special_{key}_variations"] = np.array(variation_embeddings)
                        EMBEDDING_INDICES[f"special_{key}_variations"] = variation_tasks
                        
            except Exception as e:
                print(f"Warning: Failed to process special tasks key {key}: {e}")
    
    elapsed = time.time() - start_time
    print(f"Pre-computed {len(PRECOMPUTED_EMBEDDINGS)} embedding matrices in {elapsed:.2f}s")

async def initialize_assets():
    """ASYNC: Load assets with both old and new code compatibility"""
    global embedding_model, master_data_map, special_tasks_data_map
    global ad_sb_data_map, combination_recipes, nested_task_mhs_lookup, _assets_initialized
    
    if _assets_initialized:
        print("Assets already initialized, skipping...")
        return True
    
    try:
        print("data import started...")
        print("Loading embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"Loading master data map from '{MASTER_DATA_MAP_FILE}'...")
        if not os.path.exists(MASTER_DATA_MAP_FILE):
            print(f"ERROR: Master data file not found: {MASTER_DATA_MAP_FILE}")
            return False
            
        with open(MASTER_DATA_MAP_FILE, 'rb') as f:
            master_data_map = pickle.load(f)

        special_tasks_data_map = master_data_map.get('special_tasks', {})
        ad_sb_data_map = master_data_map.get('ad_sb', {})
        combination_recipes = master_data_map.get('combination_recipes', {})
        nested_task_mhs_lookup = master_data_map.get('task_mhs_lookup', {})

        # Validation
        print(f"Loaded data successfully:")
        print(f"  - Special task aircraft types: {len(special_tasks_data_map)}")
        print(f"  - Total special tasks: {sum(len(tasks) for tasks in special_tasks_data_map.values())}")
        print(f"  - AD/SB combinations: {len(ad_sb_data_map)}")
        print(f"  - Total AD/SB tasks: {sum(len(tasks) for tasks in ad_sb_data_map.values())}")
        print(f"  - Combination recipes: {len(combination_recipes)}")
        print(f"  - Task MHS lookups: {len(nested_task_mhs_lookup)}")

        # Initialize precomputed matrices for performance
        initialize_precomputed_matrices()

        print("Enhanced API ready with preserved old code logic.")
        _assets_initialized = True
        return True

    except Exception as e:
        import traceback
        print(f"FATAL ERROR: Failed to initialize API: {e}")
        traceback.print_exc()
        return False

# Getter functions for global variables
def get_embedding_model():
    return embedding_model

def get_master_data_map():
    return master_data_map

def get_special_tasks_data_map():
    return special_tasks_data_map

def get_ad_sb_data_map():
    return ad_sb_data_map

def get_combination_recipes():
    return combination_recipes

def get_nested_task_mhs_lookup():
    return nested_task_mhs_lookup

def get_precomputed_embeddings():
    return PRECOMPUTED_EMBEDDINGS

def get_embedding_indices():
    return EMBEDDING_INDICES