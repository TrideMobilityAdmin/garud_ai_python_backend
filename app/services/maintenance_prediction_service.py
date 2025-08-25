import numpy as np
import pandas as pd
import time
import os
import pickle
import re
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import List, Dict, Any, Tuple
from app.core.maintenance_config import MaintenanceConfig
from app.utils.maintenance_utils import MaintenanceUtils
from app.services.maintenance_helper_service import MaintenanceHelperService
from app.models.maintenance_models import TaskResult

logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)

class MaintenancePredictionService:
    def __init__(self):
        self.config = MaintenanceConfig()
        self.utils = MaintenanceUtils()
        self.helper = MaintenanceHelperService()
        self.embedding_model = None
        self.master_data_map = {}
        self.special_tasks_data_map = {}
        self.ad_sb_data_map = {}
        self.combination_recipes = {}
        self.nested_task_mhs_lookup = {}
        self.precomputed_embeddings = {}
        self.embedding_indices = {}
        self.similarity_cache = {}
        
        # Initialize the service
        self._load_assets()
    
    def _load_assets(self):
        """Load all required assets for the service"""
        try:
            print("Loading maintenance prediction assets...")
            
            # Load embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load master data map
            with open(self.config.MASTER_DATA_MAP_FILE, 'rb') as f:
                self.master_data_map = pickle.load(f)
            
            self.special_tasks_data_map = self.master_data_map.get('special_tasks', {})
            self.ad_sb_data_map = self.master_data_map.get('ad_sb', {})
            self.combination_recipes = self.master_data_map.get('combination_recipes', {})
            self.nested_task_mhs_lookup = self.master_data_map.get('task_mhs_lookup', {})
            
            self._initialize_precomputed_matrices()
            
            print("Maintenance prediction service ready.")
            
        except Exception as e:
            print(f"FATAL ERROR: Failed to initialize maintenance service: {e}")
            raise
    
    def _initialize_precomputed_matrices(self):
        """Pre-compute embedding matrices for instant similarity calculation"""
        print("Pre-computing maintenance embedding matrices...")
        start_time = time.time()
        
        # AD/SB matrices with proper keys
        for key, tasks in self.ad_sb_data_map.items():
            if tasks:
                try:
                    # Ensure consistent key format
                    key_str = f"{key[0]}_{key[1]}" if isinstance(key, tuple) else str(key)
                    
                    orig_emb1 = np.array([task['original_embedding1'] for task in tasks])
                    orig_emb2 = np.array([task['original_embedding2'] for task in tasks])
                    clean_emb1 = np.array([task['embedding1'] for task in tasks])
                    clean_emb2 = np.array([task['embedding2'] for task in tasks])
                    
                    self.precomputed_embeddings[f"adsb_{key_str}_orig_desc1"] = orig_emb1
                    self.precomputed_embeddings[f"adsb_{key_str}_orig_desc2"] = orig_emb2
                    self.precomputed_embeddings[f"adsb_{key_str}_clean_desc1"] = clean_emb1
                    self.precomputed_embeddings[f"adsb_{key_str}_clean_desc2"] = clean_emb2
                    
                    for suffix in ['_orig_desc1', '_orig_desc2', '_clean_desc1', '_clean_desc2']:
                        self.embedding_indices[f"adsb_{key_str}{suffix}"] = tasks
                except Exception as e:
                    print(f"Warning: Failed to process AD/SB key {key}: {e}")
        
        # Special Tasks matrices
        for key, tasks in self.special_tasks_data_map.items():
            if tasks:
                try:
                    original_embeddings = np.array([task['original_embedding'] for task in tasks])
                    clean_embeddings = np.array([task['embedding'] for task in tasks])
                    
                    self.precomputed_embeddings[f"special_{key}_original"] = original_embeddings
                    self.precomputed_embeddings[f"special_{key}_clean"] = clean_embeddings
                    self.embedding_indices[f"special_{key}_original"] = tasks
                    self.embedding_indices[f"special_{key}_clean"] = tasks
                except Exception as e:
                    print(f"Warning: Failed to process special tasks key {key}: {e}")
        
        elapsed = time.time() - start_time
        print(f"Pre-computed {len(self.precomputed_embeddings)} matrices in {elapsed:.2f}s")
    
    def exact_match_search_special(self, query_desc, aircraft_type):
        """FIXED: Enhanced exact matching with directional awareness - ORIGINAL LOGIC"""
        mapped_type = self.helper.map_special_task_aircraft_type(aircraft_type)
        if mapped_type not in self.special_tasks_data_map:
            return None
        
        query_upper = query_desc.upper().strip()
        query_expanded = self.helper.expand_abbreviations_cached(query_desc).upper().strip()
        
        # Check for exact matches
        for task_data in self.special_tasks_data_map[mapped_type]:
            task_upper = task_data['task_description_upper']
            task_expanded = task_data['cleaned_description'].upper()
            
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
    
    def search_special_tasks_optimized(self, query_embedding, aircraft_type, threshold=0.80, top_k=5):
        """ORIGINAL LOGIC: Optimized special tasks search with SUM capability"""
        try:
            results = []
            mapped_type = self.helper.map_special_task_aircraft_type(aircraft_type)
            
            # Search in both embedding types
            orig_results = self.helper.ultra_fast_similarity_search(
                query_embedding, f"special_{mapped_type}_original", 
                self.precomputed_embeddings, self.embedding_indices, threshold, top_k
            )
            clean_results = self.helper.ultra_fast_similarity_search(
                query_embedding, f"special_{mapped_type}_clean",
                self.precomputed_embeddings, self.embedding_indices, threshold, top_k
            )
            
            all_results = orig_results + clean_results
            seen_tasks = set()
            
            for task_data, similarity in all_results:
                task_key = task_data['task_description']
                if task_key not in seen_tasks:
                    seen_tasks.add(task_key)
                    results.append({
                        'source': 'special_tasks',
                        'task_description': task_data['task_description'],
                        'total_mhs': task_data['total_mhs'],
                        'similarity': similarity,
                        'match_type': 'semantic'
                    })
            
            return sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]
        
        except Exception as e:
            import traceback
            print(f"ERROR in search_special_tasks_optimized: {e}")
            traceback.print_exc()
            return []
    
    def search_ad_sb_optimized(self, query_embedding, aircraft_type, aircraft_reg, threshold=0.80, top_k=5):
        """FIXED: Optimized AD/SB search with proper directional matching"""
        results = []
        
        # Try exact aircraft match first
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
                if search_key in self.precomputed_embeddings:
                    results.extend(self.helper.ultra_fast_similarity_search(
                        query_embedding, search_key, 
                        self.precomputed_embeddings, self.embedding_indices, threshold, top_k
                    ))
            
            # If we found good matches, use them
            if results and max([r[1] for r in results]) > 0.85:
                break
        
        # Fallback to other aircraft types if no good matches
        if not results or max([r[1] for r in results]) < 0.8:
            for key in self.ad_sb_data_map.keys():
                if key != adsb_key:
                    key_str = f"{key[0]}_{key[1]}" if isinstance(key, tuple) else str(key)
                    search_keys = [
                        f"adsb_{key_str}_orig_desc1",
                        f"adsb_{key_str}_orig_desc2",
                        f"adsb_{key_str}_clean_desc1", 
                        f"adsb_{key_str}_clean_desc2"
                    ]
                    
                    for search_key in search_keys:
                        if search_key in self.precomputed_embeddings:
                            results.extend(self.helper.ultra_fast_similarity_search(
                                query_embedding, search_key,
                                self.precomputed_embeddings, self.embedding_indices, threshold, 3
                            ))
        
        # Deduplicate and sort
        seen_tasks = set()
        unique_results = []
        
        for task_data, similarity in results:
            task_key = task_data.get('task_number', task_data.get('task_description', ''))
            if task_key not in seen_tasks:
                seen_tasks.add(task_key)
                
                # FIXED: Preserve directional information in matching
                desc1 = task_data.get('task_description1', '')
                desc2 = task_data.get('task_description2', '')
                
                unique_results.append({
                    'source': 'ad_sb',
                    'task_number': task_data.get('task_number', ''),
                    'task_description1': desc1,
                    'task_description2': desc2,
                    'total_mhs': task_data.get('total_mhs', 0),
                    'similarity': similarity,
                    'match_type': 'semantic',
                    'aircraft_type': task_data.get('aircraft_type', ''),
                    'aircraft_reg': task_data.get('aircraft_reg', '')
                })
        
        return sorted(unique_results, key=lambda x: x['similarity'], reverse=True)[:top_k]
    
    def process_single_task_optimized(self, task_index, task, embedding, aircraft_type, aircraft_reg):
        """
        ORIGINAL LOGIC: Optimized processing with REPLACEMENT sum logic and order preservation
        """
        task_desc = self.utils.normalize_task_description(task.get('task_description', ''))
        task_num = task.get('task_number', '')

        if not task_desc:
            return (task_index, {
                "task_number": task_num,
                "task_description": task_desc,
                "total_mhs": 0.0,
                "status": "not available",
                "task_type": "unknown",
                "similarity_score": 0.0,
                "best_match_found": "",
                "matched_description": None,
                "combined_from": []
            })

        all_matches = []

        # Step 1: REPLACEMENT LOGIC - Check for REPLACEMENT keyword first (ORIGINAL LOGIC)
        try:
            replacement_match = self.helper.find_replacement_tasks_with_sum(task_desc, aircraft_type, self.special_tasks_data_map)
            if replacement_match and replacement_match['total_mhs'] > 0:
                all_matches.append({
                    'total_mhs': replacement_match['total_mhs'],
                    'similarity': 1.0,
                    'description': replacement_match['task_description'],
                    'source': 'special_task',
                    'combined_from': replacement_match.get('combined_tasks', [])
                })
        except Exception as e:
            print(f"Error in replacement search: {e}")

        # Step 2: Try EXACT match if no replacement match
        try:
            if not all_matches:  # Only if no replacement match found
                exact_match = self.exact_match_search_special(task_desc, aircraft_type)
                if exact_match and exact_match['total_mhs'] > 0:
                    all_matches.append({
                        'total_mhs': exact_match['total_mhs'],
                        'similarity': 1.0,
                        'description': exact_match['task_description'],
                        'source': 'special_task',
                        'combined_from': []
                    })
        except Exception as e:
            print(f"Error in exact match search: {e}")

        # Step 3: Search Special Tasks with semantic similarity
        try:
            if not all_matches or max([m['similarity'] for m in all_matches]) < 0.95:
                special_matches = self.search_special_tasks_optimized(embedding, aircraft_type, threshold=0.80, top_k=3)
                for match in special_matches:
                    all_matches.append({
                        'total_mhs': match['total_mhs'],
                        'similarity': match['similarity'],
                        'description': match['task_description'],
                        'source': 'special_task',
                        'combined_from': []
                    })
        except Exception as e:
            print(f"Error in special tasks search: {e}")

        # Step 4: Search AD/SB Tasks
        try:
            has_good_special_match = any(m['source'] == 'special_task' and m['similarity'] >= 0.85 and m['total_mhs'] > 0 for m in all_matches)
            if not has_good_special_match:
                adsb_matches = self.search_ad_sb_optimized(embedding, aircraft_type, aircraft_reg, threshold=0.75, top_k=3)
                for match in adsb_matches:
                    description = match['task_description1'] or match['task_description2']
                    all_matches.append({
                        'total_mhs': match['total_mhs'],
                        'similarity': match['similarity'],
                        'description': description,
                        'source': 'ad_sb',
                        'combined_from': [],
                        'task_number': match.get('task_number', ''),
                        'source_aircraft': f"{match.get('aircraft_type', '')} - {match.get('aircraft_reg', '')}"
                    })
        except Exception as e:
            print(f"Error in AD/SB search: {e}")

        # Step 5: Combination recipe matching (only if no good direct matches)
        try:
            has_good_match = any(m['similarity'] >= 0.85 and m['total_mhs'] > 0 for m in all_matches)
            if not has_good_match:
                component = self.helper.extract_component_cached(task_desc)
                if component:
                    mapped_type = self.helper.map_special_task_aircraft_type(aircraft_type)
                    if mapped_type in self.nested_task_mhs_lookup:
                        tasks_info, mhs_info, total_mhs = self.helper.find_removal_installation_cached(
                            component, mapped_type, self.nested_task_mhs_lookup
                        )
                        if total_mhs > 0:
                            all_matches.append({
                                'total_mhs': total_mhs,
                                'similarity': 0.90,
                                'description': f"Combination of removal and installation for {component}",
                                'source': 'special_task',
                                'combined_from': [
                                    f"Removal: {mhs_info[0]} MHS",
                                    f"Installation: {mhs_info[1]} MHS"
                                ]
                            })
        except Exception as e:
            print(f"Error in combination recipe matching: {e}")

        # Find the best match - ORIGINAL STATUS LOGIC
        if all_matches:
            # Sort by similarity first, then by MHS
            all_matches.sort(key=lambda x: (x['similarity'], x['total_mhs']), reverse=True)
            best_match = all_matches[0]
            
            # ORIGINAL STATUS DETERMINATION LOGIC
            if best_match['similarity'] >= 0.85 and best_match['total_mhs'] > 0:
                status = "available"
            elif best_match['similarity'] >= 0.75 and best_match['total_mhs'] > 0:
                status = "available"
            else:
                status = "not available"

            return (task_index, {
                "task_number": task_num,
                "task_description": task_desc,
                "total_mhs": round(float(best_match['total_mhs']), 2),
                "status": status,
                "task_type": best_match['source'],
                "similarity_score": round(float(best_match['similarity']), 4),
                "best_match_found": best_match['description'],
                "matched_description": best_match['description'],
                "combined_from": best_match.get('combined_from', [])
            })
        else:
            return (task_index, {
                "task_number": task_num,
                "task_description": task_desc,
                "total_mhs": 0.0,
                "status": "not available",
                "task_type": "unknown",
                "similarity_score": 0.0,
                "best_match_found": "",
                "matched_description": None,
                "combined_from": []
            })
    
    def ultra_fast_batch_processing_ordered(self, tasks_batch, embeddings_batch, aircraft_type, aircraft_reg):
        """FIXED: Ultra-fast batch processing with order preservation"""
        if len(tasks_batch) <= 10:
            # Process sequentially for small batches
            results = []
            for i in range(len(tasks_batch)):
                task_index, result = self.process_single_task_optimized(i, tasks_batch[i], embeddings_batch[i], aircraft_type, aircraft_reg)
                results.append((task_index, result))
            return results
        else:
            # Process in parallel but preserve order
            with ThreadPoolExecutor(max_workers=min(self.config.MAX_WORKERS, len(tasks_batch))) as executor:
                futures = []
                for i in range(len(tasks_batch)):
                    future = executor.submit(self.process_single_task_optimized, i, tasks_batch[i], embeddings_batch[i], aircraft_type, aircraft_reg)
                    futures.append((i, future))
                
                # Collect results in order
                results = [None] * len(futures)
                for original_index, future in futures:
                    task_index, result = future.result()
                    results[original_index] = (task_index, result)
                
                return results
    
    async def predict_tasks(self, tasks: List[Dict], aircraft_type: str, aircraft_reg: str = None) -> Dict:
        """Main prediction method - ORIGINAL LOGIC WITH REPLACEMENT SUM"""
        start_time = time.time()
        
        try:
            # Convert input format
            tasks_list = [{"task_number": t.get("task_number", ""), 
                          "task_description": t.get("task_description", "")} for t in tasks]
            
            if not tasks_list:
                return {"results": [], "processing_time_ms": 0, "total_tasks": 0}
            
            if not aircraft_type:
                raise ValueError('aircraft_type is required')

            # PERFORMANCE: Batch embedding generation
            task_descriptions = []
            for task in tasks_list:
                desc = task.get('task_description', '')
                if not desc:
                    raise ValueError('task_description is required for all tasks')
                task_descriptions.append(self.utils.normalize_task_description(desc))
            
            # FIXED: More conservative abbreviation expansion
            expanded_descriptions = [self.helper.expand_abbreviations_cached(desc) for desc in task_descriptions]
            
            # Generate embeddings in batch (much faster)
            embedding_start = time.time()
            batch_size = min(256, len(tasks_list))  # Larger batch size for faster processing
            all_embeddings = self.embedding_model.encode(
                expanded_descriptions,
                show_progress_bar=False,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            embedding_time = time.time() - embedding_start
            
            # Process tasks with order preservation
            processing_start = time.time()
            if len(tasks_list) <= 50:
                # Direct processing for smaller batches
                indexed_results = self.ultra_fast_batch_processing_ordered(tasks_list, all_embeddings, aircraft_type, aircraft_reg)
            else:
                # Chunked processing for larger batches
                chunk_size = max(25, len(tasks_list) // (self.config.MAX_WORKERS * 2))
                chunks = []
                for i in range(0, len(tasks_list), chunk_size):
                    chunk_tasks = tasks_list[i:i+chunk_size]
                    chunk_embeddings = all_embeddings[i:i+chunk_size]
                    chunks.append((chunk_tasks, chunk_embeddings, aircraft_type, aircraft_reg, i))  # Include start index
                
                with ThreadPoolExecutor(max_workers=min(self.config.MAX_WORKERS, len(chunks))) as executor:
                    chunk_futures = []
                    for chunk_tasks, chunk_embeddings, ac_type, ac_reg, start_idx in chunks:
                        future = executor.submit(self.ultra_fast_batch_processing_ordered, chunk_tasks, chunk_embeddings, ac_type, ac_reg)
                        chunk_futures.append((start_idx, future))
                    
                    # Collect and reorder results
                    indexed_results = [None] * len(tasks_list)
                    for start_idx, future in chunk_futures:
                        chunk_results = future.result()
                        for task_idx, result in chunk_results:
                            global_idx = start_idx + task_idx
                            indexed_results[global_idx] = (global_idx, result)

            processing_time = time.time() - processing_start

            # Ensure all positions filled (fallback if something failed)
            for i in range(len(indexed_results)):
                if indexed_results[i] is None:
                    indexed_results[i] = (i, {
                        "task_number": tasks_list[i].get('task_number', ''),
                        "task_description": self.utils.normalize_task_description(tasks_list[i].get('task_description', '')),
                        "total_mhs": 0.0,
                        "status": "not available",
                        "task_type": "unknown",
                        "similarity_score": 0.0,
                        "best_match_found": "",
                        "matched_description": None,
                        "combined_from": []
                    })

            # Flatten into original order
            results_in_order = [res for (_, res) in indexed_results]

            total_time_ms = int((time.time() - start_time) * 1000)
            return {
                "results": results_in_order,
                "processing_time_ms": total_time_ms,
                "embedding_time_ms": int(embedding_time * 1000),
                "task_processing_time_ms": int(processing_time * 1000),
                "total_tasks": len(results_in_order)
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in predict_tasks: {e}")
            raise
    
    async def health_check(self) -> Dict:
        """Health check for maintenance service"""
        return {
            'status': 'healthy',
            'model_loaded': self.embedding_model is not None,
            'data_loaded': len(self.special_tasks_data_map) > 0 or len(self.ad_sb_data_map) > 0,
            'precomputed_matrices': len(self.precomputed_embeddings),
            'cache_info': self.helper.get_cache_info()
        }
    
    async def get_stats(self) -> Dict:
        """Get service statistics"""
        return {
            'special_tasks_aircraft_types': list(self.special_tasks_data_map.keys()),
            'ad_sb_combinations': len(self.ad_sb_data_map),
            'combination_recipes': len(self.combination_recipes),
            'task_mhs_lookups': len(self.nested_task_mhs_lookup),
            'precomputed_matrices': len(self.precomputed_embeddings),
            'embedding_dimensions': list(self.precomputed_embeddings.values())[0].shape[1] if self.precomputed_embeddings else 0,
            'cache_sizes': self.helper.get_cache_info()
        }
    
    async def clear_cache(self) -> Dict:
        """Clear service caches"""
        try:
            success = self.helper.clear_all_caches()
            return {
                'status': 'success' if success else 'partial_failure', 
                'message': 'All caches cleared' if success else 'Some caches may not have been cleared'
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}