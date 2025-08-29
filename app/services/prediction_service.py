import time
from typing import List, Dict, Any
from app.models.maintenance_models import PredictRequest, PredictResponse, TaskResult, HealthResponse
from app.utils.maintenance_utils import (
    get_embedding_model, get_master_data_map, get_special_tasks_data_map, 
    get_ad_sb_data_map, get_combination_recipes, get_nested_task_mhs_lookup, 
    get_precomputed_embeddings, get_embedding_indices
)
from app.utils.text_processing import normalize_task_description, expand_abbreviations_cached
from app.utils.search_utils import exact_match_search_special, search_special_tasks_enhanced, search_ad_sb_enhanced, search_combination_recipes
from app.utils.task_processing import process_single_task_enhanced, ultra_fast_batch_processing_ordered
from app.core.maintenance_config import MAX_WORKERS

async def predict_tasks(payload: PredictRequest) -> PredictResponse:
    """
    Enhanced prediction service with preserved old code logic
    """
    start_time = time.time()
    
    tasks_list = [{"task_number": task.task_number, "task_description": task.task_description} for task in payload.tasks]
    aircraft_type = payload.aircraft_type.strip()
    aircraft_reg = payload.aircraft_reg.strip() if payload.aircraft_reg else None
    
    print(f"Processing {len(tasks_list)} tasks for aircraft: {aircraft_type}, reg: {aircraft_reg}")
    
    embedding_model = get_embedding_model()
    if not embedding_model:
        raise Exception("Embedding model not loaded")
    
    # PRESERVED: Old code batch embedding generation
    task_descriptions = []
    for task in tasks_list:
        desc = task.get('task_description', '')
        task_descriptions.append(normalize_task_description(desc))
    
    # PRESERVED: Conservative abbreviation expansion from old code
    expanded_descriptions = [expand_abbreviations_cached(desc) for desc in task_descriptions]
    
    # Generate embeddings in batch
    embedding_start = time.time()
    batch_size = min(256, len(tasks_list))
    all_embeddings = embedding_model.encode(
        expanded_descriptions,
        show_progress_bar=False,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    embedding_time = time.time() - embedding_start
    
    # PRESERVED: Old code processing with order preservation
    processing_start = time.time()
    if len(tasks_list) <= 50:
        indexed_results = ultra_fast_batch_processing_ordered(tasks_list, all_embeddings, aircraft_type, aircraft_reg)
    else:
        # Chunked processing for larger batches
        chunk_size = max(25, len(tasks_list) // (MAX_WORKERS * 2))
        chunks = []
        for i in range(0, len(tasks_list), chunk_size):
            chunk_tasks = tasks_list[i:i+chunk_size]
            chunk_embeddings = all_embeddings[i:i+chunk_size]
            chunks.append((chunk_tasks, chunk_embeddings, aircraft_type, aircraft_reg, i))
        
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(chunks))) as executor:
            chunk_futures = []
            for chunk_tasks, chunk_embeddings, ac_type, ac_reg, start_idx in chunks:
                future = executor.submit(ultra_fast_batch_processing_ordered, chunk_tasks, chunk_embeddings, ac_type, ac_reg)
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
                "task_description": normalize_task_description(tasks_list[i].get('task_description', '')),
                "total_mhs": 0.0,
                "status": "not available",
                "task_type": "unknown",
                "similarity_score": 0.0,
                "best_match_found": "",
                "matched_description": None,
                "combined_from": []
            })

    # Flatten into original order and create TaskResult objects (removing age fields)
    results_in_order = []
    for (_, res) in indexed_results:
        task_result = TaskResult(
            task_number=res['task_number'],
            task_description=res['task_description'],
            total_mhs=res['total_mhs'],
            status=res['status'],
            task_type=res['task_type'],
            similarity_score=res['similarity_score'],
            best_match_found=res['best_match_found'],
            matched_description=res['matched_description'],
            combined_from=res['combined_from']
            # Removed age_difference and matched_age as requested
        )
        results_in_order.append(task_result)
    
    # Summary statistics
    available_count = sum(1 for r in results_in_order if r.status == 'available')
    total_mhs = sum(r.total_mhs for r in results_in_order if r.status == 'available')

    total_time_ms = int((time.time() - start_time) * 1000)
    
    return PredictResponse(
        results=results_in_order,
        processing_time_ms=total_time_ms,
        embedding_time_ms=int(embedding_time * 1000),
        task_processing_time_ms=int(processing_time * 1000),
        total_tasks=len(results_in_order),
        available_tasks=available_count,
        total_available_mhs=round(total_mhs, 2)
    )

async def get_health_status() -> HealthResponse:
    """Health status check"""
    from app.utils.text_processing import expand_abbreviations_cached, extract_component_cached, find_removal_installation_cached
    
    embedding_model = get_embedding_model()
    special_tasks_data_map = get_special_tasks_data_map()
    ad_sb_data_map = get_ad_sb_data_map()
    combination_recipes = get_combination_recipes()
    precomputed_embeddings = get_precomputed_embeddings()
    
    cache_info = {}
    try:
        cache_info = {
            'expand_abbreviations': expand_abbreviations_cached.cache_info()._asdict() if hasattr(expand_abbreviations_cached, 'cache_info') else {},
            'extract_component': extract_component_cached.cache_info()._asdict() if hasattr(extract_component_cached, 'cache_info') else {},
            'find_removal_installation': find_removal_installation_cached.cache_info()._asdict() if hasattr(find_removal_installation_cached, 'cache_info') else {}
        }
    except:
        cache_info = {}
    
    return HealthResponse(
        status='healthy',
        model_loaded=embedding_model is not None,
        special_tasks_loaded=len(special_tasks_data_map) > 0,
        ad_sb_loaded=len(ad_sb_data_map) > 0,
        combination_recipes_loaded=len(combination_recipes) > 0,
        precomputed_matrices=len(precomputed_embeddings),
        aircraft_types=list(special_tasks_data_map.keys()) if special_tasks_data_map else [],
        total_special_tasks=sum(len(tasks) for tasks in special_tasks_data_map.values()),
        total_ad_sb_tasks=sum(len(tasks) for tasks in ad_sb_data_map.values()),
        total_recipes=len(combination_recipes),
        cache_info=cache_info
    )

async def debug_aircraft_type(aircraft_type: str) -> Dict[str, Any]:
    """Debug endpoint to check aircraft type data"""
    from app.utils.text_processing import map_special_task_aircraft_type
    
    special_tasks_data_map = get_special_tasks_data_map()
    ad_sb_data_map = get_ad_sb_data_map()
    combination_recipes = get_combination_recipes()
    nested_task_mhs_lookup = get_nested_task_mhs_lookup()
    precomputed_embeddings = get_precomputed_embeddings()
    
    mapped_type = map_special_task_aircraft_type(aircraft_type)
    result = {
        'aircraft_type': aircraft_type,
        'mapped_type': mapped_type,
        'found_in_special_tasks': mapped_type in special_tasks_data_map,
        'available_aircraft_types': list(special_tasks_data_map.keys()),
        'special_tasks_count': len(special_tasks_data_map.get(mapped_type, [])),
        'ad_sb_keys': [str(k) for k in ad_sb_data_map.keys() if k[0] == aircraft_type],
        'combination_recipes_sample': list(combination_recipes.keys())[:5],
        'nested_task_mhs_lookup': mapped_type in nested_task_mhs_lookup,
        'precomputed_matrices': [k for k in precomputed_embeddings.keys() if mapped_type in k]
    }
    
    if mapped_type in special_tasks_data_map:
        sample_tasks = special_tasks_data_map[mapped_type][:3]
        result['sample_tasks'] = [
            {
                'description': task.get('task_description', 'N/A')[:100],
                'mhs': task.get('total_mhs', 0),
                'has_embedding': 'embedding' in task,
                'has_variations': 'variation_embeddings' in task and bool(task['variation_embeddings'])
            }
            for task in sample_tasks
        ]
    
    return result

async def get_stats() -> Dict[str, Any]:
    """Get comprehensive API statistics"""
    from app.utils.text_processing import expand_abbreviations_cached, extract_component_cached, find_removal_installation_cached, map_special_task_aircraft_type
    
    special_tasks_data_map = get_special_tasks_data_map()
    ad_sb_data_map = get_ad_sb_data_map()
    combination_recipes = get_combination_recipes()
    nested_task_mhs_lookup = get_nested_task_mhs_lookup()
    precomputed_embeddings = get_precomputed_embeddings()
    
    cache_sizes = {}
    try:
        cache_sizes = {
            'expand_abbreviations': expand_abbreviations_cached.cache_info()._asdict() if hasattr(expand_abbreviations_cached, 'cache_info') else {},
            'extract_component': extract_component_cached.cache_info()._asdict() if hasattr(extract_component_cached, 'cache_info') else {},
            'find_removal_installation': find_removal_installation_cached.cache_info()._asdict() if hasattr(find_removal_installation_cached, 'cache_info') else {},
            'map_aircraft_type': map_special_task_aircraft_type.cache_info()._asdict() if hasattr(map_special_task_aircraft_type, 'cache_info') else {}
        }
    except:
        cache_sizes = {}
    
    return {
        'special_tasks_aircraft_types': list(special_tasks_data_map.keys()),
        'ad_sb_combinations': len(ad_sb_data_map),
        'combination_recipes': len(combination_recipes),
        'task_mhs_lookups': len(nested_task_mhs_lookup),
        'precomputed_matrices': len(precomputed_embeddings),
        'embedding_dimensions': list(precomputed_embeddings.values())[0].shape[1] if precomputed_embeddings else 0,
        'cache_sizes': cache_sizes
    }

async def clear_cache() -> Dict[str, Any]:
    """Clear LRU caches to free memory"""
    from app.utils.text_processing import expand_abbreviations_cached, extract_component_cached, find_removal_installation_cached, map_special_task_aircraft_type
    from app.utils.search_utils import should_exclude_match, extract_component_type
    
    expand_abbreviations_cached.cache_clear()
    extract_component_cached.cache_clear()
    find_removal_installation_cached.cache_clear()
    map_special_task_aircraft_type.cache_clear()
    should_exclude_match.cache_clear()
    extract_component_type.cache_clear()
    
    return {
        'status': 'success',
        'message': 'All caches cleared'
    }