from concurrent.futures import ThreadPoolExecutor
from app.utils.text_processing import normalize_task_description, extract_component_cached, find_removal_installation_cached, map_special_task_aircraft_type
from app.utils.search_utils import exact_match_search_special, search_special_tasks_enhanced, search_ad_sb_enhanced, search_combination_recipes
from app.core.maintenance_config import MAX_WORKERS

def process_single_task_enhanced(task_index, task, embedding, aircraft_type, aircraft_reg):
    """ENHANCED: Combine old and new code processing logic - NO AGE FIELDS"""
    task_desc = normalize_task_description(task.get('task_description', ''))
    task_num = task.get('task_number', '')

    if not task_desc:
        return (task_index, {
            "task_number": task_num,
            "task_description": task_desc,
            "total_mhs": 0.0,
            "status": "not available",
            "task_type": "unknown",
            "similarity_score": 0.0,
            "best_match_found": "No task description provided",
            "matched_description": None,
            "combined_from": []
        })

    all_matches = []

    # Step 1: PRESERVED - Old code exact match logic
    try:
        exact_match = exact_match_search_special(task_desc, aircraft_type)
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

    # Step 2: Enhanced special tasks search
    try:
        special_matches = search_special_tasks_enhanced(embedding, aircraft_type, task_desc, threshold=0.75, top_k=5)
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

    # Step 3: PRESERVED - Old code AD/SB search logic
    try:
        if aircraft_reg:
            adsb_matches = search_ad_sb_enhanced(embedding, aircraft_type, aircraft_reg, task_desc, threshold=0.75, top_k=3)
            for match in adsb_matches:
                all_matches.append({
                    'total_mhs': match['total_mhs'],
                    'similarity': match['similarity'],
                    'description': match['best_description'],
                    'source': 'ad_sb',
                    'combined_from': [],
                    'task_number': match.get('task_number', ''),
                    'source_aircraft': f"{match.get('aircraft_type', '')} - {match.get('aircraft_reg', '')}"
                })
    except Exception as e:
        print(f"Error in AD/SB search: {e}")

    # Step 4: PRESERVED - Old code sum logic for removal + installation
    try:
        has_good_match = any(m['similarity'] >= 0.85 and m['total_mhs'] > 0 for m in all_matches)
        if not has_good_match:
            # First try combination recipes
            recipe_match = search_combination_recipes(task_desc, aircraft_type)
            if recipe_match and recipe_match['total_mhs'] > 0:
                all_matches.append({
                    'total_mhs': recipe_match['total_mhs'],
                    'similarity': recipe_match['similarity'],
                    'description': recipe_match['task_description'],
                    'source': 'combination_recipe',
                    'combined_from': recipe_match.get('combined_from', [])
                })
            else:
                # PRESERVED: Old code's sum logic - try to find removal + installation
                component = extract_component_cached(task_desc)
                if component:
                    mapped_type = map_special_task_aircraft_type(aircraft_type)
                    from app.utils.maintenance_utils import get_nested_task_mhs_lookup
                    nested_task_mhs_lookup = get_nested_task_mhs_lookup()
                    
                    if mapped_type in nested_task_mhs_lookup:
                        tasks_info, mhs_info, total_mhs = find_removal_installation_cached(component, mapped_type)
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

    # Find the best match with old code's sorting logic
    if all_matches:
        # PRESERVED: Old code sorting by similarity first, then by MHS
        all_matches.sort(key=lambda x: (x['similarity'], x['total_mhs']), reverse=True)
        best_match = all_matches[0]
        
        # PRESERVED: Old code status determination
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
            # Removed age_difference and matched_age as requested
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

def ultra_fast_batch_processing_ordered(tasks_batch, embeddings_batch, aircraft_type, aircraft_reg):
    """PRESERVED: Ultra-fast batch processing with order preservation from old code"""
    if len(tasks_batch) <= 10:
        # Process sequentially for small batches
        results = []
        for i in range(len(tasks_batch)):
            task_index, result = process_single_task_enhanced(i, tasks_batch[i], embeddings_batch[i], aircraft_type, aircraft_reg)
            results.append((task_index, result))
        return results
    else:
        # Process in parallel but preserve order
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks_batch))) as executor:
            futures = []
            for i in range(len(tasks_batch)):
                future = executor.submit(process_single_task_enhanced, i, tasks_batch[i], embeddings_batch[i], aircraft_type, aircraft_reg)
                futures.append((i, future))
            
            # Collect results in order
            results = [None] * len(futures)
            for original_index, future in futures:
                task_index, result = future.result()
                results[original_index] = (task_index, result)
            
            return results