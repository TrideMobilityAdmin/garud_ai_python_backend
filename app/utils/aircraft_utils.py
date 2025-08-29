from functools import lru_cache

@lru_cache(maxsize=1000)
def map_special_task_aircraft_type(user_type):
    """PRESERVED: Old code aircraft type mapping"""
    if 'A320' in str(user_type) or 'A321' in str(user_type):
        return 'A320 & B737'
    return user_type