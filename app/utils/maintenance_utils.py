import numpy as np
import pandas as pd
import re
from typing import List, Dict, Any

class MaintenanceUtils:
    """Utility functions for maintenance prediction"""
    
    @staticmethod
    def normalize_task_description(text):
        """Normalize task description text"""
        if not text:
            return ""
        text = str(text).strip()
        return re.sub(r'\s+', ' ', text)
    
    @staticmethod
    def float_round(value) -> float:
        """Round float values safely"""
        if pd.notna(value):
            return round(float(value), 2)
        return 0
    
    @staticmethod
    def replace_nan_inf(obj):
        """Replace NaN and Inf values for JSON serialization"""
        if isinstance(obj, dict):
            return {k: MaintenanceUtils.replace_nan_inf(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [MaintenanceUtils.replace_nan_inf(v) for v in obj]
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return 0.0
            return obj
        return obj
    
    @staticmethod
    def safe_get(data, key, default=None):
        """Safely get value from dictionary"""
        return data.get(key, default) if isinstance(data, dict) else default