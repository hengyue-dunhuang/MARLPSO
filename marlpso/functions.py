#!/usr/bin/env python3
"""
Optimization Functions Library
Provides a clean interface for various benchmark functions using the opfunu library.
Supports various CEC benchmark suites.
"""

import numpy as np
from typing import Tuple, Dict, List, Union, Optional

class OptimizationFunction:
    """Base class for optimization functions"""
    
    def __init__(self, name: str, bounds: Union[Tuple[float, float], np.ndarray], 
                 global_minimum: float, recommended_dim: int = 30):
        self.name = name
        self.bounds = bounds
        self.global_minimum = global_minimum
        self.recommended_dim = recommended_dim
    
    def __call__(self, x: np.ndarray) -> float:
        """Call the function for optimization"""
        return self.evaluate(x)
    
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate function value, to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_info(self) -> Dict:
        """Get function information"""
        return {
            "name": self.name,
            "bounds": self.bounds,
            "global_minimum": self.global_minimum,
            "recommended_dim": self.recommended_dim
        }

class CECFunction(OptimizationFunction):
    """Generic CEC function wrapper supporting multiple CEC benchmark suites via opfunu"""
    
    # Supported CEC benchmark suites
    SUPPORTED_CEC_SUITES = {
        2005: {'module': 'cec2005', 'functions': list(range(1, 26))},
        2008: {'module': 'cec2008', 'functions': list(range(1, 7))},
        2010: {'module': 'cec2010', 'functions': list(range(1, 21))},
        2013: {'module': 'cec2013', 'functions': list(range(1, 29))},
        2014: {'module': 'cec2014', 'functions': list(range(1, 31))},
        2015: {'module': 'cec2015', 'functions': list(range(1, 16))},
        2017: {'module': 'cec2017', 'functions': list(range(1, 31))},
        2019: {'module': 'cec2019', 'functions': list(range(1, 11))},
        2020: {'module': 'cec2020', 'functions': list(range(1, 11))},
        2021: {'module': 'cec2021', 'functions': list(range(1, 11))},
        2022: {'module': 'cec2022', 'functions': list(range(1, 13))},
    }
    
    def __init__(self, year: int, function_id: int, dim: int = 30):
        try:
            if year not in self.SUPPORTED_CEC_SUITES:
                raise ValueError(f"Unsupported CEC year: {year}. Supported years: {list(self.SUPPORTED_CEC_SUITES.keys())}")
            
            suite_info = self.SUPPORTED_CEC_SUITES[year]
            if function_id not in suite_info['functions']:
                raise ValueError(f"CEC{year} does not support function ID {function_id}. Available IDs: {suite_info['functions']}")
            
            # Dynamically import the corresponding CEC module
            module_name = f"opfunu.cec_based.{suite_info['module']}"
            cec_module = __import__(module_name, fromlist=[''])
            
            func_name = f'F{function_id}{year}'
            if hasattr(cec_module, func_name):
                func_class = getattr(cec_module, func_name)
                self.opfunu_func = func_class(ndim=dim)
            else:
                raise ValueError(f"CEC{year} function {func_name} not found")
            
            self.year = year
            self.function_id = function_id
            self.dim = dim
            
            # Get bounds
            if hasattr(self.opfunu_func, 'lb') and hasattr(self.opfunu_func, 'ub'):
                bounds = (float(self.opfunu_func.lb[0]), float(self.opfunu_func.ub[0]))
            else:
                bounds = (-100.0, 100.0)
            
            # Get global minimum
            if hasattr(self.opfunu_func, 'f_global'):
                global_min = float(self.opfunu_func.f_global)
            else:
                global_min = 0.0
            
            super().__init__(
                name=f"cec{year}_f{function_id}",
                bounds=bounds,
                global_minimum=global_min,
                recommended_dim=dim
            )
            
        except ImportError as e:
            if "opfunu" in str(e):
                raise ImportError("opfunu library is required: pip install opfunu")
            else:
                raise ImportError(f"Failed to import CEC{year} module: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CEC{year} F{function_id}: {e}")
    
    def evaluate(self, x: np.ndarray) -> float:
        if len(x) != self.dim:
            raise ValueError(f"CEC{self.year} F{self.function_id} requires {self.dim} dimensions, got {len(x)}")
        return float(self.opfunu_func.evaluate(x))

def get_cec_function(year: int, function_id: int, dim: int = 30) -> CECFunction:
    """Get a CEC function instance"""
    return CECFunction(year, function_id, dim)

def list_cec_functions(year: Optional[int] = None) -> Dict[int, List[int]]:
    """List available CEC functions"""
    try:
        if year is not None:
            if year not in CECFunction.SUPPORTED_CEC_SUITES:
                raise ValueError(f"Unsupported CEC year: {year}")
            
            suite_info = CECFunction.SUPPORTED_CEC_SUITES[year]
            module_name = f"opfunu.cec_based.{suite_info['module']}"
            
            try:
                cec_module = __import__(module_name, fromlist=[''])
                available_ids = []
                for func_id in suite_info['functions']:
                    func_name = f'F{func_id}{year}'
                    if hasattr(cec_module, func_name):
                        available_ids.append(func_id)
                return {year: available_ids}
            except ImportError:
                return {year: []}
        else:
            all_functions = {}
            for y in CECFunction.SUPPORTED_CEC_SUITES.keys():
                all_functions.update(list_cec_functions(y))
            return all_functions
    except Exception:
        return {}

def get_cec_info(year: Optional[int] = None) -> Dict:
    """Get CEC benchmark suite information"""
    info = {}
    if year is not None:
        if year in CECFunction.SUPPORTED_CEC_SUITES:
            suite_info = CECFunction.SUPPORTED_CEC_SUITES[year]
            available_funcs = list_cec_functions(year).get(year, [])
            info[year] = {
                'module': suite_info['module'],
                'total_functions': len(suite_info['functions']),
                'available_functions': len(available_funcs),
                'function_ids': available_funcs
            }
    else:
        for y in CECFunction.SUPPORTED_CEC_SUITES.keys():
            info.update(get_cec_info(y))
    return info
