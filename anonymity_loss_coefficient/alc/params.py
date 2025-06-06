from typing import Any

class _ParamGroup:
    """A simple namespace for parameter grouping."""
    # Type hints
    prc_abs_weight: float  
    halt_thresh_low: float 
    halt_thresh_high: float 
    halt_interval_thresh: float 
    halt_min_significant_attack_prcs: int 
    halt_min_prc_improvement: float 
    halt_check_count: int 
    si_type: str 
    si_confidence: float 
    max_score_interval: float 
    prc_abs_weight: float 
    recall_adjust_min_intercept: float 
    recall_adjust_strength: float 
    disc_max: int 
    disc_bins: int 
    discretize_in_place: bool 
    max_cntl_size: int 
    max_cntl_percent: float 
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class ALCParams:
    """Class to manage constants and defaults for ALC (Anonymity Loss Coefficient) calculations."""
    def __init__(self):
        # Initialize with default groups
        self.alcm = _ParamGroup(
            halt_thresh_low=0.25,
            halt_thresh_high=0.9,
            halt_interval_thresh=0.1,
            halt_min_significant_attack_prcs=3,  # Must be at least 3
            halt_min_prc_improvement=0.01,
            halt_check_count=20,
        )
        self.si = _ParamGroup(
            si_type='wilson_score_interval',
            si_confidence=0.95,
            max_score_interval=0.5,
        )
        self.alc = _ParamGroup(
            prc_abs_weight=0.0,
            recall_adjust_min_intercept=1/10000,
            recall_adjust_strength=3.0,
        )
        self.df = _ParamGroup(
            disc_max=50,
            disc_bins=20,
            discretize_in_place=False,
            max_cntl_size=1000,
            max_cntl_percent=0.1,
        )

    def set_param(self, group, param_name: str, value: Any):
        """Set a parameter in a given group by name."""
        if value is None:
            return
        setattr(group, param_name, value)

    def add_group(self, group_name: str, **params):
        """Dynamically add a new parameter group."""
        setattr(self, group_name, _ParamGroup(**params))

    def iter_params(self):
        """Yield tuples for group, param, and value for all groups and parameters."""
        for group_name, group in self.__dict__.items():
            if isinstance(group, _ParamGroup):
                for param, value in group.__dict__.items():
                    yield (str(group_name), str(param), value)

# Example usage:
# ap = ALCParams()
# ap.add_group('mygroup', foo=1, bar=2)
# print(ap.mygroup.foo)
# ap.mygroup.foo = 42
