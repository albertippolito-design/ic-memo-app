"""
Calibration configuration for property development simulation
"""
from dataclasses import dataclass

@dataclass
class CalibrationTargets:
    """Target metrics for calibrated simulation"""
    target_win_rate: float = 0.55  # 55% target win rate
    target_irr_p50_wins: float = 0.175  # 17.5% median IRR for wins
    target_irr_p25_wins: float = 0.10   # 10% P25 IRR for wins
    target_irr_p75_wins: float = 0.25   # 25% P75 IRR for wins
    target_moic_p50_wins: float = 1.60  # 1.6x median MOIC for wins
    target_moic_p25_wins: float = 1.40  # 1.4x P25 MOIC for wins
    target_moic_p75_wins: float = 1.80  # 1.8x P75 MOIC for wins
    max_conservation_violations: int = 0  # Zero tolerance
    
    # For display in reports
    def __post_init__(self):
        # Ensure all percentile attributes exist
        if not hasattr(self, 'target_irr_p25_wins'):
            self.target_irr_p25_wins = 0.10
        if not hasattr(self, 'target_irr_p75_wins'):
            self.target_irr_p75_wins = 0.25
        if not hasattr(self, 'target_moic_p25_wins'):
            self.target_moic_p25_wins = 1.40
        if not hasattr(self, 'target_moic_p75_wins'):
            self.target_moic_p75_wins = 1.80

CALIBRATION_TARGETS = CalibrationTargets()

@dataclass
class StressConfig:
    """Realistic stress parameters (calibrated)"""
    # Event probability (reduced from ~10% to 2-3%)
    event_probability_per_month: float = 0.025  # 2.5% per month
    
    # Rate shocks (reduced from 500bp+ to 100-250bp)
    rate_shock_mean_bp: int = 100  # 100bp mean
    rate_shock_max_bp: int = 250   # 250bp max
    
    # Sales shocks (reduced from 70-90% drops to 15-40%)
    sales_drop_min_pct: float = 0.15  # 15% min drop
    sales_drop_max_pct: float = 0.40  # 40% max drop
    
    # Construction overruns (reduced from large to 5-15%)
    construction_overrun_min_pct: float = 0.05  # 5% min
    construction_overrun_max_pct: float = 0.15  # 15% max
    
    # Event duration (shorter)
    event_duration_min_months: int = 2
    event_duration_max_months: int = 6
    
    # Resolve cost (proportional to burn, not fixed)
    resolve_cost_months_of_burn: float = 1.5  # 1.5 months of burn
    resolve_cost_cap_pct_capex: float = 0.03  # Cap at 3% of remaining capex
    
    # Revolver / working capital
    revolver_capacity: int = 20_000_000  # €20m revolver
    revolver_spread_bp: int = 200  # 200bp over base rate
    
    # Smooth bank gate
    bank_gate_base_availability: int = 100_000_000  # €100m base
    bank_gate_slope: float = 10.0  # €10 debt per €1 deposit

STRESS_CONFIG = StressConfig()
