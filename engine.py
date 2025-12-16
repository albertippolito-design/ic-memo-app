"""
Liquidity Crunch - Game Engine (Pure Logic, No UI)
===================================================
Headless financial simulation engine for property development game.
Used by both pygame desktop version and Streamlit cloud deployment.
"""

import random
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy import optimize

# Import calibration configuration
try:
    from calibration_config import CALIBRATION_TARGETS, STRESS_CONFIG
except ImportError:
    # Fallback if config file not found
    @dataclass
    class CalibrationTargets:
        target_win_rate: float = 0.55
        target_irr_p50_wins: float = 0.175
        target_irr_p25_wins: float = 0.15
        target_irr_p75_wins: float = 0.20
        target_moic_p50_wins: float = 1.60
        target_moic_p25_wins: float = 1.50
        target_moic_p75_wins: float = 1.70
        max_conservation_violations: int = 0
    
    @dataclass
    class StressConfig:
        event_probability_per_month: float = 0.025
        rate_shock_mean_bp: int = 100
        rate_shock_max_bp: int = 250
        sales_drop_min_pct: float = 0.15
        sales_drop_max_pct: float = 0.40
        construction_overrun_min_pct: float = 0.05
        construction_overrun_max_pct: float = 0.15
        event_duration_min_months: int = 2
        event_duration_max_months: int = 6
        resolve_cost_months_of_burn: float = 1.5
        resolve_cost_cap_pct_capex: float = 0.03
        revolver_capacity: int = 20_000_000
        revolver_spread_bp: int = 200
        bank_gate_base_availability: int = 100_000_000
        bank_gate_slope: float = 10.0
    
    CALIBRATION_TARGETS = CalibrationTargets()
    STRESS_CONFIG = StressConfig()


# ==================== Constants ====================

# Project Economics
TOTAL_EQUITY = 150_000_000  # €150m
LAND_COST = 100_000_000     # €100m
DEV_COSTS = 600_000_000     # €600m
CONSTRUCTION_MONTHS = 30
MAX_DEBT_FACILITY = 600_000_000  # €600m max

# Unit Economics
TOTAL_UNITS = 1000
UNIT_PRICE = 1_000_000      # €1m per unit
GDV = TOTAL_UNITS * UNIT_PRICE  # €1bn

# Payment Structure
DEPOSIT_1_PCT = 0.15        # 15% on contract
DEPOSIT_2_PCT = 0.15        # 15% at 12 months
COMPLETION_PCT = 0.70       # 70% at completion

# Bank Gate Requirements
BANK_GATE_UNITS_PCT = 0.15  # 15% units sold
BANK_GATE_DEPOSITS = 30_000_000  # €30m deposits

# Revolver
MAX_REVOLVER = STRESS_CONFIG.revolver_capacity


# ==================== Enums ====================

class FailureReason(Enum):
    """Categorizes why a simulation failed"""
    LIQUIDITY_ZERO = "Cash exhausted"
    RUNWAY_TOO_LOW = "Cash runway < 1 month, no financing"
    BANK_GATE_NOT_MET = "Bank gate thresholds not met"
    DEBT_CAP_EXCEEDED = "Debt capacity exhausted"
    RATE_SHOCK = "Interest expense spike"
    CONSTRUCTION_OVERRUN = "Construction cost blowout"
    SALES_STALL = "Sales pace too low vs burn"
    EVENT_STACK = "Multiple adverse events"
    TIMEOUT = "Max months without completion"
    OTHER = "Other causes"


class IRRStatus(Enum):
    """Status of IRR calculation"""
    VALID = "valid"
    NO_SIGN_CHANGE = "no_sign_change"
    DID_NOT_CONVERGE = "did_not_converge"


# ==================== Data Classes ====================

@dataclass
class StressRunMetrics:
    """Tracks all stresses that occurred during a simulation run"""
    # Construction delays
    had_delay: bool = False
    delay_months: int = 0
    
    # Capex overruns
    had_capex_overrun: bool = False
    capex_overrun_pct: float = 0.0
    capex_overrun_amount: float = 0.0
    
    # Interest rate shocks
    had_rate_shock: bool = False
    max_rate_bps: int = 0
    rate_shock_months: int = 0
    
    # Sales shocks
    had_sales_shock: bool = False
    sales_shock_months: int = 0
    min_sales_multiplier: float = 1.0
    
    # Event costs
    had_event_cost: bool = False
    total_event_cost: float = 0.0
    event_count: int = 0
    
    # Bank gate blocked
    bank_gate_blocked_months: int = 0


@dataclass
class UnitSale:
    """Represents a single unit sale"""
    sale_month: int
    deposit_1_paid: bool = False
    deposit_2_paid: bool = False
    completion_paid: bool = False


@dataclass
class EventCard:
    """Represents an active stress event"""
    name: str
    months_left: int
    description: str = ""
    dev_cost_mult: float = 1.0
    sales_mult: float = 1.0
    progress_mult: float = 1.0
    rate_add: float = 0.0
    debt_lock: bool = False
    one_off_cost: int = 0
    
    def tick(self):
        """Decrement duration"""
        self.months_left -= 1


@dataclass
class SimulationSettings:
    """Configuration for simulation runs"""
    events_enabled: bool = True
    event_frequency_mult: float = 1.0
    sales_volatility_mult: float = 1.0
    deposit_gate_mult: float = 1.0
    
    @staticmethod
    def baseline():
        return SimulationSettings()


@dataclass
class GameState:
    """Complete game state (pure data, no UI)"""
    # Time
    month: int = 0
    
    # Project
    phase: str = "LAND"
    progress: float = 0.0  # 0..100
    units_total: int = TOTAL_UNITS
    units_sold: int = 0
    unit_sales: list = field(default_factory=list)  # List of UnitSale objects
    
    # Finance
    cash: int = TOTAL_EQUITY  # Start with total equity
    debt: int = 0
    max_debt: int = MAX_DEBT_FACILITY
    equity_total: int = TOTAL_EQUITY
    equity_used: int = 0  # Will be set to LAND_COST after land purchase
    equity_remaining: int = TOTAL_EQUITY
    
    base_rate: float = 0.06  # annual
    monthly_dev_cost: int = int(DEV_COSTS / CONSTRUCTION_MONTHS)  # €20m/month
    
    deposits_collected: int = 0  # Total deposits received (CASH inflow, but LIABILITY)
    deposits_liability: int = 0  # Amount owed to buyers (released at completion)
    bank_gate_met: bool = False
    completion_triggered: bool = False  # Flag to prevent double-processing completion
    
    # Revolver facility
    revolver_drawn: int = 0
    max_revolver: int = MAX_REVOLVER
    
    # Cost tracking for conservation checks
    total_costs_incurred: int = 0  # Running total of all costs paid
    total_interest_paid: int = 0  # Running total of interest expenses
    peak_debt: int = 0  # Maximum debt balance reached
    
    # Comprehensive cash tracking (Sources & Uses)
    # SOURCES
    equity_in_total: int = 0  # Cumulative equity injected
    debt_drawn_total: int = 0  # Cumulative debt drawn (lifetime)
    debt_outstanding: int = 0  # Current debt balance (alias for debt)
    sales_cash_received_total: int = 0  # Total cash from sales (deposits + deed)
    deposits_collected_total: int = 0  # Total deposit cash collected
    
    # USES
    land_cost_paid: int = 0
    construction_cost_paid: int = 0
    interest_paid_total: int = 0
    debt_principal_repaid_total: int = 0
    event_costs_paid_total: int = 0
    revolver_drawn_total: int = 0  # Cumulative revolver draws
    revolver_repaid_total: int = 0  # Cumulative revolver repayments
    
    # Difficulty / ramp
    difficulty: int = 0  # increases over time
    event_chance: float = 0.20  # base monthly
    severity: float = 1.0
    
    # Active modifiers (actions)
    slow_build: bool = False
    slow_build_months_left: int = 0
    
    sales_push: bool = False
    sales_push_months_left: int = 0
    
    # Event(s)
    active_events: list = field(default_factory=list)
    
    # Multipliers from events
    sales_mult: float = 1.0
    progress_mult: float = 1.0
    rate_add: float = 0.0
    min_rate_add: float = 0.0  # Track minimum (most negative) rate add
    max_rate_add: float = 0.0  # Track maximum (most positive) rate add
    dev_cost_mult: float = 1.0  # Added for construction cost impacts
    
    # Score / tracking
    near_miss_count: int = 0
    last_net_cf: int = 0
    last_month_burn: int = 0  # Track monthly burn for runway calculation
    history_cash: list = field(default_factory=list)
    
    # Diagnostics
    cash_trough: int = TOTAL_EQUITY  # Track minimum cash reached
    max_concurrent_events: int = 0
    total_event_months: int = 0
    
    # Equity cashflows for IRR/MOIC calculation
    equity_cashflows: list = field(default_factory=list)  # [{'month': 0, 'amount': -100m}, ...]
    
    # Stress tracking
    stresses: StressRunMetrics = field(default_factory=StressRunMetrics)
    
    # End state
    game_over: bool = False
    win: bool = False
    reason: str = ""
    failure_reason: Optional[FailureReason] = None
    failure_details: dict = field(default_factory=dict)


# ==================== Helper Functions ====================

def clamp(x, a, b):
    """Clamp value between min and max"""
    return max(a, min(b, x))


def money(x):
    """Format number as money string"""
    if abs(x) >= 1_000_000:
        return f"€{x/1_000_000:.1f}M"
    return f"€{x:,.0f}"


# ==================== Engine Class ====================

class Engine:
    """Pure game logic engine (no UI dependencies)"""
    
    def __init__(self, settings: Optional[SimulationSettings] = None, seed: Optional[int] = None):
        self.gs = GameState()
        self.settings = settings if settings else SimulationSettings.baseline()
        self._seed = seed
        
        if seed is not None:
            random.seed(seed)
        
        # Initialize game
        self.apply_land_purchase()
    
    def apply_land_purchase(self):
        """Initialize project by purchasing land"""
        gs = self.gs
        gs.cash -= LAND_COST
        gs.equity_used = LAND_COST
        gs.equity_remaining -= LAND_COST
        gs.phase = "BUILD"
        gs.reason = f"Land purchased for {money(LAND_COST)}. Construction begins."
        
        # Track comprehensive sources & uses
        gs.land_cost_paid = LAND_COST
        gs.total_costs_incurred = LAND_COST
        gs.equity_in_total = TOTAL_EQUITY  # Initial equity injection
        
        # Record initial equity outflow
        gs.equity_cashflows.append({'month': 0, 'amount': -TOTAL_EQUITY})
    
    def current_rate(self):
        """Calculate current interest rate including event impacts"""
        rate = self.gs.base_rate
        for ev in self.gs.active_events:
            rate += ev.rate_add
        return max(0.0, rate)
    
    def has_debt_lock(self):
        """Check if any active event prevents debt raising"""
        return any(ev.debt_lock for ev in self.gs.active_events)
    
    def check_bank_gate(self):
        """Check if bank financing gate conditions are met"""
        gs = self.gs
        units_threshold = int(TOTAL_UNITS * BANK_GATE_UNITS_PCT)  # 150 units
        deposits_threshold = int(BANK_GATE_DEPOSITS * self.settings.deposit_gate_mult)
        
        units_ok = gs.units_sold >= units_threshold
        deposits_ok = gs.deposits_collected >= deposits_threshold
        
        if units_ok and deposits_ok and not gs.bank_gate_met:
            gs.bank_gate_met = True
            gs.reason = f"🏦 Bank financing UNLOCKED! {gs.units_sold} units sold, {money(gs.deposits_collected)} deposits collected."
        
        return gs.bank_gate_met
    
    def combined_multipliers(self):
        """Calculate combined multipliers from all active effects"""
        dev_cost_mult = 1.0
        sales_mult = 1.0
        progress_mult = 1.0
        
        for ev in self.gs.active_events:
            dev_cost_mult *= ev.dev_cost_mult
            sales_mult *= ev.sales_mult
            progress_mult *= ev.progress_mult
        
        # Actions
        if self.gs.slow_build and self.gs.slow_build_months_left > 0:
            dev_cost_mult *= 0.70
            progress_mult *= 0.70
        
        if self.gs.sales_push and self.gs.sales_push_months_left > 0:
            sales_mult *= 1.60
        
        return dev_cost_mult, sales_mult, progress_mult
    
    def base_sales_per_month(self):
        """Calculate base sales velocity"""
        base = 35.0  # ~35 units/month baseline
        base *= max(0.6, 1.0 - 0.03 * self.gs.difficulty)
        return base
    
    def monthly_construction_cost(self):
        """Calculate monthly construction cost with multipliers"""
        dev_cost_mult, _, _ = self.combined_multipliers()
        return int(self.gs.monthly_dev_cost * dev_cost_mult)
    
    def roll_event(self):
        """Possibly spawn a random stress event"""
        gs = self.gs
        if gs.phase != "BUILD":
            return
        
        # Skip events if disabled in settings
        if not self.settings.events_enabled:
            return
        
        # Check event probability
        if random.random() > STRESS_CONFIG.event_probability_per_month * self.settings.event_frequency_mult:
            return
        
        # Generate event with STRESS_CONFIG parameters
        event_types = [
            ("construction_overrun", "Construction cost overrun", "Material/labor costs spike."),
            ("sales_drop", "Sales slowdown", "Market demand weakens."),
            ("rate_shock", "Interest rate spike", "Rates jump unexpectedly."),
        ]
        
        event_type, name, desc = random.choice(event_types)
        months = random.randint(STRESS_CONFIG.event_duration_min_months, STRESS_CONFIG.event_duration_max_months)
        
        # Generate parameters from STRESS_CONFIG ranges
        if event_type == "construction_overrun":
            dev_cost_mult = 1.0 + random.uniform(STRESS_CONFIG.construction_overrun_min_pct, STRESS_CONFIG.construction_overrun_max_pct)
            kwargs = {"dev_cost_mult": dev_cost_mult}
        elif event_type == "sales_drop":
            sales_mult = 1.0 - random.uniform(STRESS_CONFIG.sales_drop_min_pct, STRESS_CONFIG.sales_drop_max_pct)
            kwargs = {"sales_mult": sales_mult}
        elif event_type == "rate_shock":
            rate_add = random.uniform(STRESS_CONFIG.rate_shock_mean_bp * 0.5, STRESS_CONFIG.rate_shock_max_bp) / 10000
            kwargs = {"rate_add": rate_add}
        else:
            kwargs = {}
        
        ev = EventCard(
            name=name,
            months_left=months,
            description=desc,
            dev_cost_mult=kwargs.get("dev_cost_mult", 1.0),
            sales_mult=kwargs.get("sales_mult", 1.0),
            progress_mult=kwargs.get("progress_mult", 1.0),
            rate_add=kwargs.get("rate_add", 0.0),
            debt_lock=kwargs.get("debt_lock", False),
            one_off_cost=kwargs.get("one_off_cost", 0),
        )
        
        # Apply one-off cost immediately
        if ev.one_off_cost > 0:
            gs.cash -= ev.one_off_cost
            gs.event_costs_paid_total += ev.one_off_cost
            gs.total_costs_incurred += ev.one_off_cost
            gs.reason = f"Event: {ev.name}. Immediate cost {money(ev.one_off_cost)}."
            
            # Track event stress
            gs.stresses.had_event_cost = True
            gs.stresses.total_event_cost += ev.one_off_cost
            gs.stresses.event_count += 1
        
        gs.active_events.append(ev)
    
    def tick_month(self):
        """Advance game state by one month"""
        gs = self.gs
        if gs.game_over:
            return
        
        gs.month += 1
        
        # Difficulty ramp every 6 months
        if gs.month % 6 == 0:
            gs.difficulty += 1
            gs.severity = 1.0 + 0.10 * gs.difficulty
        
        dev_cost_mult, sales_mult, progress_mult = self.combined_multipliers()
        
        # Track stresses when active (only from events/actions, not natural variance)
        if progress_mult < 1.0 and gs.progress < 100.0:
            # Check if delay is from events or slow_build action (not random variance)
            has_delay_event = any(ev.progress_mult < 1.0 for ev in gs.active_events) or (gs.slow_build and gs.slow_build_months_left > 0)
            if has_delay_event:
                gs.stresses.had_delay = True
                gs.stresses.delay_months += 1
        
        if dev_cost_mult > 1.0 and gs.progress < 100.0:
            # Only track if from events (not natural variance)
            has_capex_event = any(ev.dev_cost_mult > 1.0 for ev in gs.active_events)
            if has_capex_event:
                gs.stresses.had_capex_overrun = True
                overrun_pct = (dev_cost_mult - 1.0) * 100
                if overrun_pct > gs.stresses.capex_overrun_pct:
                    gs.stresses.capex_overrun_pct = overrun_pct
        
        if sales_mult < 1.0:
            # Only track if from events (not natural variance or actions)
            has_sales_event = any(ev.sales_mult < 1.0 for ev in gs.active_events)
            if has_sales_event:
                gs.stresses.had_sales_shock = True
                gs.stresses.sales_shock_months += 1
                if sales_mult < gs.stresses.min_sales_multiplier:
                    gs.stresses.min_sales_multiplier = sales_mult
        
        # Track rate stress (only from events, not base rate)
        total_rate_add = sum(ev.rate_add for ev in gs.active_events)
        if total_rate_add > 0:
            gs.stresses.had_rate_shock = True
            gs.stresses.rate_shock_months += 1
            rate_add_bps = int(total_rate_add * 10000)
            if rate_add_bps > gs.stresses.max_rate_bps:
                gs.stresses.max_rate_bps = rate_add_bps
        
        # --- NEW UNIT SALES ---
        base_sales = self.base_sales_per_month()
        vol_range = 0.3 * self.settings.sales_volatility_mult
        sales = base_sales * sales_mult * random.uniform(1.0 - vol_range, 1.0 + vol_range)
        sentiment = 1.0 + 0.05 * math.sin(gs.month / 2.5)
        sales *= sentiment
        
        new_sold = int(round(sales))
        new_sold = clamp(new_sold, 0, gs.units_total - gs.units_sold)
        
        # Record new sales
        for _ in range(new_sold):
            unit_sale = UnitSale(sale_month=gs.month)
            gs.unit_sales.append(unit_sale)
        gs.units_sold += new_sold
        
        # --- DEPOSIT COLLECTION ---
        deposit_inflow = 0
        
        # Collect deposit #1 for units sold this month
        deposit_1_amount = int(UNIT_PRICE * DEPOSIT_1_PCT)
        for unit in gs.unit_sales:
            if unit.sale_month == gs.month and not unit.deposit_1_paid:
                deposit_inflow += deposit_1_amount
                unit.deposit_1_paid = True
                gs.deposits_collected += deposit_1_amount
                gs.deposits_liability += deposit_1_amount
                gs.sales_cash_received_total += deposit_1_amount
                gs.deposits_collected_total += deposit_1_amount
        
        # Collect deposit #2 for units sold 12 months ago
        deposit_2_amount = int(UNIT_PRICE * DEPOSIT_2_PCT)
        for unit in gs.unit_sales:
            if unit.sale_month == gs.month - 12 and not unit.deposit_2_paid:
                deposit_inflow += deposit_2_amount
                unit.deposit_2_paid = True
                gs.deposits_collected += deposit_2_amount
                gs.deposits_liability += deposit_2_amount
                gs.sales_cash_received_total += deposit_2_amount
                gs.deposits_collected_total += deposit_2_amount
        
        # --- CONSTRUCTION PROGRESS ---
        base_progress = 100.0 / CONSTRUCTION_MONTHS  # ~3.33% per month
        prog_gain = base_progress * progress_mult * random.uniform(0.85, 1.15)
        gs.progress = clamp(gs.progress + prog_gain, 0.0, 100.0)
        
        # --- COMPLETION PAYMENTS (only ONCE when construction completes) ---
        completion_inflow = 0
        if not gs.completion_triggered and (gs.month == CONSTRUCTION_MONTHS or (gs.progress >= 100.0 and gs.month >= CONSTRUCTION_MONTHS)):
            # Construction complete! All deed payments arrive at once
            deposit_2_amount = int(UNIT_PRICE * DEPOSIT_2_PCT)
            total_deed_inflow = 0
            deposits_cleared_count = 0
            
            for unit in gs.unit_sales:
                if unit.completion_paid:
                    continue
                
                # Collect deed payment (70%)
                deed = int(UNIT_PRICE * COMPLETION_PCT)
                
                # If deposit #2 wasn't paid yet, collect it now
                if not unit.deposit_2_paid:
                    deed += deposit_2_amount
                    unit.deposit_2_paid = True
                    gs.deposits_collected += deposit_2_amount
                    gs.deposits_liability += deposit_2_amount
                    gs.sales_cash_received_total += deposit_2_amount
                    gs.deposits_collected_total += deposit_2_amount
                
                total_deed_inflow += deed
                gs.sales_cash_received_total += int(UNIT_PRICE * COMPLETION_PCT)
                
                # Mark unit as completed and CLEAR its deposits from liability
                unit.completion_paid = True
                deposit_to_clear = int(UNIT_PRICE * (DEPOSIT_1_PCT + DEPOSIT_2_PCT))
                gs.deposits_liability -= deposit_to_clear
                deposits_cleared_count += 1
            
            # Pay down debt from completion proceeds
            debt_repayment = min(gs.debt, total_deed_inflow)
            gs.debt -= debt_repayment
            gs.debt_outstanding = gs.debt
            gs.debt_principal_repaid_total += debt_repayment
            
            # Net inflow after debt repayment
            completion_inflow = total_deed_inflow - debt_repayment
            gs.completion_triggered = True
            
            gs.reason = f"🎉 COMPLETION! Deed inflow: {money(total_deed_inflow)}. Debt repaid: {money(debt_repayment)}. Net: {money(completion_inflow)}."
        
        # --- CONSTRUCTION COSTS ---
        dev_cost = self.monthly_construction_cost() if gs.progress < 100.0 else 0
        if dev_cost > 0:
            gs.total_costs_incurred += dev_cost
            gs.construction_cost_paid += dev_cost
            
            # Track capex overrun amount
            baseline_cost = gs.monthly_dev_cost
            if dev_cost > baseline_cost:
                overrun_amount = dev_cost - baseline_cost
                gs.stresses.capex_overrun_amount += overrun_amount
        
        # --- INTEREST ---
        rate = self.current_rate()
        interest = int(gs.debt * rate / 12.0)
        
        # Revolver interest (higher spread)
        revolver_rate = rate + STRESS_CONFIG.revolver_spread_bp / 10000.0
        revolver_interest = int(gs.revolver_drawn * revolver_rate / 12.0)
        
        total_interest = interest + revolver_interest
        if total_interest > 0:
            gs.total_interest_paid += total_interest
            gs.total_costs_incurred += total_interest
            gs.interest_paid_total += total_interest
        
        interest = total_interest
        
        # --- APPLY CASH FLOWS ---
        net = deposit_inflow + completion_inflow - dev_cost - interest
        gs.cash += net
        gs.last_net_cf = net
        gs.last_month_burn = net
        
        # Check bank gate
        self.check_bank_gate()
        
        # Action durations tick down
        if gs.slow_build_months_left > 0:
            gs.slow_build_months_left -= 1
            if gs.slow_build_months_left == 0:
                gs.slow_build = False
        
        if gs.sales_push_months_left > 0:
            gs.sales_push_months_left -= 1
            if gs.sales_push_months_left == 0:
                gs.sales_push = False
        
        # Event tick down and expire
        for ev in gs.active_events:
            ev.tick()
        gs.active_events = [ev for ev in gs.active_events if ev.months_left > 0]
        
        # Roll new event after applying month flows
        self.roll_event()
        
        # Track cash history and diagnostics
        gs.history_cash.append(gs.cash)
        if len(gs.history_cash) > 120:
            gs.history_cash.pop(0)
        
        if gs.cash < gs.cash_trough:
            gs.cash_trough = gs.cash
        
        if len(gs.active_events) > gs.max_concurrent_events:
            gs.max_concurrent_events = len(gs.active_events)
        gs.total_event_months += len(gs.active_events)
        
        # Near-miss tracking
        if gs.cash < 10_000_000 and gs.cash >= 0:
            gs.near_miss_count += 1
        
        # Lose conditions
        sold_ratio = gs.units_sold / gs.units_total
        
        if gs.cash < 0:
            gs.game_over = True
            gs.win = False
            gs.reason = "Insolvent! Cash < €0. Game over."
            gs.failure_reason = FailureReason.LIQUIDITY_ZERO
            gs.failure_details = {
                'final_cash': gs.cash,
                'final_debt': gs.debt,
                'units_sold': gs.units_sold,
                'deposits_collected': gs.deposits_collected,
                'progress': gs.progress,
                'month': gs.month
            }
            return
        
        # Timeout condition
        if gs.month >= 60 and (gs.progress < 100.0 or sold_ratio < 0.70):
            gs.game_over = True
            gs.win = False
            gs.reason = "Timeout! Project took too long without completion."
            gs.failure_reason = FailureReason.TIMEOUT
            gs.failure_details = {
                'final_cash': gs.cash,
                'final_debt': gs.debt,
                'units_sold': gs.units_sold,
                'deposits_collected': gs.deposits_collected,
                'progress': gs.progress,
                'month': gs.month
            }
            return
        
        # Win condition
        if gs.month >= CONSTRUCTION_MONTHS and gs.progress >= 100.0 and sold_ratio >= 0.70 and gs.cash >= 0:
            gs.game_over = True
            gs.win = True
            gs.reason = "Project complete! 70%+ units sold, debt repaid, positive cash. Victory!"
    
    # ==================== Actions ====================
    
    def action_raise_debt(self):
        """Attempt to draw debt from bank facility"""
        gs = self.gs
        if gs.game_over:
            return
        
        # Check bank gate
        if not gs.bank_gate_met:
            units_needed = int(TOTAL_UNITS * BANK_GATE_UNITS_PCT)
            deposits_needed = BANK_GATE_DEPOSITS
            gs.reason = f"🔒 Bank financing LOCKED. Need {units_needed} units sold ({gs.units_sold} current) and {money(deposits_needed)} deposits ({money(gs.deposits_collected)} collected)."
            gs.stresses.bank_gate_blocked_months += 1
            return
        
        if self.has_debt_lock():
            gs.reason = "Cannot raise debt right now (bank covenants tightened)."
            return
        
        # Check facility headroom
        headroom = gs.max_debt - gs.debt
        if headroom <= 0:
            gs.reason = "No debt capacity left (facility fully drawn)."
            return
        
        # Calculate graduated debt availability
        base_avail = STRESS_CONFIG.bank_gate_base_availability
        slope = STRESS_CONFIG.bank_gate_slope
        deposits_above_gate = max(0, gs.deposits_collected - BANK_GATE_DEPOSITS)
        extra_avail = int(deposits_above_gate * slope)
        max_available = min(base_avail + extra_avail, headroom)
        
        if max_available <= 0:
            gs.reason = "Bank says no additional debt available at this time."
            return
        
        # Draw 50% of available
        draw_amount = int(max_available * 0.50)
        draw_amount = max(10_000_000, draw_amount)  # Minimum €10m
        draw_amount = min(draw_amount, headroom)
        
        gs.debt += draw_amount
        gs.cash += draw_amount
        gs.debt_outstanding = gs.debt
        
        # Track debt drawn
        gs.debt_drawn_total += draw_amount
        
        if gs.debt > gs.peak_debt:
            gs.peak_debt = gs.debt
        
        gs.reason = f"Drew {money(draw_amount)} debt. Debt: {money(gs.debt)}/{money(gs.max_debt)}."
    
    def action_inject_equity(self):
        """Inject additional equity (dilution)"""
        gs = self.gs
        if gs.game_over:
            return
        
        if gs.equity_remaining <= 0:
            gs.reason = "No equity remaining to inject."
            return
        
        inject = min(10_000_000, gs.equity_remaining)  # €10m or remaining
        gs.cash += inject
        gs.equity_used += inject
        gs.equity_total += inject
        gs.equity_remaining -= inject
        
        # Track equity injection
        gs.equity_in_total += inject
        
        # Record equity cashflow for IRR
        gs.equity_cashflows.append({'month': gs.month, 'amount': -inject})
        
        gs.reason = f"Injected {money(inject)} equity. Total equity: {money(gs.equity_total)}. Remaining: {money(gs.equity_remaining)}."
    
    def action_sales_push(self):
        """Activate sales acceleration (temporary)"""
        gs = self.gs
        if gs.game_over or gs.sales_push:
            return
        
        gs.sales_push = True
        gs.sales_push_months_left = 3
        gs.reason = "Sales push activated for 3 months (1.6x sales velocity)."
    
    def action_slow_build(self):
        """Slow construction to reduce costs (temporary)"""
        gs = self.gs
        if gs.game_over or gs.slow_build:
            return
        
        gs.slow_build = True
        gs.slow_build_months_left = 6
        gs.reason = "Slow build activated for 6 months (0.7x cost, 0.7x progress)."
    
    def action_draw_revolver(self):
        """Draw from revolver facility"""
        gs = self.gs
        if gs.game_over:
            return
        
        headroom = gs.max_revolver - gs.revolver_drawn
        if headroom <= 0:
            gs.reason = "Revolver fully drawn."
            return
        
        draw = min(5_000_000, headroom)  # €5m increments
        gs.revolver_drawn += draw
        gs.cash += draw
        
        # Track revolver draws
        gs.revolver_drawn_total += draw
        
        gs.reason = f"Drew {money(draw)} from revolver. Total: {money(gs.revolver_drawn)}/{money(gs.max_revolver)}."
    
    def action_resolve_event(self):
        """Resolve active event by paying resolution cost"""
        gs = self.gs
        if gs.game_over:
            return
        if not gs.active_events:
            gs.reason = "No active event to resolve."
            return
        
        # Resolve the worst event (by impact score)
        def impact_score(e):
            s = 0.0
            s += (e.dev_cost_mult - 1.0) * 3.0
            s += (1.0 - e.sales_mult) * 2.0
            s += (1.0 - e.progress_mult) * 2.0
            s += e.rate_add * 50.0
            s += 1.5 if e.debt_lock else 0.0
            return s * e.months_left
        
        ev = max(gs.active_events, key=impact_score)
        
        # Calculate resolve cost
        monthly_burn = self.monthly_construction_cost() + int(gs.debt * self.current_rate() / 12.0)
        remaining_capex = int(DEV_COSTS * (100 - gs.progress) / 100)
        cost = int(STRESS_CONFIG.resolve_cost_months_of_burn * monthly_burn)
        cost_cap = int(STRESS_CONFIG.resolve_cost_cap_pct_capex * remaining_capex)
        cost = min(cost, max(cost_cap, 5_000_000))  # Min €5m
        
        if gs.cash < cost:
            gs.reason = f"Not enough cash to resolve {ev.name} (need {money(cost)})."
            return
        
        gs.cash -= cost
        gs.event_costs_paid_total += cost
        gs.total_costs_incurred += cost
        gs.active_events.remove(ev)
        gs.reason = f"Resolved {ev.name} for {money(cost)}. Impacts removed."
    
    # ==================== AI Decision Making ====================
    
    def calculate_runway(self):
        """Calculate months of cash runway based on monthly burn"""
        gs = self.gs
        if gs.last_month_burn >= 0:
            return float('inf')  # Positive cashflow = infinite runway
        burn_rate = abs(gs.last_month_burn)
        if burn_rate == 0:
            return float('inf')
        return gs.cash / burn_rate
    
    def ai_decide_action(self):
        """AI decides which action to take this month (if any)"""
        gs = self.gs
        if gs.game_over:
            return
        
        runway = self.calculate_runway()
        
        # Priority: Use revolver for short-term liquidity
        if gs.cash < 20_000_000 and gs.revolver_drawn < gs.max_revolver:
            self.action_draw_revolver()
            return
        
        # Priority 1: Survive (runway < 2 months)
        if runway < 2.0 and runway != float('inf'):
            # Try to raise debt first
            if gs.bank_gate_met and not self.has_debt_lock() and gs.debt < gs.max_debt:
                self.action_raise_debt()
                return
            # Otherwise inject equity
            elif gs.equity_remaining > 0:
                self.action_inject_equity()
                return
        
        # Priority 2: Unlock bank gate if close
        if not gs.bank_gate_met:
            units_threshold = int(TOTAL_UNITS * BANK_GATE_UNITS_PCT)
            deposits_needed = BANK_GATE_DEPOSITS - gs.deposits_collected
            units_needed = units_threshold - gs.units_sold
            
            # If close to unlocking and have cash, push sales
            if (units_needed < 30 or deposits_needed < 10_000_000) and gs.cash > 20_000_000:
                if not gs.sales_push:
                    self.action_sales_push()
                    return
        
        # Priority 3: Manage construction costs if cash tight
        if runway < 4.0 and runway != float('inf') and gs.progress < 90:
            if not gs.slow_build and gs.cash < 50_000_000:
                self.action_slow_build()
                return
        
        # Priority 4: Push sales if lagging
        sold_ratio = gs.units_sold / gs.units_total
        progress_ratio = gs.progress / 100.0
        if progress_ratio > sold_ratio + 0.15 and gs.cash > 15_000_000:
            if not gs.sales_push:
                self.action_sales_push()
                return
        
        # Priority 5: Resolve expensive events
        if gs.active_events and gs.cash > 30_000_000:
            worst_event = max(gs.active_events, key=lambda e: e.months_left * (2.0 - e.sales_mult + e.dev_cost_mult))
            if worst_event.months_left > 2:
                self.action_resolve_event()
                return
    
    # ==================== Calculations ====================
    
    def score(self):
        """Calculate game score"""
        gs = self.gs
        score = gs.cash + gs.deposits_collected - gs.debt - gs.revolver_drawn
        return score
    
    def compute_equity_distribution_end(self):
        """Calculate equity distribution using waterfall"""
        gs = self.gs
        distributable = gs.cash
        
        # Pay off debt
        distributable -= gs.debt
        
        # Pay off revolver
        distributable -= gs.revolver_drawn
        
        # Pay off deposit liabilities
        distributable -= gs.deposits_liability
        
        return max(0, distributable)
    
    def reconcile_sources_uses(self):
        """Reconcile all cash sources and uses"""
        gs = self.gs
        
        # SOURCES
        sources = {
            'equity_in': gs.equity_in_total,
            'debt_drawn': gs.debt_drawn_total,
            'sales_gross': gs.sales_cash_received_total,
            'revolver_drawn': gs.revolver_drawn_total,
        }
        
        # USES
        uses = {
            'land': gs.land_cost_paid,
            'construction': gs.construction_cost_paid,
            'interest': gs.interest_paid_total,
            'debt_repaid': gs.debt_principal_repaid_total,
            'events': gs.event_costs_paid_total,
            'liability_debt_outstanding': gs.debt,
            'liability_revolver_outstanding': gs.revolver_drawn,
            'liability_deposits_liability': gs.deposits_liability,
        }
        
        total_sources = sum(sources.values())
        total_uses = sum(uses.values())
        expected_cash = total_sources - total_uses
        delta = gs.cash - expected_cash
        
        return sources, uses, delta
    
    def calculate_irr_moic(self):
        """Calculate IRR and MOIC from equity cashflows"""
        gs = self.gs
        
        # Add final distribution if game over
        if gs.game_over:
            equity_out = self.compute_equity_distribution_end()
            
            # Check if already recorded
            has_final = any(cf['month'] == gs.month for cf in gs.equity_cashflows if cf['amount'] > 0)
            if not has_final and equity_out > 0:
                gs.equity_cashflows.append({'month': gs.month, 'amount': equity_out})
        
        # Calculate totals
        equity_in = sum(-cf['amount'] for cf in gs.equity_cashflows if cf['amount'] < 0)
        equity_out = sum(cf['amount'] for cf in gs.equity_cashflows if cf['amount'] > 0)
        
        # MOIC
        moic = equity_out / equity_in if equity_in > 0 else 0.0
        
        # IRR calculation
        if len(gs.equity_cashflows) < 2:
            return 0.0, moic, equity_in, equity_out, IRRStatus.NO_SIGN_CHANGE
        
        # Prepare cashflows for IRR (monthly)
        months = [cf['month'] for cf in gs.equity_cashflows]
        amounts = [cf['amount'] for cf in gs.equity_cashflows]
        
        # Check for sign change
        pos = any(a > 0 for a in amounts)
        neg = any(a < 0 for a in amounts)
        if not (pos and neg):
            return 0.0, moic, equity_in, equity_out, IRRStatus.NO_SIGN_CHANGE
        
        # Calculate IRR using scipy
        try:
            # Convert to annual from monthly
            monthly_times = np.array(months) / 12.0  # Convert months to years
            monthly_amounts = np.array(amounts)
            
            def npv(rate):
                return np.sum(monthly_amounts / (1 + rate) ** monthly_times)
            
            result = optimize.newton(npv, 0.1, maxiter=100)
            irr = float(result)
            
            # Sanity check
            if abs(irr) > 10.0:  # IRR > 1000%
                return 0.0, moic, equity_in, equity_out, IRRStatus.DID_NOT_CONVERGE
            
            return irr, moic, equity_in, equity_out, IRRStatus.VALID
            
        except (RuntimeError, ValueError):
            return 0.0, moic, equity_in, equity_out, IRRStatus.DID_NOT_CONVERGE
    
    def diagnose_failure(self):
        """Diagnose why the simulation failed"""
        gs = self.gs
        if gs.win or gs.failure_reason:
            return
        
        # Already categorized
        if gs.failure_reason:
            return
        
        # Analyze failure
        if gs.cash < 0:
            gs.failure_reason = FailureReason.LIQUIDITY_ZERO
        elif gs.month >= 60:
            gs.failure_reason = FailureReason.TIMEOUT
        elif not gs.bank_gate_met:
            gs.failure_reason = FailureReason.BANK_GATE_NOT_MET
        else:
            gs.failure_reason = FailureReason.OTHER


# ==================== Public API ====================

def new_game(seed: Optional[int] = None, settings: Optional[SimulationSettings] = None) -> Engine:
    """Create a new game instance
    
    Args:
        seed: Random seed for reproducibility
        settings: Simulation settings (events, difficulty, etc.)
    
    Returns:
        Engine instance ready to play
    """
    return Engine(settings=settings, seed=seed)


def step_month(engine: Engine, action: Optional[str] = None) -> Engine:
    """Advance game by one month with optional action
    
    Args:
        engine: Current engine instance
        action: Optional action to take ('raise_debt', 'inject_equity', 'sales_push', 'slow_build', 'draw_revolver')
    
    Returns:
        Updated engine instance (same object, mutated)
    """
    # Execute action if specified
    if action == 'raise_debt':
        engine.action_raise_debt()
    elif action == 'inject_equity':
        engine.action_inject_equity()
    elif action == 'sales_push':
        engine.action_sales_push()
    elif action == 'slow_build':
        engine.action_slow_build()
    elif action == 'draw_revolver':
        engine.action_draw_revolver()
    
    # Advance month
    engine.tick_month()
    
    return engine


def is_finished(engine: Engine) -> bool:
    """Check if game is over
    
    Args:
        engine: Engine instance
    
    Returns:
        True if game is finished
    """
    return engine.gs.game_over


def get_results(engine: Engine) -> dict:
    """Get final results from completed game
    
    Args:
        engine: Engine instance (should be finished)
    
    Returns:
        Dictionary with win status, IRR, MOIC, etc.
    """
    irr, moic, equity_in, equity_out, irr_status = engine.calculate_irr_moic()
    
    return {
        'win': engine.gs.win,
        'month': engine.gs.month,
        'cash': engine.gs.cash,
        'debt': engine.gs.debt,
        'units_sold': engine.gs.units_sold,
        'progress': engine.gs.progress,
        'irr': irr,
        'moic': moic,
        'equity_in': equity_in,
        'equity_out': equity_out,
        'irr_status': irr_status.value,
        'reason': engine.gs.reason,
        'failure_reason': engine.gs.failure_reason.value if engine.gs.failure_reason else None,
        'stresses': engine.gs.stresses,
    }


def run_one_simulation(seed: int, settings: Optional[SimulationSettings] = None) -> dict:
    """Run a single headless simulation
    
    Args:
        seed: Random seed for reproducibility
        settings: Optional simulation settings
    
    Returns:
        Dictionary with simulation results
    """
    random.seed(seed)
    engine = new_game(seed=seed, settings=settings)
    
    # Run AI to completion (max 60 months)
    for _ in range(60):
        if engine.gs.game_over:
            break
        engine.ai_decide_action()
        engine.tick_month()
    
    # Diagnose failure if lost
    if engine.gs.game_over and not engine.gs.win:
        engine.diagnose_failure()
    
    # Calculate results
    irr, moic, equity_in, equity_out, irr_status = engine.calculate_irr_moic()
    
    # Calculate equity recovery
    equity_recovery_pct = (equity_out / equity_in * 100) if equity_in > 0 else 0.0
    
    # Conservation check
    gdv_realized = engine.gs.units_sold * UNIT_PRICE
    max_equity_upper_bound = max(0,
        engine.gs.equity_in_total
        + engine.gs.debt_drawn_total
        + gdv_realized
        - engine.gs.total_costs_incurred
        - engine.gs.debt_principal_repaid_total
        - engine.gs.debt
        - engine.gs.deposits_liability
    )
    
    conservation_violated = False
    violation_margin = equity_out - max_equity_upper_bound
    
    if violation_margin > 500_000:
        conservation_violated = True
    
    return {
        'valid_run': not conservation_violated,
        'violation_margin': violation_margin if conservation_violated else 0,
        'revolver_drawn': engine.gs.revolver_drawn,
        'win': engine.gs.win,
        'score': engine.score(),
        'irr': irr,
        'moic': moic,
        'irr_status': irr_status.value,
        'duration': engine.gs.month,
        'equity_in': equity_in,
        'equity_out': equity_out,
        'equity_recovery_pct': equity_recovery_pct,
        'final_cash': engine.gs.cash,
        'final_debt': engine.gs.debt,
        'units_sold': engine.gs.units_sold,
        'deposits_collected': engine.gs.deposits_collected,
        'deposits_liability': engine.gs.deposits_liability,
        'progress': engine.gs.progress,
        'failure_reason': engine.gs.failure_reason.value if engine.gs.failure_reason else None,
        'failure_details': engine.gs.failure_details,
        'cash_trough': engine.gs.cash_trough,
        'max_concurrent_events': engine.gs.max_concurrent_events,
        'total_event_months': engine.gs.total_event_months,
        'conservation_violated': conservation_violated,
        'total_costs_incurred': engine.gs.total_costs_incurred,
        'gdv_realized': gdv_realized,
        'stresses': engine.gs.stresses
    }


def run_monte_carlo(n: int, config: Optional[dict] = None) -> dict:
    """Run Monte Carlo simulation
    
    Args:
        n: Number of simulations to run
        config: Optional configuration dictionary (not used yet, for future expansion)
    
    Returns:
        Dictionary with aggregate statistics
    """
    settings = SimulationSettings.baseline()
    results = []
    
    for i in range(n):
        result = run_one_simulation(i, settings=settings)
        results.append(result)
    
    # Calculate statistics
    valid_results = [r for r in results if r['valid_run']]
    wins = [r for r in valid_results if r['win']]
    
    win_rate = len(wins) / len(valid_results) if valid_results else 0.0
    
    # IRR stats (wins only, valid IRR only)
    irrs = [r['irr'] for r in wins if r['irr_status'] == 'valid']
    median_irr = float(np.median(irrs)) if irrs else 0.0
    
    # MOIC stats (wins only)
    moics = [r['moic'] for r in wins]
    median_moic = float(np.median(moics)) if moics else 0.0
    
    return {
        'n': len(valid_results),
        'win_rate': win_rate,
        'median_irr': median_irr,
        'median_moic': median_moic,
        'results': results,  # Full results for detailed analysis
    }
