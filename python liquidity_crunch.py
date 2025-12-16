# ==================== DESKTOP-ONLY VERSION (pygame) ====================
# This file contains the pygame UI and is NOT used in Streamlit deployment.
# For cloud deployment, use: liquiditycrunchapp.py (imports from engine.py)
# ========================================================================

import pygame
import random
import math
import time
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import numpy as np

# Import game logic from engine
from engine import (
    Engine, GameState, EventCard, UnitSale, FailureReason, IRRStatus,
    StressRunMetrics, SimulationSettings,
    TOTAL_EQUITY, LAND_COST, DEV_COSTS, CONSTRUCTION_MONTHS,
    TOTAL_UNITS, UNIT_PRICE, GDV,
    DEPOSIT_1_PCT, DEPOSIT_2_PCT, COMPLETION_PCT,
    BANK_GATE_UNITS_PCT, BANK_GATE_DEPOSITS, MAX_REVOLVER,
    money, clamp, new_game, step_month, is_finished, get_results,
    run_one_simulation,
)

# Import calibration configuration
try:
    from calibration_config import CALIBRATION_TARGETS, STRESS_CONFIG
except ImportError:
    # Fallback if config file not found
    from dataclasses import dataclass
    @dataclass
    class CalibrationTargets:
        target_win_rate: float = 0.55
        target_irr_p50_wins: float = 0.175
        target_moic_p50_wins: float = 1.60
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

matplotlib.use('TkAgg')  # Use TkAgg backend for separate window

# -----------------------------
# Liquidity Crunch: Greenfield Dev (single-file prototype)
# Controls: click buttons
# Goal: finish construction and sell enough units without going bankrupt.
# -----------------------------

WIDTH, HEIGHT = 1400, 800
FPS = 60

# Colors
BG_COLOR = (15, 20, 30)
PANEL_BG = (25, 35, 50)
PANEL_BORDER = (60, 80, 120)
TEXT_PRIMARY = (240, 245, 255)
TEXT_SECONDARY = (180, 190, 210)
BUTTON_NORMAL = (45, 100, 180)
BUTTON_HOVER = (60, 130, 220)
BUTTON_DISABLED = (40, 45, 55)
BUTTON_ACCENT = (80, 160, 100)
BUTTON_ACCENT_HOVER = (100, 190, 120)
POSITIVE_COLOR = (80, 200, 120)
NEGATIVE_COLOR = (220, 80, 80)
WARNING_COLOR = (255, 180, 60)
PROGRESS_BAR = (70, 140, 200)

# Layout constants (strict vertical grid)
LEFT_X = 30
LEFT_WIDTH = 520
TOP_Y = 110
CARD_GAP = 10

CARD_HEIGHTS = {
    "liquidity": 90,
    "equity": 90,
    "debt": 70,
    "sales": 70,
    "deposits": 70,
    "bank": 90,
    "completion": 85,
    "returns": 70,
    "progress_bar": 50
}

# Font sizes (strict hierarchy)

# Stress tracking for Monte Carlo analysis
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

# Failure Reasons Enum
class FailureReason(Enum):
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
    VALID = "valid"
    NO_SIGN_CHANGE = "no_sign_change"
    DID_NOT_CONVERGE = "did_not_converge"

# Font sizes (strict hierarchy)
FONT_TITLE = 16
FONT_BODY = 13
FONT_SMALL = 11
FONT_MICRO = 10

# ---------- Helpers ----------
def clamp(x, a, b):
    return max(a, min(b, x))

def money(x):
    sign = "-" if x < 0 else ""
    x = abs(int(x))
    if x >= 1_000_000_000:
        return f"{sign}€{x/1_000_000_000:.2f}bn"
    if x >= 1_000_000:
        return f"{sign}€{x/1_000_000:.2f}m"
    if x >= 1_000:
        return f"{sign}€{x/1_000:.1f}k"
    return f"{sign}€{x}"

def pct(x):
    return f"{x*100:.1f}%"

# ---------- Project Constants ----------
GDV = 1_000_000_000  # €1bn
UNIT_PRICE = 1_000_000  # €1m per unit
TOTAL_UNITS = 1_000  # 1,000 units
DEV_COSTS = 600_000_000  # €600m construction
LAND_COST = 100_000_000  # €100m land (equity)
TOTAL_EQUITY = 150_000_000  # €150m (€100m land + €50m available)
MAX_DEBT_FACILITY = 600_000_000  # €600m senior debt cap
CONSTRUCTION_MONTHS = 30  # Expected construction duration

# Deposit structure
DEPOSIT_1_PCT = 0.15  # 15% on contract
DEPOSIT_2_PCT = 0.15  # 15% at month +12
COMPLETION_PCT = 0.70  # 70% at completion

# Bank gate requirements
BANK_GATE_UNITS_PCT = 0.15  # 15% of units sold
BANK_GATE_DEPOSITS = 45_000_000  # €45m deposits collected

# ---------- Simulation Settings (for sensitivity analysis) ----------
@dataclass
class SimulationSettings:
    """Configuration for sensitivity analysis"""
    events_enabled: bool = True
    sales_volatility_mult: float = 1.0  # 1.0 = baseline, 0.5 = half volatility
    event_frequency_mult: float = 1.0  # 1.0 = baseline, 0.5 = half frequency
    rate_shock_mult: float = 1.0  # 1.0 = baseline, 0.5 = half rate shock severity
    deposit_gate_mult: float = 1.0  # 1.0 = baseline, 0.8 = relaxed 20%
    cost_overrun_mult: float = 1.0  # 1.0 = baseline, 0.5 = reduced overruns
    
    @staticmethod
    def baseline():
        return SimulationSettings()
    
    @staticmethod
    def no_events():
        return SimulationSettings(events_enabled=False)
    
    @staticmethod
    def reduced_volatility():
        return SimulationSettings(sales_volatility_mult=0.5)
    
    @staticmethod
    def reduced_event_freq():
        return SimulationSettings(event_frequency_mult=0.5)
    
    @staticmethod
    def reduced_rate_shocks():
        return SimulationSettings(rate_shock_mult=0.5)
    
    @staticmethod
    def relaxed_deposit_gate():
        return SimulationSettings(deposit_gate_mult=0.8)
    
    @staticmethod
    def reduced_cost_overruns():
        return SimulationSettings(cost_overrun_mult=0.5)

# ---------- Unit Sale Tracking ----------
@dataclass
class UnitSale:
    sale_month: int
    deposit_1_paid: bool = False
    deposit_2_paid: bool = False
    completion_paid: bool = False

# ---------- Event system ----------
@dataclass
class EventCard:
    name: str
    months_left: int
    # effects are applied each month while active
    burn_mult: float = 1.0
    sales_mult: float = 1.0
    progress_mult: float = 1.0
    rate_add: float = 0.0
    debt_lock: bool = False
    one_off_cost: int = 0
    dev_cost_mult: float = 1.0  # Added for construction cost impacts
    description: str = ""

    def tick(self):
        self.months_left -= 1

@dataclass
class GameState:
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
    max_revolver: int = 20_000_000
    
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

    # Score / tracking
    near_miss_count: int = 0
    last_net_cf: int = 0
    last_month_burn: int = 0  # Track monthly burn for runway calculation
    history_cash: list = field(default_factory=list)
    
    # Equity cashflows for IRR/MOIC calculation
    equity_cashflows: list = field(default_factory=list)  # [{'month': 0, 'amount': -100m}, ...]

    # Stress tracking
    stresses: StressRunMetrics = field(default_factory=StressRunMetrics)

    # End state
    game_over: bool = False
    win: bool = False
    reason: str = ""
    
    # Failure diagnostics (for analytics)
    failure_reason: FailureReason = None
    failure_details: dict = field(default_factory=dict)
    cash_trough: int = field(default=TOTAL_EQUITY)  # Track minimum cash reached
    max_concurrent_events: int = 0
    total_event_months: int = 0  # Sum of all event durations experienced
    
    # Comprehensive cash tracking (Sources & Uses)
    # SOURCES
    equity_in_total: int = 0  # Cumulative equity injected
    debt_drawn_total: int = 0  # Cumulative debt drawn (lifetime)
    debt_outstanding: int = 0  # Current debt balance (same as debt)
    sales_cash_received_total: int = 0  # Total cash from sales (deposits + deed)
    deposits_collected_total: int = 0  # Total deposit cash collected (subset of sales_cash)
    
    # USES
    land_cost_paid: int = 0
    construction_cost_paid: int = 0
    interest_paid_total: int = 0
    debt_principal_repaid_total: int = 0
    event_costs_paid_total: int = 0
    revolver_drawn_total: int = 0  # Cumulative revolver draws
    revolver_repaid_total: int = 0  # Cumulative revolver repayments

# ---------- Game engine ----------
class Engine:
    def __init__(self, settings=None):
        self.gs = GameState()
        self.settings = settings if settings else SimulationSettings.baseline()
        self.apply_land_purchase()

    def apply_land_purchase(self):
        # Land purchase at start (phase LAND) - paid from equity
        self.gs.cash -= LAND_COST
        self.gs.equity_used = LAND_COST
        self.gs.equity_remaining = TOTAL_EQUITY - LAND_COST
        self.gs.phase = "BUILD"
        self.gs.total_costs_incurred += LAND_COST
        
        # Track sources & uses
        # Note: We start with TOTAL_EQUITY in cash, so that's the total equity injected
        self.gs.equity_in_total = TOTAL_EQUITY  # €150m total equity available from start
        self.gs.land_cost_paid = LAND_COST
        
        self.gs.reason = f"Land acquired for {money(LAND_COST)}. Equity used: {money(LAND_COST)}. Construction begins!"
        
        # Track equity cashflow (negative = equity paid in)
        # This is the ONLY equity injection at start
        self.gs.equity_cashflows.append({'month': 0, 'amount': -LAND_COST})

    def current_rate(self):
        # Base + active event add-ons
        rate = self.gs.base_rate
        for ev in self.gs.active_events:
            rate += ev.rate_add
        return max(0.0, rate)

    def has_debt_lock(self):
        return any(ev.debt_lock for ev in self.gs.active_events)
    
    def check_bank_gate(self):
        """Check if bank financing gate conditions are met"""
        gs = self.gs
        units_threshold = int(TOTAL_UNITS * BANK_GATE_UNITS_PCT)  # 150 units
        deposits_threshold = int(BANK_GATE_DEPOSITS * self.settings.deposit_gate_mult)  # Adjusted by settings
        
        # Graduated unlock: debt grows with deposits
        base_availability = STRESS_CONFIG.bank_gate_base_availability
        slope = STRESS_CONFIG.bank_gate_slope
        graduated_debt = min(MAX_DEBT_FACILITY, 
                            int(base_availability + slope * gs.deposits_collected))
        gs.max_debt = max(gs.max_debt, graduated_debt)
        
        units_ok = gs.units_sold >= units_threshold
        deposits_ok = gs.deposits_collected >= deposits_threshold
        
        if units_ok and deposits_ok and not gs.bank_gate_met:
            gs.bank_gate_met = True
            gs.reason = f"🏦 Bank financing UNLOCKED! {gs.units_sold} units sold, {money(gs.deposits_collected)} deposits collected."
        
        return gs.bank_gate_met

    def combined_multipliers(self):
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
        # Simple market drift with difficulty penalty
        # Scale up for 1,000 units (sell ~3-4% per month early game)
        base = 35.0  # ~35 units/month baseline
        # as difficulty increases, sales get harder
        base *= max(0.6, 1.0 - 0.03 * self.gs.difficulty)
        return base

    def monthly_construction_cost(self):
        # Budget-based construction cost with multipliers
        dev_cost_mult, _, _ = self.combined_multipliers()
        return int(self.gs.monthly_dev_cost * dev_cost_mult)

    def roll_event(self):
        gs = self.gs
        if gs.phase != "BUILD":
            return
        
        # Skip events if disabled in settings
        if not self.settings.events_enabled:
            return

        # event chance uses STRESS_CONFIG baseline (adjusted by settings)
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
        # Apply sales volatility from settings (reduce variance if < 1.0)
        vol_range = 0.3 * self.settings.sales_volatility_mult  # baseline ±30%, can be reduced
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
                gs.deposits_liability += deposit_1_amount  # Track as LIABILITY
                
                # Track sales cash (deposits are part of sales revenue)
                gs.sales_cash_received_total += deposit_1_amount
                gs.deposits_collected_total += deposit_1_amount
        
        # Collect deposit #2 for units sold 12 months ago
        deposit_2_amount = int(UNIT_PRICE * DEPOSIT_2_PCT)
        for unit in gs.unit_sales:
            if unit.sale_month == gs.month - 12 and not unit.deposit_2_paid:
                deposit_inflow += deposit_2_amount
                unit.deposit_2_paid = True
                gs.deposits_collected += deposit_2_amount
                gs.deposits_liability += deposit_2_amount  # Track as LIABILITY
                
                # Track sales cash
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
            # For each unit, collect: 70% completion + any unpaid deposit #2
            deposit_2_amount = int(UNIT_PRICE * DEPOSIT_2_PCT)
            total_deed_inflow = 0
            deposits_cleared_count = 0
            
            for unit in gs.unit_sales:
                if unit.completion_paid:
                    continue  # Skip already completed units
                
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
            
            # Assertion: if all units sold, deposits should be fully cleared
            if gs.units_sold == gs.units_total and abs(gs.deposits_liability) > 1000:
                print(f"WARNING: All {gs.units_total} units sold but deposits_liability = €{gs.deposits_liability:,.0f} (should be ~0)")
            
            gs.reason = f"🎉 COMPLETION! Deed inflow: {money(total_deed_inflow)}. Debt repaid: {money(debt_repayment)}. Deposits cleared: {deposits_cleared_count} units. Net: {money(completion_inflow)}."

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
        
        # Use combined interest in cash flow
        interest = total_interest

        # --- APPLY CASH FLOWS ---
        net = deposit_inflow + completion_inflow - dev_cost - interest
        gs.cash += net
        gs.last_net_cf = net
        gs.last_month_burn = net  # Track for runway calculation

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
        
        # Track cash trough
        if gs.cash < gs.cash_trough:
            gs.cash_trough = gs.cash
        
        # Track event diagnostics
        if len(gs.active_events) > gs.max_concurrent_events:
            gs.max_concurrent_events = len(gs.active_events)
        gs.total_event_months += len(gs.active_events)

        # Near-miss tracking
        if gs.cash < 10_000_000 and gs.cash >= 0:
            gs.near_miss_count += 1

        # Lose conditions with failure reason tagging
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
        
        # Timeout condition (60 months without completion)
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

        # Win condition (check at completion)
        sold_ratio = gs.units_sold / gs.units_total
        if gs.month >= CONSTRUCTION_MONTHS and gs.progress >= 100.0 and sold_ratio >= 0.70 and gs.cash >= 0:
            gs.game_over = True
            gs.win = True
            gs.reason = "Project complete! 70%+ units sold, debt repaid, positive cash. Victory!"

    # -------- Actions (buttons) --------
    def action_raise_debt(self):
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

        draw = min(50_000_000, headroom)  # Draw up to €50m at a time
        gs.debt += draw
        gs.cash += draw
        gs.peak_debt = max(gs.peak_debt, gs.debt)
        
        # Track cumulative debt drawn
        gs.debt_drawn_total += draw
        gs.debt_outstanding = gs.debt
        
        gs.reason = f"Drew debt: +{money(draw)}. Total debt: {money(gs.debt)}. Interest increases."

    def action_draw_revolver(self):
        """Draw from short-term working capital revolver"""
        gs = self.gs
        if gs.game_over:
            return
        
        revolver_headroom = gs.max_revolver - gs.revolver_drawn
        if revolver_headroom <= 0:
            gs.reason = "Revolver fully drawn."
            return
        
        draw = min(10_000_000, revolver_headroom)
        gs.revolver_drawn += draw
        gs.cash += draw
        
        # Track cumulative revolver draws
        gs.revolver_drawn_total += draw
        
        gs.reason = f"Drew revolver: +{money(draw)}. Total revolver: {money(gs.revolver_drawn)}."

    def action_slow_construction(self):
        gs = self.gs
        if gs.game_over:
            return
        if gs.progress >= 100.0:
            gs.reason = "Construction already complete."
            return
        gs.slow_build = True
        gs.slow_build_months_left = 4
        gs.reason = "Slowed construction for 4 months (dev costs -30%, progress -30%)."

    def action_sales_push(self):
        gs = self.gs
        if gs.game_over:
            return
        # Sales push costs money (marketing)
        marketing_cost = 5_000_000  # €5m campaign
        if gs.cash < marketing_cost:
            gs.reason = "Not enough cash for marketing campaign."
            return
        gs.cash -= marketing_cost
        gs.sales_push = True
        gs.sales_push_months_left = 3
        gs.reason = f"Sales campaign launched ({money(marketing_cost)}). Sales boosted 60% for 3 months."

    def action_equity_injection(self):
        gs = self.gs
        if gs.game_over:
            return
        if gs.equity_remaining <= 0:
            gs.reason = "No equity capacity left."
            return
        inject = min(10_000_000, gs.equity_remaining)  # Inject up to €10m at a time
        gs.cash += inject
        gs.equity_used += inject
        gs.equity_remaining -= inject
        gs.equity_in_total += inject
        gs.reason = f"Equity injected: +{money(inject)}. Total equity used: {money(gs.equity_used)}/{money(TOTAL_EQUITY)}."
        
        # Track equity cashflow (negative = equity paid in)
        gs.equity_cashflows.append({'month': gs.month, 'amount': -inject})

    def action_resolve_event(self):
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
        
        # Calculate resolve cost using STRESS_CONFIG parameters
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
        # reduce duration heavily
        ev.months_left = max(1, ev.months_left - 3)
        # soften effects
        ev.dev_cost_mult = 1.0 + (ev.dev_cost_mult - 1.0) * 0.35
        ev.sales_mult = 1.0 - (1.0 - ev.sales_mult) * 0.35
        ev.progress_mult = 1.0 - (1.0 - ev.progress_mult) * 0.35
        ev.rate_add *= 0.35
        ev.debt_lock = False
        gs.reason = f"Mitigated {ev.name} for {money(cost)} (impact reduced 65%, duration -3 months)."

    def score(self):
        gs = self.gs
        # Profit-like: cash - debt
        nav = gs.cash - gs.debt
        speed_penalty = gs.month * 1_000_000  # €1m per month penalty
        equity_penalty = (gs.equity_used - LAND_COST) * 2  # 2x penalty on equity beyond land
        near_miss_penalty = gs.near_miss_count * 3_000_000
        return max(0, nav - speed_penalty - equity_penalty - near_miss_penalty)
    
    def calculate_returns(self):
        """Calculate money multiple and IRR"""
        gs = self.gs
        equity_used = gs.equity_used if gs.equity_used > 0 else 1  # Avoid division by zero
        
        # Money Multiple = Net Cash / Total Equity Used
        money_multiple = gs.cash / equity_used
        
        # IRR approximation: (Money Multiple)^(12 / Project Months) - 1
        if gs.month > 0:
            irr = (money_multiple ** (12.0 / gs.month)) - 1.0
        else:
            irr = 0.0
        
        return money_multiple, irr
    
    def calculate_runway(self):
        """Calculate months of cash runway based on monthly burn"""
        gs = self.gs
        if gs.last_month_burn >= 0:
            return float('inf')  # Positive cashflow = infinite runway
        burn_rate = abs(gs.last_month_burn)
        if burn_rate == 0:
            return float('inf')
        return gs.cash / burn_rate
    
    def reconcile_sources_uses(self):
        """Reconcile all sources and uses of cash to verify accounting
        Returns: (sources_dict, uses_dict, delta)
        
        Accounting equation:
        Cash = Equity + Debt_Drawn + Sales_GROSS + Revolver - Costs - Debt_Repaid
        
        Note: Sales_GROSS includes amounts paid directly to lenders (debt repayment at completion).
        So we must subtract debt_repaid separately to get actual cash retained.
        """
        gs = self.gs
        
        # SOURCES (all cash flowing INTO the project)
        sources = {
            "equity_in": gs.equity_in_total,
            "debt_drawn": gs.debt_drawn_total,
            "sales_gross": gs.sales_cash_received_total,  # Includes amounts paid to lenders
            "revolver_drawn": gs.revolver_drawn_total,
        }
        
        # USES (all cash flowing OUT of the project)
        uses = {
            "land": gs.land_cost_paid,
            "construction": gs.construction_cost_paid,
            "interest": gs.interest_paid_total,
            "debt_repaid": gs.debt_principal_repaid_total,  # Paid back to lenders
            "events": gs.event_costs_paid_total,
        }
        
        # Outstanding liabilities (not yet paid, still in cash)
        liabilities = {
            "debt_outstanding": gs.debt,
            "revolver_outstanding": gs.revolver_drawn,
            "deposits_liability": gs.deposits_liability,
        }
        
        total_sources = sum(sources.values())
        total_uses = sum(uses.values())
        total_liabilities = sum(liabilities.values())
        
        # Cash should equal: Total Sources - Total Uses
        # (Liabilities are still in cash, not yet paid out)
        expected_cash = total_sources - total_uses
        delta = gs.cash - expected_cash
        
        # Assertion check for 100% unit sales
        if gs.units_sold == gs.units_total and abs(gs.deposits_liability) > 1000:
            print(f"WARNING: All units sold but deposits_liability = €{gs.deposits_liability:,.0f} (should be ~0)")
        
        # Combine for display
        all_uses = {**uses, **{"liability_" + k: v for k, v in liabilities.items()}}
        
        return sources, all_uses, delta
    
    def compute_equity_distribution_end(self, debug=False):
        """Calculate net cash distributable to equity at end of run via waterfall.
        Returns: Net amount to equity (can be negative if equity is wiped out)
        """
        gs = self.gs
        
        if debug:
            import os
            # Only show debug if explicitly enabled
            if os.environ.get('GAME_DEBUG') == '1':
                print(f"\n=== EQUITY WATERFALL DEBUG (Month {gs.month}) ===")
                print(f"Cash on hand: €{gs.cash:,.0f}")
                print(f"Debt outstanding: €{gs.debt:,.0f}")
                print(f"Deposits liability: €{gs.deposits_liability:,.0f}")
                print(f"Units sold: {gs.units_sold}")
                print(f"Deposits collected: €{gs.deposits_collected:,.0f}")
                print(f"Total costs incurred: €{gs.total_costs_incurred:,.0f}")
                print(f"Equity used: €{gs.equity_used:,.0f}")
                
                # Cash reconciliation
                print(f"\n--- CASH RECONCILIATION ---")
                gdv_realized = gs.units_sold * UNIT_PRICE
                print(f"GDV (units × price): €{gdv_realized:,.0f}")
                print(f"Equity injected: €{gs.equity_used:,.0f}")
                print(f"")
                
                # Detailed deposit tracking
                deposit_1_count = sum(1 for u in gs.unit_sales if u.deposit_1_paid)
                deposit_2_count = sum(1 for u in gs.unit_sales if u.deposit_2_paid)
                completion_count = sum(1 for u in gs.unit_sales if u.completion_paid)
                expected_deposits = (deposit_1_count * int(UNIT_PRICE * DEPOSIT_1_PCT) + 
                                   deposit_2_count * int(UNIT_PRICE * DEPOSIT_2_PCT))
                expected_completion = completion_count * int(UNIT_PRICE * COMPLETION_PCT)
                
                print(f"Deposit #1 paid: {deposit_1_count} units (€{deposit_1_count * int(UNIT_PRICE * DEPOSIT_1_PCT):,.0f})")
                print(f"Deposit #2 paid: {deposit_2_count} units (€{deposit_2_count * int(UNIT_PRICE * DEPOSIT_2_PCT):,.0f})")
                print(f"Completion paid: {completion_count} units (€{expected_completion:,.0f})")
                print(f"Deposits collected (tracked): €{gs.deposits_collected:,.0f}")
                print(f"Deposits expected: €{expected_deposits:,.0f}")
                print(f"")
                
                total_sales_inflows = expected_deposits + expected_completion
                print(f"Total sales inflows (deposits + completion): €{total_sales_inflows:,.0f}")
                print(f"Theoretical inflows (equity + sales): €{gs.equity_used + total_sales_inflows:,.0f}")
                print(f"")
                print(f"Total costs paid: €{gs.total_costs_incurred:,.0f}")
                print(f"Debt remaining: €{gs.debt:,.0f}")
                print(f"Peak debt drawn: €{getattr(gs, 'peak_debt', 0):,.0f}")
                print(f"")
                expected_cash = gs.equity_used + total_sales_inflows - gs.total_costs_incurred
                print(f"Expected cash: €{expected_cash:,.0f}")
                print(f"Actual cash: €{gs.cash:,.0f}")
                print(f"Difference: €{gs.cash - expected_cash:,.0f}")
        
        # Start with cash on hand
        net_to_equity = gs.cash
        
        # Subtract debt (must be repaid before equity gets anything)
        net_to_equity -= gs.debt
        
        # Subtract revolver (short-term working capital debt)
        net_to_equity -= gs.revolver_drawn
        
        # Subtract deposits liability (owed to buyers, not equity)
        net_to_equity -= gs.deposits_liability
        
        if debug:
            print(f"Net to equity: €{net_to_equity:,.0f}")
            print(f"============================\n")
        
        return net_to_equity
    
    def calculate_irr_moic(self):
        """Calculate IRR and Money Multiple (MOIC) from equity cashflows
        Returns: (irr_annual, moic, equity_in, equity_out, irr_status)
        """
        gs = self.gs
        
        # Add final distribution if game over (using proper waterfall)
        cashflows = gs.equity_cashflows.copy()
        if gs.game_over:
            # Use waterfall to compute net distributable to equity
            # Enable debug for seeded run with seed=6
            debug = (hasattr(self, '_seed') and self._seed == 6)
            net_to_equity = self.compute_equity_distribution_end(debug=debug)
            
            # Only add positive distribution if equity gets paid
            # If net_to_equity <= 0, equity is wiped out (no distribution)
            if net_to_equity > 0:
                cashflows.append({'month': gs.month, 'amount': net_to_equity})
        
        # Calculate MOIC
        equity_in = sum(abs(cf['amount']) for cf in cashflows if cf['amount'] < 0)
        equity_out = sum(cf['amount'] for cf in cashflows if cf['amount'] > 0)
        moic = equity_out / equity_in if equity_in > 0 else 0.0
        
        # Calculate IRR using binary search
        irr_monthly, irr_status = self._solve_irr(cashflows)
        irr_annual = (1.0 + irr_monthly) ** 12 - 1.0 if irr_status == IRRStatus.VALID else 0.0
        
        return irr_annual, moic, equity_in, equity_out, irr_status
    
    def diagnose_failure(self):
        """Diagnose primary failure reason if game ended in loss"""
        gs = self.gs
        if not gs.game_over or gs.win or gs.failure_reason:
            return  # Already diagnosed or not applicable
        
        # Analyze state to determine most likely primary failure reason
        runway = self.calculate_runway() if gs.last_month_burn < 0 else float('inf')
        units_threshold = int(TOTAL_UNITS * BANK_GATE_UNITS_PCT)
        
        # Check various failure modes
        if gs.cash < 0:
            gs.failure_reason = FailureReason.LIQUIDITY_ZERO
        elif runway < 1.0 and not gs.bank_gate_met:
            gs.failure_reason = FailureReason.RUNWAY_TOO_LOW
        elif not gs.bank_gate_met and gs.month > 15:
            gs.failure_reason = FailureReason.BANK_GATE_NOT_MET
        elif gs.debt >= gs.max_debt * 0.95 and gs.cash < 20_000_000:
            gs.failure_reason = FailureReason.DEBT_CAP_EXCEEDED
        elif gs.max_concurrent_events >= 3:
            gs.failure_reason = FailureReason.EVENT_STACK
        elif self.current_rate() > 0.12:  # Interest rate > 12%
            gs.failure_reason = FailureReason.RATE_SHOCK
        elif gs.units_sold < 300 and gs.month > 20:  # <30% sold after 20 months
            gs.failure_reason = FailureReason.SALES_STALL
        elif gs.month >= 60:
            gs.failure_reason = FailureReason.TIMEOUT
        else:
            gs.failure_reason = FailureReason.OTHER
        
        # Capture detailed state
        gs.failure_details = {
            'final_cash': gs.cash,
            'final_debt': gs.debt,
            'units_sold': gs.units_sold,
            'deposits_collected': gs.deposits_collected,
            'progress': gs.progress,
            'month': gs.month,
            'runway': runway,
            'bank_gate_met': gs.bank_gate_met,
            'max_concurrent_events': gs.max_concurrent_events,
            'total_event_months': gs.total_event_months,
            'current_rate': self.current_rate(),
            'cash_trough': gs.cash_trough
        }
    
    def _solve_irr(self, cashflows, tolerance=1e-6, max_iterations=200):
        """Solve for monthly IRR using binary search where NPV = 0
        Returns: (irr_monthly, irr_status)
        """
        if not cashflows:
            return 0.0, IRRStatus.NO_SIGN_CHANGE
        
        # Check for sign change (required for IRR to exist)
        amounts = [cf['amount'] for cf in cashflows]
        has_positive = any(a > 0 for a in amounts)
        has_negative = any(a < 0 for a in amounts)
        
        if not (has_positive and has_negative):
            return 0.0, IRRStatus.NO_SIGN_CHANGE
        
        # Binary search between -99% and +500% monthly return
        low, high = -0.99, 5.0
        
        for iteration in range(max_iterations):
            mid = (low + high) / 2.0
            npv = sum(cf['amount'] / ((1.0 + mid) ** cf['month']) for cf in cashflows)
            
            if abs(npv) < tolerance:
                return mid, IRRStatus.VALID
            
            if npv > 0:
                low = mid
            else:
                high = mid
        
        # Did not converge
        return mid, IRRStatus.DID_NOT_CONVERGE
    
    def ai_decide_action(self):
        """AI decides which action to take this month (if any)"""
        gs = self.gs
        runway = self.calculate_runway()
        
        # Priority: Use revolver for short-term liquidity
        if gs.cash < 20_000_000 and gs.revolver_drawn < gs.max_revolver:
            self.action_draw_revolver()
            return "Drew revolver (short-term liquidity)"
        
        # Priority 1: Survive (runway < 2 months)
        if runway < 2.0 and runway != float('inf'):
            # Try to raise debt first
            if gs.bank_gate_met and not self.has_debt_lock() and gs.debt < gs.max_debt:
                self.action_raise_debt()
                return "Raised debt (survival)"
            # Otherwise inject equity
            elif gs.equity_remaining > 0:
                self.action_equity_injection()
                return "Injected equity (survival)"
        
        # Priority 2: Unlock bank gate if close
        if not gs.bank_gate_met:
            units_threshold = int(TOTAL_UNITS * BANK_GATE_UNITS_PCT)
            deposits_needed = BANK_GATE_DEPOSITS - gs.deposits_collected
            units_needed = units_threshold - gs.units_sold
            
            # If close to unlocking and have cash, push sales
            if (units_needed < 30 or deposits_needed < 10_000_000) and gs.cash > 20_000_000:
                if not gs.sales_push:
                    self.action_sales_push()
                    return "Sales push (unlock bank)"
        
        # Priority 3: Manage construction costs if cash tight
        if runway < 4.0 and runway != float('inf') and gs.progress < 90:
            if not gs.slow_build and gs.cash < 50_000_000:
                self.action_slow_construction()
                return "Slow construction (conserve cash)"
        
        # Priority 4: Push sales if lagging
        sold_ratio = gs.units_sold / gs.units_total
        progress_ratio = gs.progress / 100.0
        if progress_ratio > sold_ratio + 0.15 and gs.cash > 15_000_000:
            if not gs.sales_push:
                self.action_sales_push()
                return "Sales push (catch up)"
        
        # Priority 5: Resolve expensive events
        if gs.active_events and gs.cash > 30_000_000:
            worst_event = max(gs.active_events, key=lambda e: e.months_left * (2.0 - e.sales_mult + e.dev_cost_mult))
            if worst_event.months_left > 2:
                self.action_resolve_event()
                return f"Resolved event: {worst_event.name}"
        
        return "No action (holding)"

@dataclass
class SimulationProgress:
    """Track Monte Carlo simulation progress"""
    total: int = 0
    completed: int = 0
    cancelled: bool = False
    start_time: float = 0
    results: list = field(default_factory=list)
    settings: SimulationSettings = None  # Track settings used
    
    def progress_pct(self):
        return self.completed / self.total if self.total > 0 else 0.0
    
    def elapsed_time(self):
        import time
        return time.time() - self.start_time
    
    def eta(self):
        pct = self.progress_pct()
        if pct <= 0:
            return 0
        elapsed = self.elapsed_time()
        return elapsed * (1.0 / pct - 1.0)
    
    def compute_stats(self):
        """Compute current statistics from completed results"""
        if not self.results:
            return None
        
        wins = sum(1 for r in self.results if r['win'])
        win_rate = wins / len(self.results)
        
        irrs = sorted([r['irr'] for r in self.results])
        moics = sorted([r['moic'] for r in self.results])
        
        n = len(self.results)
        return {
            'win_rate': win_rate,
            'mean_irr': sum(irrs) / n,
            'median_irr': irrs[n//2],
            'p10_irr': irrs[max(0, n//10)],
            'p50_irr': irrs[n//2],
            'p90_irr': irrs[min(n-1, n*9//10)],
            'mean_moic': sum(moics) / n,
            'median_moic': moics[n//2],
            'p10_moic': moics[max(0, n//10)],
            'p50_moic': moics[n//2],
            'p90_moic': moics[min(n-1, n*9//10)],
        }

def run_one_simulation(seed, settings=None):
    """Run a single simulation headlessly with optional settings"""
    random.seed(seed)
    eng = Engine(settings=settings if settings else SimulationSettings.baseline())
    eng._seed = seed  # Store seed for debug logging
    
    # Run AI to completion (max 60 months)
    for _ in range(60):
        if eng.gs.game_over:
            break
        eng.ai_decide_action()
        eng.tick_month()
    
    # Diagnose failure reason if lost
    if eng.gs.game_over and not eng.gs.win:
        eng.diagnose_failure()
    
    # Calculate results
    irr, moic, equity_in, equity_out, irr_status = eng.calculate_irr_moic()
    
    # Calculate equity recovery
    equity_recovery_pct = (equity_out / equity_in * 100) if equity_in > 0 else 0.0
    
    # === CONSERVATION OF VALUE SANITY CHECK ===
    # Core equation: equity to distribute = total inflows - total costs - liabilities
    # 
    # Cash = Equity_In + Debt_Drawn + Revolver_Drawn + Sales - Costs_Paid - Debt_Repaid
    # Equity_Out (from waterfall) = Cash - Debt_Outstanding - Revolver_Outstanding - Deposits_Liability
    # 
    # Therefore Max_Equity = Equity_In + Debt_Drawn + Revolver_Drawn + Sales - Costs - Debt_Repaid - Debt_Outstanding - Revolver - Deposits
    # 
    # Note: Revolver appears twice (once as inflow, once as liability to subtract) so they cancel:
    # Max_Equity = Equity_In + Debt_Drawn + Sales - Costs - Debt_Repaid - Debt_Outstanding - Deposits
    # 
    # Debt terms simplify: (Debt_Drawn - Debt_Repaid - Debt_Outstanding) should equal 0 if fully repaid
    gdv_realized = eng.gs.units_sold * UNIT_PRICE
    
    # Max possible equity distribution (what equity_out should not exceed)
    max_equity_upper_bound = max(0, 
        eng.gs.equity_in_total  # Total equity injected (including initial + any additional)
        + eng.gs.debt_drawn_total  # Total debt drawn (cash inflow)
        + gdv_realized  # Total unit sale proceeds
        - eng.gs.total_costs_incurred  # All costs paid (land, construction, interest, events)
        - eng.gs.debt_principal_repaid_total  # Debt principal paid back (cash outflow)
        - eng.gs.debt  # Debt still owed
        # NOTE: Revolver NOT subtracted here - it's already handled in equity_out waterfall
        - eng.gs.deposits_liability  # Deposits owed to buyers
    )
    
    # Check if equity payouts exceed conservation of value
    # Use tighter tolerance: allow only €500k for rounding/post-completion sales
    conservation_violated = False
    violation_margin = equity_out - max_equity_upper_bound
    
    if violation_margin > 500_000:  # Stricter: €500k tolerance
        conservation_violated = True
        print(f"\n*** CONSERVATION VIOLATION (seed={seed}):")
        print(f"   Equity out: €{equity_out:,.0f}")
        print(f"   Max possible: €{max_equity_upper_bound:,.0f}")
        print(f"   Violation margin: €{violation_margin:,.0f}")
        print(f"   --- Waterfall Components ---")
        print(f"   Equity injected: €{eng.gs.equity_used:,.0f}")
        print(f"   GDV realized: €{gdv_realized:,.0f}")
        print(f"   Total costs: €{eng.gs.total_costs_incurred:,.0f}")
        print(f"   Debt remaining: €{eng.gs.debt:,.0f}")
        print(f"   Revolver drawn: €{eng.gs.revolver_drawn:,.0f}")
        print(f"   Deposits liability: €{eng.gs.deposits_liability:,.0f}")
        print(f"   Units sold: {eng.gs.units_sold}, Completion triggered: {eng.gs.completion_triggered}\n")
    
    return {
        'valid_run': not conservation_violated,  # Mark invalid runs as FIRST field
        'violation_margin': violation_margin if conservation_violated else 0,
        'revolver_drawn': eng.gs.revolver_drawn,
        'win': eng.gs.win,
        'score': eng.score(),
        'irr': irr,
        'moic': moic,
        'irr_status': irr_status.value,
        'duration': eng.gs.month,
        'equity_in': equity_in,
        'equity_out': equity_out,
        'equity_recovery_pct': equity_recovery_pct,
        'final_cash': eng.gs.cash,
        'final_debt': eng.gs.debt,
        'units_sold': eng.gs.units_sold,
        'deposits_collected': eng.gs.deposits_collected,
        'deposits_liability': eng.gs.deposits_liability,
        'progress': eng.gs.progress,
        'failure_reason': eng.gs.failure_reason.value if eng.gs.failure_reason else None,
        'failure_details': eng.gs.failure_details,
        'cash_trough': eng.gs.cash_trough,
        'max_concurrent_events': eng.gs.max_concurrent_events,
        'total_event_months': eng.gs.total_event_months,
        'conservation_violated': conservation_violated,
        'total_costs_incurred': eng.gs.total_costs_incurred,
        'gdv_realized': gdv_realized,
        'stresses': eng.gs.stresses
    }

def run_monte_carlo_chunk(progress, batch_size, seed_base=42, settings=None):
    """Run a chunk of simulations and update progress"""
    start_idx = progress.completed
    end_idx = min(start_idx + batch_size, progress.total)
    
    for i in range(start_idx, end_idx):
        if progress.cancelled:
            break
        
        result = run_one_simulation(seed_base + i, settings=settings)
        progress.results.append(result)
        progress.completed += 1
    
    # Return True if more work remains
    return progress.completed < progress.total and not progress.cancelled

# ---------- ANALYTICS & STATISTICS ----------

def compute_detailed_stats(results):
    """Compute comprehensive statistics from simulation results"""
    if not results:
        return None
    
    # Filter out invalid runs (conservation violations)
    total_runs = len(results)
    valid_results = [r for r in results if r.get('valid_run', True)]
    invalid_count = total_runs - len(valid_results)
    
    # Use valid_results for all subsequent stats
    n = len(valid_results)
    if n == 0:
        return None
    
    wins = [r for r in valid_results if r['win']]
    losses = [r for r in valid_results if not r['win']]
    
    # Filter valid IRRs only
    valid_irrs = [r['irr'] for r in valid_results if r['irr_status'] == 'valid']
    valid_moics = [r['moic'] for r in valid_results if r['irr_status'] == 'valid']
    all_moics = [r['moic'] for r in valid_results]
    
    # Win/loss IRRs
    win_irrs = [r['irr'] for r in wins if r['irr_status'] == 'valid']
    loss_irrs = [r['irr'] for r in losses if r['irr_status'] == 'valid']
    
    # Equity recovery
    equity_recoveries = [r['equity_recovery_pct'] for r in valid_results]
    
    # IRR status breakdown
    irr_valid_count = sum(1 for r in valid_results if r['irr_status'] == 'valid')
    irr_no_sign_change = sum(1 for r in valid_results if r['irr_status'] == 'no_sign_change')
    irr_no_converge = sum(1 for r in valid_results if r['irr_status'] == 'did_not_converge')
    irr_undefined_count = irr_no_sign_change + irr_no_converge
    irr_undefined_pct = (irr_undefined_count / n * 100) if n > 0 else 0
    
    # Conservation violations (already filtered out, but keep for backward compatibility)
    conservation_violations = invalid_count
    
    # Failure reasons
    failure_reasons = {}
    for r in losses:
        reason = r.get('failure_reason', 'OTHER')
        if reason:
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    
    # Stats helper
    def calc_stats(values):
        if not values:
            return None
        arr = np.array(values)
        return {
            'count': len(values),
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            'skew': float(stats.skew(arr)) if len(arr) > 2 else 0.0,
            'kurtosis': float(stats.kurtosis(arr)) if len(arr) > 3 else 0.0,
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'p5': float(np.percentile(arr, 5)),
            'p10': float(np.percentile(arr, 10)),
            'p25': float(np.percentile(arr, 25)),
            'p50': float(np.percentile(arr, 50)),
            'p75': float(np.percentile(arr, 75)),
            'p90': float(np.percentile(arr, 90)),
            'p95': float(np.percentile(arr, 95))
        }
    
    # Calculate thresholds (% of VALID IRRs only)
    irr_above_0_pct = (sum(1 for v in valid_irrs if v > 0.0) / len(valid_irrs) * 100) if valid_irrs else 0
    irr_above_10_pct = (sum(1 for v in valid_irrs if v > 0.10) / len(valid_irrs) * 100) if valid_irrs else 0
    irr_above_15_pct = (sum(1 for v in valid_irrs if v > 0.15) / len(valid_irrs) * 100) if valid_irrs else 0
    
    # Also calculate as % of total runs for context
    irr_above_0_pct_total = (sum(1 for v in valid_irrs if v > 0.0) / n * 100) if n > 0 else 0
    irr_above_10_pct_total = (sum(1 for v in valid_irrs if v > 0.10) / n * 100) if n > 0 else 0
    irr_above_15_pct_total = (sum(1 for v in valid_irrs if v > 0.15) / n * 100) if n > 0 else 0
    
    return {
        'total_runs': total_runs,
        'valid_runs': n,
        'invalid_runs': invalid_count,
        'n': n,
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / n * 100,
        'irr_stats': calc_stats(valid_irrs),
        'moic_stats': calc_stats(all_moics),
        'equity_recovery_stats': calc_stats(equity_recoveries),
        'win_irr_stats': calc_stats(win_irrs),
        'loss_irr_stats': calc_stats(loss_irrs),
        'irr_valid_count': irr_valid_count,
        'irr_no_sign_change': irr_no_sign_change,
        'irr_no_converge': irr_no_converge,
        'irr_undefined_pct': irr_undefined_pct,
        'irr_above_0_pct': irr_above_0_pct,
        'irr_above_10_pct': irr_above_10_pct,
        'irr_above_15_pct': irr_above_15_pct,
        'irr_above_0_pct_total': irr_above_0_pct_total,
        'irr_above_10_pct_total': irr_above_10_pct_total,
        'irr_above_15_pct_total': irr_above_15_pct_total,
        'failure_reasons': failure_reasons,
        'conservation_violations': conservation_violations,
        'raw_results': results
    }

def compute_failure_analysis(results):
    """Analyze failure reasons with detailed metrics"""
    losses = [r for r in results if not r['win']]
    if not losses:
        return None
    
    failure_analysis = {}
    for loss in losses:
        reason = loss.get('failure_reason', 'OTHER')
        if reason not in failure_analysis:
            failure_analysis[reason] = {
                'count': 0,
                'months': [],
                'cash': [],
                'debt': [],
                'units_sold': [],
                'deposits': [],
                'progress': []
            }
        
        fa = failure_analysis[reason]
        fa['count'] += 1
        fa['months'].append(loss['duration'])
        fa['cash'].append(loss['final_cash'])
        fa['debt'].append(loss['final_debt'])
        fa['units_sold'].append(loss['units_sold'])
        fa['deposits'].append(loss['deposits_collected'])
        fa['progress'].append(loss['progress'])
    
    # Calculate averages
    for reason, data in failure_analysis.items():
        data['avg_month'] = np.mean(data['months']) if data['months'] else 0
        data['avg_cash'] = np.mean(data['cash']) if data['cash'] else 0
        data['avg_debt'] = np.mean(data['debt']) if data['debt'] else 0
        data['avg_units_sold'] = np.mean(data['units_sold']) if data['units_sold'] else 0
        data['avg_deposits'] = np.mean(data['deposits']) if data['deposits'] else 0
        data['avg_progress'] = np.mean(data['progress']) if data['progress'] else 0
        data['pct_of_losses'] = data['count'] / len(losses) * 100
    
    return failure_analysis

def extract_sample_runs(results):
    """Extract sample runs for audit trail (3 best wins, 3 median wins, 4 diverse losses)"""
    wins = [r for r in results if r['win']]
    losses = [r for r in results if not r['win']]
    
    samples = {
        'best_wins': [],
        'median_wins': [],
        'diverse_losses': []
    }
    
    # Best 3 wins by IRR
    if wins:
        wins_sorted = sorted([w for w in wins if w['irr_status'] == 'valid'], 
                           key=lambda x: x['irr'], reverse=True)
        samples['best_wins'] = wins_sorted[:3]
    
    # Median 3 wins
    if len(wins) >= 3:
        median_idx = len(wins) // 2
        samples['median_wins'] = wins[median_idx-1:median_idx+2] if median_idx > 0 else wins[:3]
    
    # 4 diverse losses (one from each major failure reason if possible)
    if losses:
        failure_buckets = {}
        for loss in losses:
            reason = loss.get('failure_reason', 'OTHER')
            if reason not in failure_buckets:
                failure_buckets[reason] = []
            failure_buckets[reason].append(loss)
        
        # Pick one from each bucket (up to 4)
        for reason, bucket in sorted(failure_buckets.items(), 
                                     key=lambda x: len(x[1]), reverse=True)[:4]:
            samples['diverse_losses'].append(bucket[0])
    
    return samples

def run_sensitivity_comparison(n=1000, seed_base=5000):
    """Run sensitivity analysis comparing different scenarios"""
    print(f"\n{'='*80}")
    print(f"SENSITIVITY ANALYSIS - Running {n} simulations per scenario...")
    print(f"{'='*80}\n")
    
    scenarios = [
        ("Baseline", SimulationSettings.baseline()),
        ("No Events", SimulationSettings.no_events()),
        ("Reduced Sales Volatility (-50%)", SimulationSettings.reduced_volatility()),
        ("Reduced Event Frequency (-50%)", SimulationSettings.reduced_event_freq()),
        ("Reduced Rate Shocks (-50%)", SimulationSettings.reduced_rate_shocks()),
        ("Relaxed Deposit Gate (-20%)", SimulationSettings.relaxed_deposit_gate()),
        ("Reduced Cost Overruns (-50%)", SimulationSettings.reduced_cost_overruns()),
    ]
    
    comparison_results = []
    
    for name, settings in scenarios:
        print(f"Running: {name}...")
        results = []
        for i in range(n):
            result = run_one_simulation(seed_base + i, settings=settings)
            results.append(result)
        
        stats = compute_detailed_stats(results)
        comparison_results.append({
            'name': name,
            'settings': settings,
            'stats': stats,
            'results': results
        })
        
        print(f"  ✓ Win Rate: {stats['win_rate']:.1f}% | Median IRR: {stats['irr_stats']['median']*100:.1f}%")
    
    print(f"\n{'='*80}")
    print("SENSITIVITY ANALYSIS SUMMARY")
    print(f"{'='*80}\n")
    print(f"{'Scenario':<40} {'Win%':>8} {'Med IRR':>10} {'Δ Win%':>10} {'Δ IRR':>10}")
    print("-" * 80)
    
    baseline_stats = comparison_results[0]['stats']
    for result in comparison_results:
        stats = result['stats']
        delta_win = stats['win_rate'] - baseline_stats['win_rate']
        delta_irr = (stats['irr_stats']['median'] - baseline_stats['irr_stats']['median']) * 100 if stats['irr_stats'] and baseline_stats['irr_stats'] else 0
        
        print(f"{result['name']:<40} {stats['win_rate']:>7.1f}% {stats['irr_stats']['median']*100:>9.1f}% {delta_win:>9.1f}% {delta_irr:>9.1f}%")
    
    return comparison_results

def compute_stress_analysis(results):
    """Compute stress incidence and impact metrics across all runs"""
    valid_results = [r for r in results if r.get('valid_run', True)]
    if not valid_results:
        return None
    
    n = len(valid_results)
    
    # Extract stress data
    had_delay = [r for r in valid_results if r['stresses'].had_delay]
    had_capex = [r for r in valid_results if r['stresses'].had_capex_overrun]
    had_rate = [r for r in valid_results if r['stresses'].had_rate_shock]
    had_sales = [r for r in valid_results if r['stresses'].had_sales_shock]
    had_event = [r for r in valid_results if r['stresses'].had_event_cost]
    had_gate_block = [r for r in valid_results if r['stresses'].bank_gate_blocked_months > 0]
    
    # Compute incidence (frequency)
    incidence = {
        'delay': len(had_delay) / n * 100,
        'capex': len(had_capex) / n * 100,
        'rate': len(had_rate) / n * 100,
        'sales': len(had_sales) / n * 100,
        'event': len(had_event) / n * 100,
        'gate_block': len(had_gate_block) / n * 100,
    }
    
    # Compute average magnitudes (for runs where stress occurred)
    magnitudes = {}
    
    if had_delay:
        magnitudes['delay_months'] = np.mean([r['stresses'].delay_months for r in had_delay])
    else:
        magnitudes['delay_months'] = 0
    
    if had_capex:
        magnitudes['capex_overrun_pct'] = np.mean([r['stresses'].capex_overrun_pct for r in had_capex])
        magnitudes['capex_overrun_amount'] = np.mean([r['stresses'].capex_overrun_amount for r in had_capex])
    else:
        magnitudes['capex_overrun_pct'] = 0
        magnitudes['capex_overrun_amount'] = 0
    
    if had_rate:
        magnitudes['max_rate_bps'] = np.mean([r['stresses'].max_rate_bps for r in had_rate])
        magnitudes['rate_shock_months'] = np.mean([r['stresses'].rate_shock_months for r in had_rate])
    else:
        magnitudes['max_rate_bps'] = 0
        magnitudes['rate_shock_months'] = 0
    
    if had_sales:
        magnitudes['sales_shock_months'] = np.mean([r['stresses'].sales_shock_months for r in had_sales])
        magnitudes['min_sales_mult'] = np.mean([r['stresses'].min_sales_multiplier for r in had_sales])
    else:
        magnitudes['sales_shock_months'] = 0
        magnitudes['min_sales_mult'] = 1.0
    
    if had_event:
        magnitudes['total_event_cost'] = np.mean([r['stresses'].total_event_cost for r in had_event])
        magnitudes['event_count'] = np.mean([r['stresses'].event_count for r in had_event])
    else:
        magnitudes['total_event_cost'] = 0
        magnitudes['event_count'] = 0
    
    if had_gate_block:
        magnitudes['gate_blocked_months'] = np.mean([r['stresses'].bank_gate_blocked_months for r in had_gate_block])
    else:
        magnitudes['gate_blocked_months'] = 0
    
    # Compute impact (Δ metrics with vs without each stress)
    impact = {}
    
    # For each stress type, compute win rate / IRR / MOIC with vs without
    for stress_name, stress_filter in [
        ('delay', lambda r: r['stresses'].had_delay),
        ('capex', lambda r: r['stresses'].had_capex_overrun),
        ('rate', lambda r: r['stresses'].had_rate_shock),
        ('sales', lambda r: r['stresses'].had_sales_shock),
        ('event', lambda r: r['stresses'].had_event_cost),
        ('gate_block', lambda r: r['stresses'].bank_gate_blocked_months > 0),
    ]:
        with_stress = [r for r in valid_results if stress_filter(r)]
        without_stress = [r for r in valid_results if not stress_filter(r)]
        
        if with_stress and without_stress:
            # Win rate
            win_rate_with = len([r for r in with_stress if r['win']]) / len(with_stress) * 100
            win_rate_without = len([r for r in without_stress if r['win']]) / len(without_stress) * 100
            delta_win = win_rate_with - win_rate_without
            
            # IRR (wins only, valid only)
            irr_with = [r['irr'] for r in with_stress if r['win'] and r['irr_status'] == 'valid']
            irr_without = [r['irr'] for r in without_stress if r['win'] and r['irr_status'] == 'valid']
            
            if irr_with and irr_without:
                median_irr_with = np.median(irr_with) * 100
                median_irr_without = np.median(irr_without) * 100
                delta_irr = median_irr_with - median_irr_without
            else:
                median_irr_with = median_irr_without = delta_irr = 0
            
            # MOIC (wins only)
            moic_with = [r['moic'] for r in with_stress if r['win']]
            moic_without = [r['moic'] for r in without_stress if r['win']]
            
            if moic_with and moic_without:
                median_moic_with = np.median(moic_with)
                median_moic_without = np.median(moic_without)
                delta_moic = median_moic_with - median_moic_without
            else:
                median_moic_with = median_moic_without = delta_moic = 0
            
            impact[stress_name] = {
                'win_rate_with': win_rate_with,
                'win_rate_without': win_rate_without,
                'delta_win': delta_win,
                'irr_with': median_irr_with,
                'irr_without': median_irr_without,
                'delta_irr': delta_irr,
                'moic_with': median_moic_with,
                'moic_without': median_moic_without,
                'delta_moic': delta_moic,
            }
        else:
            # Not enough data for comparison
            impact[stress_name] = None
    
    return {
        'incidence': incidence,
        'magnitudes': magnitudes,
        'impact': impact,
        'n': n,
    }

def show_simulation_results(results, bins=50, save_path=None):
    """Display comprehensive simulation results in matplotlib window
    
    Args:
        results: List of simulation results
        bins: Number of histogram bins
        save_path: If provided, save to this path instead of showing
    """
    stats = compute_detailed_stats(results)
    if not stats:
        print("No results to display")
        return
    
    # Print diagnostic information
    print(f"\n{'='*80}")
    print("PLOTTING DIAGNOSTICS")
    print(f"{'='*80}")
    print(f"Total runs: {stats['n']}")
    print(f"Wins: {stats['wins']}")
    print(f"Losses: {stats['losses']}")
    
    # Check IRR validity
    valid_irrs = [r['irr'] for r in results if r['irr_status'] == 'valid' and r['irr'] is not None and np.isfinite(r['irr'])]
    print(f"Valid IRRs: {len(valid_irrs)} ({len(valid_irrs)/stats['n']*100:.1f}%)")
    print(f"No sign change: {stats['irr_no_sign_change']}")
    print(f"Did not converge: {stats['irr_no_converge']}")
    print(f"IRR undefined %: {stats['irr_undefined_pct']:.1f}%")
    
    # Check MOIC validity
    valid_moics = [r['moic'] for r in results if r['moic'] is not None and np.isfinite(r['moic'])]
    print(f"Valid MOICs: {len(valid_moics)}")
    print(f"Invalid MOICs: {stats['n'] - len(valid_moics)}")
    print(f"{'='*80}\n")
    
    failure_analysis = compute_failure_analysis(results)
    stress_analysis = compute_stress_analysis(results)
    
    # Create figure with multiple subplots (4 rows x 3 cols for stress charts)
    fig = plt.figure(figsize=(18, 16), facecolor='white')
    fig.suptitle(f'Monte Carlo Simulation Results (n={stats["n"]})', fontsize=18, fontweight='bold', color='black')
    
    # Set white background explicitly for entire figure
    fig.patch.set_facecolor('white')
    
    # Use default matplotlib style for maximum visibility
    plt.style.use('default')
    
    # 1. IRR Distribution (top left)
    ax1 = plt.subplot(4, 3, 1, facecolor='white')
    ax1.set_facecolor('white')
    
    # Filter valid IRRs (convert to % and filter out None/NaN/inf)
    valid_irrs = [r['irr'] * 100 for r in results 
                  if r['irr_status'] == 'valid' and r['irr'] is not None and np.isfinite(r['irr'])]
    win_irrs = [r['irr'] * 100 for r in results 
                if r['win'] and r['irr_status'] == 'valid' and r['irr'] is not None and np.isfinite(r['irr'])]
    loss_irrs = [r['irr'] * 100 for r in results 
                 if not r['win'] and r['irr_status'] == 'valid' and r['irr'] is not None and np.isfinite(r['irr'])]
    
    if len(valid_irrs) > 0:
        # Plot histograms with solid, high-contrast colors
        ax1.hist(valid_irrs, bins=bins, alpha=1.0, color='#1f77b4', label='All', density=True, edgecolor='black', linewidth=1.5)
        if len(win_irrs) > 0:
            ax1.hist(win_irrs, bins=bins, alpha=0.7, color='#2ca02c', label='Wins', density=True, edgecolor='darkgreen', linewidth=1.5)
        if len(loss_irrs) > 0:
            ax1.hist(loss_irrs, bins=bins, alpha=0.7, color='#d62728', label='Losses', density=True, edgecolor='darkred', linewidth=1.5)
        
        # KDE overlay
        from scipy.stats import gaussian_kde
        if len(valid_irrs) > 5:
            try:
                kde = gaussian_kde(valid_irrs)
                x_range = np.linspace(min(valid_irrs), max(valid_irrs), 200)
                ax1.plot(x_range, kde(x_range), 'k-', linewidth=3, label='KDE')
            except:
                pass  # Skip KDE if it fails
    else:
        # No valid data - show message in black on yellow
        ax1.text(0.5, 0.5, f'No valid IRR values to plot\n{stats["irr_no_sign_change"]} runs: no sign change\n{stats["irr_no_converge"]} runs: did not converge',
                ha='center', va='center', transform=ax1.transAxes, fontsize=14, color='black',
                bbox=dict(boxstyle='round', facecolor='yellow', edgecolor='black', linewidth=2))
    
    ax1.set_title('IRR Distribution (Annualized %)', fontweight='bold', fontsize=14, color='black')
    ax1.set_xlabel('IRR (%)', fontsize=12, color='black')
    ax1.set_ylabel('Density', fontsize=12, color='black')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.5, color='gray', linestyle='--', linewidth=0.5)
    ax1.tick_params(labelsize=10, colors='black')
    
    # Add undefined IRR callout
    if stats['irr_undefined_pct'] > 0 and len(valid_irrs) > 0:
        ax1.text(0.02, 0.98, f"IRR undefined in {stats['irr_undefined_pct']:.1f}% of runs",
                transform=ax1.transAxes, verticalalignment='top', fontsize=10, color='black',
                bbox=dict(boxstyle='round', facecolor='yellow', edgecolor='black', linewidth=1.5))
    
    # 2. MOIC Distribution (top middle)
    ax2 = plt.subplot(4, 3, 2, facecolor='white')
    ax2.set_facecolor('white')
    
    # Filter valid MOICs
    all_moics = [r['moic'] for r in results if r['moic'] is not None and np.isfinite(r['moic'])]
    win_moics = [r['moic'] for r in results if r['win'] and r['moic'] is not None and np.isfinite(r['moic'])]
    loss_moics = [r['moic'] for r in results if not r['win'] and r['moic'] is not None and np.isfinite(r['moic'])]
    
    if len(all_moics) > 0:
        ax2.hist(all_moics, bins=bins, alpha=1.0, color='#ff7f0e', label='All', density=True, edgecolor='black', linewidth=1.5)
        if len(win_moics) > 0:
            ax2.hist(win_moics, bins=bins, alpha=0.7, color='#2ca02c', label='Wins', density=True, edgecolor='darkgreen', linewidth=1.5)
        if len(loss_moics) > 0:
            ax2.hist(loss_moics, bins=bins, alpha=0.7, color='#d62728', label='Losses', density=True, edgecolor='darkred', linewidth=1.5)
        
        # KDE overlay
        if len(all_moics) > 5:
            try:
                kde = gaussian_kde(all_moics)
                x_range = np.linspace(min(all_moics), max(all_moics), 200)
                ax2.plot(x_range, kde(x_range), 'k-', linewidth=3, label='KDE')
            except:
                pass
    else:
        ax2.text(0.5, 0.5, 'No valid MOIC values to plot',
                ha='center', va='center', transform=ax2.transAxes, fontsize=14, color='black',
                bbox=dict(boxstyle='round', facecolor='yellow', edgecolor='black', linewidth=2))
    
    ax2.set_title('MOIC Distribution', fontweight='bold', fontsize=14, color='black')
    ax2.set_xlabel('MOIC (x)', fontsize=12, color='black')
    ax2.set_ylabel('Density', fontsize=12, color='black')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.5, color='gray', linestyle='--', linewidth=0.5)
    ax2.tick_params(labelsize=10, colors='black')
    
    # 3. Equity Recovery % (top right)
    ax3 = plt.subplot(3, 3, 3, facecolor='white')
    ax3.set_facecolor('white')
    
    equity_recoveries = [r['equity_recovery_pct'] for r in results 
                        if r['equity_recovery_pct'] is not None and np.isfinite(r['equity_recovery_pct'])]
    if len(equity_recoveries) > 0:
        ax3.hist(equity_recoveries, bins=bins, alpha=1.0, color='#9467bd', density=True, edgecolor='black', linewidth=1.5)
        if len(equity_recoveries) > 5:
            try:
                kde = gaussian_kde(equity_recoveries)
                x_range = np.linspace(min(equity_recoveries), max(equity_recoveries), 200)
                ax3.plot(x_range, kde(x_range), 'k-', linewidth=3)
            except:
                pass
    else:
        ax3.text(0.5, 0.5, 'No valid equity recovery values',
                ha='center', va='center', transform=ax3.transAxes, fontsize=14, color='black',
                bbox=dict(boxstyle='round', facecolor='yellow', edgecolor='black', linewidth=2))
    
    ax3.set_title('Equity Recovery %', fontweight='bold', fontsize=14, color='black')
    ax3.set_xlabel('Recovery (%)', fontsize=12, color='black')
    ax3.set_ylabel('Density', fontsize=12, color='black')
    ax3.grid(True, alpha=0.5, color='gray', linestyle='--', linewidth=0.5)
    ax3.tick_params(labelsize=10, colors='black')
    
    # 4. Failure Reasons Pareto (middle left)
    ax4 = plt.subplot(3, 3, 4, facecolor='white')
    ax4.set_facecolor('white')
    
    if failure_analysis and stats['losses'] > 0:
        reasons = list(failure_analysis.keys())
        counts = [failure_analysis[r]['count'] for r in reasons]
        pcts = [failure_analysis[r]['pct_of_losses'] for r in reasons]
        
        # Sort by count descending
        sorted_data = sorted(zip(reasons, counts, pcts), key=lambda x: x[1], reverse=True)
        top_reasons = sorted_data[:5]
        
        if len(sorted_data) > 5:
            other_count = sum(x[1] for x in sorted_data[5:])
            other_pct = sum(x[2] for x in sorted_data[5:])
            top_reasons.append(('OTHER', other_count, other_pct))
        
        labels = [r[0] for r in top_reasons]
        values = [r[2] for r in top_reasons]
        
        bars = ax4.barh(labels, values, color='#ff6361', edgecolor='black', linewidth=1.5)
        ax4.set_title('Failure Reasons (% of Losses)', fontweight='bold', fontsize=14, color='black')
        ax4.set_xlabel('% of Losses', fontsize=12, color='black')
        
        # Add count labels in black
        for i, (bar, data) in enumerate(zip(bars, top_reasons)):
            ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{data[1]}', va='center', fontsize=11, color='black', fontweight='bold')
    
    ax4.grid(True, alpha=0.5, axis='x', color='gray', linestyle='--', linewidth=0.5)
    ax4.tick_params(labelsize=10, colors='black')
    
    # 5. IRR Statistics Table (middle middle)
    ax5 = plt.subplot(4, 3, 5)
    ax5.axis('off')
    
    if stats['irr_stats']:
        irr_s = stats['irr_stats']
        table_data = [
            ['Metric', 'Value'],
            ['Count', f"{irr_s['count']}"],
            ['Mean', f"{irr_s['mean']*100:.1f}%"],
            ['Median', f"{irr_s['median']*100:.1f}%"],
            ['Std Dev', f"{irr_s['std']*100:.1f}%"],
            ['Skewness', f"{irr_s['skew']:.2f}"],
            ['Kurtosis', f"{irr_s['kurtosis']:.2f}"],
            ['Min', f"{irr_s['min']*100:.1f}%"],
            ['P5', f"{irr_s['p5']*100:.1f}%"],
            ['P10', f"{irr_s['p10']*100:.1f}%"],
            ['P25', f"{irr_s['p25']*100:.1f}%"],
            ['P50', f"{irr_s['p50']*100:.1f}%"],
            ['P75', f"{irr_s['p75']*100:.1f}%"],
            ['P90', f"{irr_s['p90']*100:.1f}%"],
            ['P95', f"{irr_s['p95']*100:.1f}%"],
            ['Max', f"{irr_s['max']*100:.1f}%"],
        ]
        
        table = ax5.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Header styling
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Make all text black
        for key, cell in table.get_celld().items():
            if key[0] > 0:  # Not header
                cell.set_text_props(color='black')
    
    ax5.set_title('IRR Statistics', fontweight='bold', fontsize=14, color='black', pad=20)
    
    # 6. MOIC Statistics Table (middle right)
    ax6 = plt.subplot(4, 3, 6)
    ax6.axis('off')
    
    if stats['moic_stats']:
        moic_s = stats['moic_stats']
        table_data = [
            ['Metric', 'Value'],
            ['Count', f"{moic_s['count']}"],
            ['Mean', f"{moic_s['mean']:.2f}x"],
            ['Median', f"{moic_s['median']:.2f}x"],
            ['Std Dev', f"{moic_s['std']:.2f}x"],
            ['Min', f"{moic_s['min']:.2f}x"],
            ['P10', f"{moic_s['p10']:.2f}x"],
            ['P25', f"{moic_s['p25']:.2f}x"],
            ['P50', f"{moic_s['p50']:.2f}x"],
            ['P75', f"{moic_s['p75']:.2f}x"],
            ['P90', f"{moic_s['p90']:.2f}x"],
            ['Max', f"{moic_s['max']:.2f}x"],
        ]
        
        table = ax6.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Header styling
        for i in range(2):
            table[(0, i)].set_facecolor('#2196F3')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Make all text black
        for key, cell in table.get_celld().items():
            if key[0] > 0:  # Not header
                cell.set_text_props(color='black')
    
    ax6.set_title('MOIC Statistics', fontweight='bold', fontsize=14, color='black', pad=20)
    
    # 7. Summary Stats (row 3 left)
    ax7 = plt.subplot(4, 3, 7)
    ax7.axis('off')
    
    # Show calibration targets
    targets = CALIBRATION_TARGETS
    target_text = f"""
    === CALIBRATION TARGETS ===
    Win Rate: {targets.target_win_rate*100:.0f}%
    IRR (wins): P50={targets.target_irr_p50_wins*100:.1f}%, P25={targets.target_irr_p25_wins*100:.1f}%, P75={targets.target_irr_p75_wins*100:.1f}%
    MOIC (wins): P50={targets.target_moic_p50_wins:.2f}x, P25={targets.target_moic_p25_wins:.2f}x, P75={targets.target_moic_p75_wins:.2f}x
    """
    
    summary_text = f"""
    SIMULATION SUMMARY
    
    Total Runs: {stats['total_runs']}
    Valid Runs: {stats['valid_runs']}
    Invalid Runs: {stats['invalid_runs']} (conservation violations)
    
    Wins: {stats['wins']} ({stats['win_rate']:.1f}% of valid)
    Losses: {stats['losses']} ({100-stats['win_rate']:.1f}% of valid)
    
    IRR STATUS
    Valid: {stats['irr_valid_count']} ({stats['irr_valid_count']/stats['n']*100:.1f}%)
    No sign change: {stats['irr_no_sign_change']}
    Did not converge: {stats['irr_no_converge']}
    
    IRR THRESHOLDS (of valid only)
    > 0%: {stats['irr_above_0_pct']:.1f}%
    > 10%: {stats['irr_above_10_pct']:.1f}%
    > 15%: {stats['irr_above_15_pct']:.1f}%
    
    Conservation violations: {stats['conservation_violations']}
    """
    
    ax7.text(0.1, 0.5, summary_text, transform=ax7.transAxes,
            fontsize=12, verticalalignment='center', fontfamily='monospace', color='black',
            bbox=dict(boxstyle='round', facecolor='wheat', edgecolor='black', linewidth=2))
    
    # 8. Failure Details Table (row 3 middle)
    ax8 = plt.subplot(4, 3, 8)
    ax8.axis('off')
    
    if failure_analysis:
        # Top 5 failure reasons with details
        sorted_failures = sorted(failure_analysis.items(), 
                               key=lambda x: x[1]['count'], reverse=True)[:5]
        
        table_data = [['Reason', 'Count', 'Avg Month', 'Avg Units']]
        for reason, data in sorted_failures:
            table_data.append([
                reason[:20],  # Truncate long names
                str(data['count']),
                f"{data['avg_month']:.1f}",
                f"{data['avg_units_sold']:.0f}"
            ])
        
        table = ax8.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.4, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Header styling
        for i in range(4):
            table[(0, i)].set_facecolor('#FF5722')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Make all text black
        for key, cell in table.get_celld().items():
            if key[0] > 0:  # Not header
                cell.set_text_props(color='black')
    
    ax8.set_title('Top Failure Reasons Detail', fontweight='bold', fontsize=14, color='black', pad=20)
    
    # 9. Equity Recovery Stats (row 3 right)
    ax9 = plt.subplot(4, 3, 9)
    ax9.axis('off')
    
    if stats['equity_recovery_stats']:
        eq_s = stats['equity_recovery_stats']
        table_data = [
            ['Metric', 'Value'],
            ['Mean', f"{eq_s['mean']:.1f}%"],
            ['Median', f"{eq_s['median']:.1f}%"],
            ['Std Dev', f"{eq_s['std']:.1f}%"],
            ['Min', f"{eq_s['min']:.1f}%"],
            ['P25', f"{eq_s['p25']:.1f}%"],
            ['P50', f"{eq_s['p50']:.1f}%"],
            ['P75', f"{eq_s['p75']:.1f}%"],
            ['Max', f"{eq_s['max']:.1f}%"],
        ]
        
        table = ax9.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        
        # Header styling
        for i in range(2):
            table[(0, i)].set_facecolor('#9C27B0')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Make all text black
        for key, cell in table.get_celld().items():
            if key[0] > 0:  # Not header
                cell.set_text_props(color='black')
    
    ax9.set_title('Equity Recovery Stats', fontweight='bold', fontsize=14, color='black', pad=20)
    
    # 10. Stress Incidence Bar Chart (row 4 left)
    ax10 = plt.subplot(4, 3, 10, facecolor='white')
    ax10.set_facecolor('white')
    
    if stress_analysis:
        inc = stress_analysis['incidence']
        stress_names = ['Delay', 'Capex', 'Rate', 'Sales', 'Event', 'Gate Block']
        stress_freqs = [inc['delay'], inc['capex'], inc['rate'], inc['sales'], inc['event'], inc['gate_block']]
        
        bars = ax10.barh(stress_names, stress_freqs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F'])
        ax10.set_xlabel('% of Runs Affected', fontsize=11, color='black', fontweight='bold')
        ax10.set_title('Stress Incidence (Frequency)', fontweight='bold', fontsize=14, color='black', pad=10)
        ax10.grid(True, axis='x', alpha=0.3, color='gray')
        ax10.set_xlim(0, 100)
        
        # Add percentage labels
        for i, (bar, freq) in enumerate(zip(bars, stress_freqs)):
            if freq > 0:
                ax10.text(freq + 2, i, f'{freq:.1f}%', va='center', fontsize=10, color='black', fontweight='bold')
    
    # 11. Stress Impact Bar Chart (row 4 middle)
    ax11 = plt.subplot(4, 3, 11, facecolor='white')
    ax11.set_facecolor('white')
    
    if stress_analysis and stress_analysis['impact']:
        imp = stress_analysis['impact']
        stress_names = []
        delta_wins = []
        
        for stress_name, stress_label in [('delay', 'Delay'), ('capex', 'Capex'), ('rate', 'Rate'), 
                                          ('sales', 'Sales'), ('event', 'Event'), ('gate_block', 'Gate Block')]:
            if imp[stress_name]:
                stress_names.append(stress_label)
                delta_wins.append(imp[stress_name]['delta_win'])
        
        if stress_names:
            colors = ['#FF6B6B' if dw < 0 else '#4ECDC4' for dw in delta_wins]
            bars = ax11.barh(stress_names, delta_wins, color=colors)
            ax11.set_xlabel('Δ Win Rate (%)', fontsize=11, color='black', fontweight='bold')
            ax11.set_title('Stress Impact (Win Rate Effect)', fontweight='bold', fontsize=14, color='black', pad=10)
            ax11.grid(True, axis='x', alpha=0.3, color='gray')
            ax11.axvline(0, color='black', linewidth=1.5)
            
            # Add delta labels
            for i, (bar, dw) in enumerate(zip(bars, delta_wins)):
                x_pos = dw + (1 if dw > 0 else -1)
                align = 'left' if dw > 0 else 'right'
                ax11.text(x_pos, i, f'{dw:+.1f}%', va='center', ha=align, fontsize=10, color='black', fontweight='bold')
    
    # 12. Stress Magnitude Table (row 4 right)
    ax12 = plt.subplot(4, 3, 12)
    ax12.axis('off')
    
    if stress_analysis:
        mag = stress_analysis['magnitudes']
        
        table_data = [['Stress', 'Avg Magnitude']]
        if mag['delay_months'] > 0:
            table_data.append(['Delay', f"{mag['delay_months']:.1f} mos"])
        if mag['capex_overrun_pct'] > 0:
            table_data.append(['Capex', f"{mag['capex_overrun_pct']:.1f}%"])
        if mag['max_rate_bps'] > 0:
            table_data.append(['Rate', f"{mag['max_rate_bps']:.0f} bps"])
        if mag['sales_shock_months'] > 0:
            table_data.append(['Sales', f"{mag['min_sales_mult']*100:.1f}% vel"])
        if mag['total_event_cost'] > 0:
            table_data.append(['Event', f"€{mag['total_event_cost']/1e6:.1f}M"])
        if mag['gate_blocked_months'] > 0:
            table_data.append(['Gate', f"{mag['gate_blocked_months']:.1f} mos"])
        
        if len(table_data) > 1:
            table = ax12.table(cellText=table_data, cellLoc='left', loc='center',
                             colWidths=[0.4, 0.6])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Header styling
            for i in range(2):
                table[(0, i)].set_facecolor('#FF9800')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Make all text black
            for key, cell in table.get_celld().items():
                if key[0] > 0:  # Not header
                    cell.set_text_props(color='black')
    
    ax12.set_title('Stress Magnitudes', fontweight='bold', fontsize=14, color='black', pad=20)
    
    # Compute and print stress analysis
    if stress_analysis:
        print(f"\n{'='*80}")
        print("STRESS INCIDENCE & IMPACT ANALYSIS")
        print(f"{'='*80}\n")
        
        # Incidence table
        print("STRESS INCIDENCE (Frequency & Magnitude)")
        print("-" * 80)
        print(f"{'Stress Type':<20} {'Frequency':>12} {'Avg Magnitude':>25}")
        print("-" * 80)
        
        inc = stress_analysis['incidence']
        mag = stress_analysis['magnitudes']
        
        print(f"{'Construction Delay':<20} {inc['delay']:>11.1f}% {mag['delay_months']:>14.1f} months")
        print(f"{'Capex Overrun':<20} {inc['capex']:>11.1f}% {mag['capex_overrun_pct']:>14.1f}% (€{mag['capex_overrun_amount']/1e6:,.1f}M)")
        print(f"{'Rate Shock':<20} {inc['rate']:>11.1f}% {mag['max_rate_bps']:>14.0f} bps ({mag['rate_shock_months']:.1f} mos)")
        print(f"{'Sales Shock':<20} {inc['sales']:>11.1f}% {mag['min_sales_mult']*100:>14.1f}% velocity ({mag['sales_shock_months']:.1f} mos)")
        print(f"{'Event Cost':<20} {inc['event']:>11.1f}% €{mag['total_event_cost']/1e6:>13.1f}M ({mag['event_count']:.1f} events)")
        print(f"{'Bank Gate Block':<20} {inc['gate_block']:>11.1f}% {mag['gate_blocked_months']:>14.1f} months blocked")
        
        # Impact table
        print(f"\n\nSTRESS IMPACT (Effect on Returns)")
        print("-" * 80)
        print(f"{'Stress Type':<20} {'Win% With':>12} {'Win% Without':>14} {'Delta Win%':>12} {'Delta IRR':>12}")
        print("-" * 80)
        
        imp = stress_analysis['impact']
        
        for stress_name, stress_label in [
            ('delay', 'Construction Delay'),
            ('capex', 'Capex Overrun'),
            ('rate', 'Rate Shock'),
            ('sales', 'Sales Shock'),
            ('event', 'Event Cost'),
            ('gate_block', 'Bank Gate Block'),
        ]:
            if imp[stress_name]:
                data = imp[stress_name]
                print(f"{stress_label:<20} {data['win_rate_with']:>11.1f}% {data['win_rate_without']:>13.1f}% {data['delta_win']:>11.1f}% {data['delta_irr']:>11.1f}%")
            else:
                print(f"{stress_label:<20} {'N/A':>11} {'N/A':>13} {'N/A':>11} {'N/A':>11}")
        
        print("-" * 80)
    
    # Ensure proper layout (use rect to avoid overlap)
    # Suppress warning about uneven grid (4x3 = 12 subplots)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save or show
    if save_path:
        # Save mode - render to file
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"\n[OK] Chart saved to: {save_path}")
        plt.close(fig)
    else:
        # Display mode - force immediate rendering and keep window open
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show(block=True)  # Block to keep window open
    
    # Print sample runs to console
    samples = extract_sample_runs(results)
    print("\n" + "="*80)
    print("SAMPLE RUN AUDIT TRAIL")
    print("="*80)
    
    if samples['best_wins']:
        print("\n--- TOP 3 WINS (by IRR) ---")
        for i, run in enumerate(samples['best_wins'], 1):
            print(f"\n{i}. IRR: {run['irr']*100:.1f}%, MOIC: {run['moic']:.2f}x, Duration: {run['duration']} months")
            print(f"   Units Sold: {run['units_sold']}, Final Cash: €{run['final_cash']:,}")
    
    if samples['diverse_losses']:
        print("\n--- SAMPLE LOSSES (by Failure Reason) ---")
        for i, run in enumerate(samples['diverse_losses'], 1):
            reason = run.get('failure_reason', 'UNKNOWN')
            print(f"\n{i}. {reason}")
            print(f"   Duration: {run['duration']} months, Units Sold: {run['units_sold']}")
            print(f"   Final Cash: €{run['final_cash']:,}, Debt: €{run['final_debt']:,}")
            print(f"   Cash Trough: €{run['cash_trough']:,}")

# ---------- UI ----------
class Button:
    def __init__(self, rect, text, on_click, accent=False, tooltip=""):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.on_click = on_click
        self.enabled = True
        self.accent = accent
        self.tooltip = tooltip

    def draw(self, screen, font, mouse_pos):
        if not self.enabled:
            color = BUTTON_DISABLED
            border_color = (60, 65, 75)
        elif self.accent:
            color = BUTTON_ACCENT_HOVER if self.rect.collidepoint(mouse_pos) else BUTTON_ACCENT
            border_color = (120, 220, 150)
        else:
            color = BUTTON_HOVER if self.rect.collidepoint(mouse_pos) else BUTTON_NORMAL
            border_color = (100, 150, 220)
        
        pygame.draw.rect(screen, color, self.rect, border_radius=12)
        pygame.draw.rect(screen, border_color, self.rect, width=3, border_radius=12)
        txt = font.render(self.text, True, TEXT_PRIMARY if self.enabled else (100, 110, 120))
        text_rect = txt.get_rect(center=self.rect.center)
        screen.blit(txt, text_rect)
        
        # Draw tooltip on hover
        if self.tooltip and self.rect.collidepoint(mouse_pos):
            self.draw_tooltip(screen, mouse_pos)
    
    def draw_tooltip(self, screen, mouse_pos):
        """Draw tooltip near mouse"""
        tooltip_font = pygame.font.SysFont("segoe ui", 13)
        lines = self.tooltip.split('\n')
        
        # Calculate tooltip size
        max_width = max(tooltip_font.size(line)[0] for line in lines)
        line_height = 18
        tooltip_w = max_width + 20
        tooltip_h = len(lines) * line_height + 10
        
        # Position tooltip (avoid going off screen)
        tooltip_x = mouse_pos[0] + 15
        tooltip_y = mouse_pos[1] - tooltip_h - 10
        if tooltip_x + tooltip_w > WIDTH - 10:
            tooltip_x = mouse_pos[0] - tooltip_w - 15
        if tooltip_y < 10:
            tooltip_y = mouse_pos[1] + 25
        
        # Draw tooltip background
        pygame.draw.rect(screen, (25, 35, 50), (tooltip_x, tooltip_y, tooltip_w, tooltip_h), border_radius=8)
        pygame.draw.rect(screen, (100, 120, 160), (tooltip_x, tooltip_y, tooltip_w, tooltip_h), width=2, border_radius=8)
        
        # Draw text lines
        y_offset = tooltip_y + 8
        for line in lines:
            text = tooltip_font.render(line, True, (220, 230, 245))
            screen.blit(text, (tooltip_x + 10, y_offset))
            y_offset += line_height

    def handle(self, event):
        if not self.enabled:
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.on_click()

def draw_bar(screen, x, y, w, h, value01, label, font):
    value01 = clamp(value01, 0.0, 1.0)
    # Background
    pygame.draw.rect(screen, (30, 40, 55), (x, y, w, h), border_radius=10)
    pygame.draw.rect(screen, (80, 100, 140), (x, y, w, h), width=2, border_radius=10)
    # Progress fill with gradient effect
    fill_w = int((w-6)*value01)
    if fill_w > 0:
        pygame.draw.rect(screen, PROGRESS_BAR, (x+3, y+3, fill_w, h-6), border_radius=8)
    # Label
    txt = font.render(label, True, TEXT_PRIMARY)
    screen.blit(txt, (x, y - 26))

def draw_cash_indicator(screen, cash, max_cash, x, y, w, h, font, big_font):
    """Draw a large, prominent cash indicator with depletion bar"""
    # Determine color based on cash level
    cash_ratio = cash / max_cash if max_cash > 0 else 0
    
    if cash < 0:
        bar_color = NEGATIVE_COLOR
        bg_color = (80, 30, 30)
        status = "BANKRUPT!"
    elif cash < 500_000:
        bar_color = NEGATIVE_COLOR
        bg_color = (80, 40, 40)
        status = "CRITICAL"
    elif cash < 2_000_000:
        bar_color = WARNING_COLOR
        bg_color = (80, 70, 40)
        status = "LOW"
    elif cash < 5_000_000:
        bar_color = (120, 180, 140)
        bg_color = (40, 60, 50)
        status = "MODERATE"
    else:
        bar_color = POSITIVE_COLOR
        bg_color = (40, 70, 50)
        status = "HEALTHY"
    
    # Background panel
    pygame.draw.rect(screen, bg_color, (x, y, w, h), border_radius=12)
    pygame.draw.rect(screen, bar_color, (x, y, w, h), width=3, border_radius=12)
    
    # Cash amount (big and centered)
    cash_text = big_font.render(money(cash), True, TEXT_PRIMARY)
    screen.blit(cash_text, (x + w//2 - cash_text.get_width()//2, y + 15))
    
    # Status label
    status_text = font.render(f"💰 Cash Position: {status}", True, TEXT_SECONDARY)
    screen.blit(status_text, (x + w//2 - status_text.get_width()//2, y + 60))
    
    # Depletion bar
    bar_y = y + h - 25
    bar_h = 15
    pygame.draw.rect(screen, (20, 25, 35), (x + 10, bar_y, w - 20, bar_h), border_radius=8)
    
    # Fill based on cash ratio (capped at 1.0)
    fill_ratio = min(1.0, max(0.0, cash_ratio))
    fill_w = int((w - 24) * fill_ratio)
    if fill_w > 0:
        pygame.draw.rect(screen, bar_color, (x + 12, bar_y + 2, fill_w, bar_h - 4), border_radius=6)
    
    # Show percentage
    pct_text = font.render(f"{min(100, int(cash_ratio * 100))}%", True, TEXT_SECONDARY)
    screen.blit(pct_text, (x + w//2 - pct_text.get_width()//2, y + h - 45))

def draw_building(screen, progress, x, y, w, h):
    """Draw a simple building that grows with construction progress"""
    # Ground
    pygame.draw.rect(screen, (60, 80, 60), (x, y + h - 10, w, 10))
    
    # Building progress (0-100%)
    progress = clamp(progress, 0, 100)
    building_h = int((h - 40) * (progress / 100.0))
    
    if building_h > 0:
        # Building structure
        building_y = y + h - 10 - building_h
        building_w = int(w * 0.8)
        building_x = x + (w - building_w) // 2
        
        # Main building
        pygame.draw.rect(screen, (80, 90, 110), (building_x, building_y, building_w, building_h))
        pygame.draw.rect(screen, (120, 140, 160), (building_x, building_y, building_w, building_h), width=2)
        
        # Windows (if enough progress)
        if progress > 20:
            window_rows = max(1, int((building_h - 10) / 25))
            window_cols = max(2, int(building_w / 35))
            
            for row in range(window_rows):
                for col in range(window_cols):
                    wx = building_x + 10 + col * (building_w - 20) // (window_cols)
                    wy = building_y + 10 + row * 20
                    if wy + 12 < building_y + building_h:
                        # Window color based on progress
                        if progress >= 70:
                            window_color = (255, 220, 120)  # Lights on
                        else:
                            window_color = (100, 150, 180)  # Under construction
                        pygame.draw.rect(screen, window_color, (wx, wy, 12, 12))
        
        # Crane if not complete
        if progress < 100:
            crane_x = building_x + building_w + 5
            crane_y = building_y - 20
            pygame.draw.line(screen, (200, 180, 100), (crane_x, crane_y), (crane_x, y + h - 10), 3)
            pygame.draw.line(screen, (200, 180, 100), (crane_x, crane_y), (crane_x + 30, crane_y + 10), 3)
    
    # Progress percentage
    font = pygame.font.SysFont("segoe ui", 16, bold=True)
    prog_text = font.render(f"{progress:.0f}%", True, TEXT_PRIMARY)
    screen.blit(prog_text, (x + w//2 - prog_text.get_width()//2, y + 5))

def draw_card(screen, x, y, w, h, title, rows, fonts, border_color=(80, 100, 140)):
    """Draw a standard card with fixed height and max 4 rows"""
    pygame.draw.rect(screen, (35, 45, 60), (x, y, w, h), border_radius=10)
    pygame.draw.rect(screen, border_color, (x, y, w, h), width=2, border_radius=10)
    
    # Title
    title_surf = fonts['title'].render(title, True, TEXT_PRIMARY)
    screen.blit(title_surf, (x + 10, y + 8))
    
    # Rows (max 4)
    row_y = y + 30
    row_spacing = (h - 35) // min(len(rows), 4)
    for i, (label, value, color) in enumerate(rows[:4]):
        if label:
            label_surf = fonts['small'].render(label, True, TEXT_SECONDARY)
            screen.blit(label_surf, (x + 15, row_y))
        if value:
            value_surf = fonts['body'].render(value, True, color)
            screen.blit(value_surf, (x + 15 + (140 if label else 0), row_y))
        row_y += row_spacing

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Liquidity Crunch: Greenfield Developer")
    clock = pygame.time.Clock()

    # Strict font hierarchy
    font_title = pygame.font.SysFont("segoe ui", FONT_TITLE, bold=True)
    font_body = pygame.font.SysFont("segoe ui", FONT_BODY)
    font_small = pygame.font.SysFont("segoe ui", FONT_SMALL)
    font_micro = pygame.font.SysFont("segoe ui", FONT_MICRO)
    
    # Legacy fonts for header
    big = pygame.font.SysFont("segoe ui", 24, bold=True)
    mid = pygame.font.SysFont("segoe ui", 18, bold=True)

    eng = Engine()

    # AI player state
    ai_enabled = False
    ai_tick_timer = 0
    ai_tick_delay = 0.3  # seconds between AI moves
    
    # Monte Carlo results
    mc_stats = None
    mc_running = False
    sim_progress = None
    
    # Define callback functions first
    def advance_month(eng):
        if not eng.gs.game_over and not ai_enabled:
            eng.tick_month()
    
    def toggle_ai():
        nonlocal ai_enabled
        ai_enabled = not ai_enabled
        ai_toggle_btn.text = "⏸ PAUSE AI" if ai_enabled else "🤖 LET AI PLAY"
    
    def run_mc(n, settings=None):
        nonlocal mc_stats, mc_running, sim_progress
        mc_running = True
        mc_stats = None
        sim_progress = SimulationProgress(
            total=n, 
            start_time=time.time(),
            settings=settings if settings else SimulationSettings.baseline()
        )
    
    def cancel_mc():
        nonlocal sim_progress
        if sim_progress:
            sim_progress.cancelled = True
    
    def restart(eng):
        nonlocal ai_enabled, mc_stats, sim_progress, mc_running
        ai_enabled = False
        mc_stats = None
        sim_progress = None
        mc_running = False
        eng.__init__()

    # Manual month advancement (no auto-tick)
    paused = True

    # Buttons with tooltips
    buttons = []
    buttons.append(Button((940, 170, 410, 55), "💰 Raise Debt (+€50m)", eng.action_raise_debt,
                         tooltip="Draw from €600m debt facility\nRequires bank gate unlocked\nIncreases monthly interest\nMax €600m total"))
    buttons.append(Button((940, 235, 410, 55), "🐌 Slow Construction (4m)", eng.action_slow_construction,
                         tooltip="Reduce construction pace for 4 months\nReduces dev costs by 30%\nAlso slows progress by 30%"))
    buttons.append(Button((940, 300, 410, 55), "📢 Sales Push (3m)", eng.action_sales_push,
                         tooltip="Launch marketing campaign\nCosts €5m upfront\nBoosts sales by 60% for 3 months"))
    buttons.append(Button((940, 365, 410, 55), "🔧 Resolve Event", eng.action_resolve_event,
                         tooltip="Pay to mitigate worst active event\nCost €10m+ (scales with difficulty)\nReduces event impact by 65%\nShortens duration by 3 months"))
    buttons.append(Button((940, 430, 410, 55), "💉 Equity Injection (+€10m)", eng.action_equity_injection,
                         tooltip="Inject equity capital\nAdds up to €10m cash\nPenalizes final score\n€50m available beyond land cost"))

    ai_toggle_btn = Button((940, 510, 410, 50), "🤖 LET AI PLAY", lambda: toggle_ai(), accent=True,
                          tooltip="Let AI play automatically until game ends\nClick again to pause")
    next_month_btn = Button((940, 570, 410, 60), "▶ NEXT MONTH", lambda: advance_month(eng), accent=True,
                           tooltip="Advance to the next month\nEvents will be processed\nCash flows will update")
    
    def show_analytics():
        """Show detailed analytics window"""
        nonlocal sim_progress
        if sim_progress and sim_progress.results:
            import os
            import subprocess
            
            # Create output directory if needed
            os.makedirs('output', exist_ok=True)
            
            # Save plot to file
            save_path = os.path.abspath('output/simulation_results.png')
            show_simulation_results(sim_progress.results, save_path=save_path)
            
            # Open the saved image with default viewer
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(save_path)
                elif os.name == 'posix':  # macOS/Linux
                    subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', save_path])
                print(f"Analytics chart opened: {save_path}")
            except Exception as e:
                print(f"Chart saved to: {save_path}")
                print(f"Please open it manually. Error: {e}")
    
    mc_100_btn = Button((940, 640, 200, 40), "MC: 100x", lambda: run_mc(100),
                       tooltip="Run 100 Monte Carlo simulations")
    mc_1000_btn = Button((1145, 640, 205, 40), "MC: 1000x", lambda: run_mc(1000),
                        tooltip="Run 1000 Monte Carlo simulations")
    mc_cancel_btn = Button((940, 690, 410, 35), "❌ Cancel Simulation", cancel_mc,
                          tooltip="Stop the running simulation")
    mc_analytics_btn = Button((940, 690, 410, 35), "📊 Show Analytics", show_analytics, accent=True,
                             tooltip="Open detailed analytics window with charts and statistics")
    
    reset_btn = Button((940, 740, 410, 50), "🔄 Restart Game", lambda: restart(eng),
                      tooltip="Start a new game from scratch")

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        mouse_pos = pygame.mouse.get_pos()

        # Enable/disable buttons based on state
        gs = eng.gs
        buttons[3].enabled = (len(gs.active_events) > 0 and not gs.game_over and not ai_enabled)
        buttons[4].enabled = (gs.equity_remaining > 0 and not gs.game_over and not ai_enabled)
        # debt can be locked or gate not met
        buttons[0].enabled = (gs.bank_gate_met and not eng.has_debt_lock() and not gs.game_over and gs.debt < gs.max_debt and not ai_enabled)
        for i in [1,2]:
            buttons[i].enabled = (not gs.game_over and not ai_enabled)
        
        next_month_btn.enabled = (not gs.game_over and not ai_enabled)
        mc_100_btn.enabled = not mc_running and not gs.game_over
        mc_1000_btn.enabled = not mc_running and not gs.game_over
        mc_cancel_btn.enabled = mc_running
        mc_analytics_btn.enabled = (mc_stats is not None and sim_progress is not None and sim_progress.results)
        
        # AI player tick
        if ai_enabled and not gs.game_over:
            ai_tick_timer += dt
            if ai_tick_timer >= ai_tick_delay:
                ai_tick_timer = 0
                eng.ai_decide_action()
                eng.tick_month()
        
        # Monte Carlo simulation tick (run in chunks)
        if mc_running and sim_progress:
            batch_size = 25  # Run 25 sims per frame
            has_more = run_monte_carlo_chunk(sim_progress, batch_size, settings=sim_progress.settings)
            
            if not has_more:
                # Simulation complete or cancelled
                mc_running = False
                mc_stats = sim_progress.compute_stats()
                if sim_progress.cancelled:
                    print(f"Simulation cancelled at {sim_progress.completed}/{sim_progress.total}")

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            for b in buttons:
                b.handle(event)
            ai_toggle_btn.handle(event)
            next_month_btn.handle(event)
            if not mc_running:
                mc_100_btn.handle(event)
                mc_1000_btn.handle(event)
                if mc_stats and sim_progress and sim_progress.results:
                    mc_analytics_btn.handle(event)
            else:
                mc_cancel_btn.handle(event)
            if gs.game_over:
                reset_btn.handle(event)

        # Draw background
        screen.fill(BG_COLOR)

        # Header with shadow effect
        title = big.render("💼 LIQUIDITY CRUNCH", True, TEXT_PRIMARY)
        subtitle = mid.render("Greenfield Property Development Simulator", True, TEXT_SECONDARY)
        screen.blit(title, (40, 25))
        screen.blit(subtitle, (40, 65))

        # Calculate metrics
        rate = eng.current_rate()
        sold_ratio = gs.units_sold / gs.units_total
        money_multiple, irr = eng.calculate_returns()
        runway = eng.calculate_runway()
        
        # === STRICT VERTICAL GRID LAYOUT ===
        current_y = TOP_Y
        
        # Month indicator (above cards)
        month_text = font_body.render(f"Month {gs.month} / 30 | Phase: {gs.phase}", True, TEXT_SECONDARY)
        screen.blit(month_text, (LEFT_X + 10, current_y + 5))
        current_y += 30
        
        # Font dict for card drawing
        fonts = {'title': font_title, 'body': font_body, 'small': font_small, 'micro': font_micro}
        
        # === PROGRESS BAR (between header and cards) ===
        prog_y = current_y
        prog_h = CARD_HEIGHTS["progress_bar"]
        pygame.draw.rect(screen, PANEL_BG, (LEFT_X, prog_y, LEFT_WIDTH, prog_h), border_radius=10)
        pygame.draw.rect(screen, PANEL_BORDER, (LEFT_X, prog_y, LEFT_WIDTH, prog_h), width=2, border_radius=10)
        
        # Progress info
        prog_text = font_title.render(f"Construction: {gs.progress:.0f}%", True, TEXT_PRIMARY)
        screen.blit(prog_text, (LEFT_X + 10, prog_y + 8))
        
        # Progress bar
        bar_y = prog_y + 28
        bar_w = LEFT_WIDTH - 20
        bar_h = 14
        pygame.draw.rect(screen, (30, 40, 50), (LEFT_X + 10, bar_y, bar_w, bar_h), border_radius=6)
        if gs.progress > 0:
            fill_w = int(bar_w * (gs.progress / 100.0))
            pygame.draw.rect(screen, PROGRESS_BAR, (LEFT_X + 11, bar_y + 1, fill_w, bar_h - 2), border_radius=5)
        
        current_y += prog_h + CARD_GAP
        
        # ==========================================
        # === CARD 1: LIQUIDITY ===
        # ==========================================
        card_h = CARD_HEIGHTS["liquidity"]
        cash_color = POSITIVE_COLOR if gs.cash > 5_000_000 else (WARNING_COLOR if gs.cash > 0 else NEGATIVE_COLOR)
        flow_color = POSITIVE_COLOR if gs.last_month_burn >= 0 else NEGATIVE_COLOR
        flow_sign = "+" if gs.last_month_burn >= 0 else ""
        
        if runway == float('inf'):
            runway_text = "∞"
            runway_color = POSITIVE_COLOR
        elif runway > 3:
            runway_text = f"{runway:.1f}m"
            runway_color = POSITIVE_COLOR
        elif runway > 1:
            runway_text = f"{runway:.1f}m"
            runway_color = WARNING_COLOR
        else:
            runway_text = f"{runway:.1f}m"
            runway_color = NEGATIVE_COLOR
        
        liq_rows = [
            ("Cash:", money(gs.cash), cash_color),
            ("Burn:", f"{flow_sign}{money(gs.last_month_burn)}", flow_color),
            ("Runway:", runway_text, runway_color),
        ]
        draw_card(screen, LEFT_X, current_y, LEFT_WIDTH, card_h, "💰 LIQUIDITY", liq_rows, fonts, border_color=(100, 130, 180))
        current_y += card_h + CARD_GAP
        
        # ==========================================
        # === CARD 2: EQUITY ===
        # ==========================================
        # Conditional: hide in collapsed mode (early game)
        if gs.month >= 1:  # Always show for now
            card_h = CARD_HEIGHTS["equity"]
            additional_equity = gs.equity_used - LAND_COST
            additional_available = TOTAL_EQUITY - LAND_COST
            
            eq_rows = [
                ("Land:", f"{money(LAND_COST)} PAID", TEXT_SECONDARY),
                ("Additional:", f"{money(additional_equity)} / {money(additional_available)}", TEXT_SECONDARY),
                ("Total Used:", f"{money(gs.equity_used)} / {money(TOTAL_EQUITY)}", TEXT_PRIMARY),
            ]
            draw_card(screen, LEFT_X, current_y, LEFT_WIDTH, card_h, "🏦 EQUITY", eq_rows, fonts)
            current_y += card_h + CARD_GAP
        
        # ==========================================
        # === CARD 3: DEBT ===
        # ==========================================
        card_h = CARD_HEIGHTS["debt"]
        debt_ratio = gs.debt / MAX_DEBT_FACILITY if MAX_DEBT_FACILITY > 0 else 0
        debt_rows = [
            ("Drawn:", f"{money(gs.debt)} / {money(MAX_DEBT_FACILITY)}", TEXT_PRIMARY),
            ("Rate:", f"{rate*100:.1f}%", TEXT_SECONDARY),
        ]
        draw_card(screen, LEFT_X, current_y, LEFT_WIDTH, card_h, "💳 DEBT", debt_rows, fonts)
        current_y += card_h + CARD_GAP
        
        # ==========================================
        # === CARD 4: SALES ===
        # ==========================================
        card_h = CARD_HEIGHTS["sales"]
        sales_value = gs.units_sold * UNIT_PRICE
        sales_rows = [
            ("Units:", f"{gs.units_sold} / {gs.units_total}", TEXT_PRIMARY),
            ("Value:", f"{money(sales_value)} / {money(GDV)}", TEXT_SECONDARY),
        ]
        draw_card(screen, LEFT_X, current_y, LEFT_WIDTH, card_h, "📦 SALES", sales_rows, fonts)
        current_y += card_h + CARD_GAP
        
        # ==========================================
        # === CARD 5: DEPOSITS (conditional - hide early game) ===
        # ==========================================
        if gs.month >= 1:  # Show after first month
            card_h = CARD_HEIGHTS["deposits"]
            total_expected_deposits = int(GDV * 0.30)
            remaining_for_bank = max(0, BANK_GATE_DEPOSITS - gs.deposits_collected)
            remaining_color = POSITIVE_COLOR if remaining_for_bank == 0 else WARNING_COLOR
            
            dep_rows = [
                ("Collected:", f"{money(gs.deposits_collected)} / {money(total_expected_deposits)}", TEXT_PRIMARY),
                ("Bank Gate:", f"{money(BANK_GATE_DEPOSITS)}", TEXT_SECONDARY),
                ("To Unlock:", f"{money(remaining_for_bank)}", remaining_color),
            ]
            draw_card(screen, LEFT_X, current_y, LEFT_WIDTH, card_h, "💰 DEPOSITS", dep_rows, fonts)
            current_y += card_h + CARD_GAP
        
        # ==========================================
        # === CARD 6: BANK FINANCING (fixed height) ===
        # ==========================================
        card_h = CARD_HEIGHTS["bank"]
        units_threshold = int(TOTAL_UNITS * BANK_GATE_UNITS_PCT)
        units_met = gs.units_sold >= units_threshold
        deposits_met = gs.deposits_collected >= BANK_GATE_DEPOSITS
        
        gate_title = "🏦 BANK: 🔓 ACTIVE" if gs.bank_gate_met else "🏦 BANK: 🔒 LOCKED"
        gate_color_border = POSITIVE_COLOR if gs.bank_gate_met else NEGATIVE_COLOR
        
        check_u = "☑" if units_met else "☐"
        check_d = "☑" if deposits_met else "☐"
        col_u = POSITIVE_COLOR if units_met else TEXT_SECONDARY
        col_d = POSITIVE_COLOR if deposits_met else TEXT_SECONDARY
        
        bank_rows = [
            ("", f"{check_u} Units: {gs.units_sold}/{units_threshold}", col_u),
            ("", f"{check_d} Deposits: {money(gs.deposits_collected)}/{money(BANK_GATE_DEPOSITS)}", col_d),
        ]
        draw_card(screen, LEFT_X, current_y, LEFT_WIDTH, card_h, gate_title, bank_rows, fonts, border_color=gate_color_border)
        current_y += card_h + CARD_GAP
        
        # ==========================================
        # === CARD 7: COMPLETION PREVIEW (conditional) ===
        # ==========================================
        # Only show when progress >= 70% OR month >= 24
        if gs.progress >= 70.0 or gs.month >= 24:
            card_h = CARD_HEIGHTS["completion"]
            expected_deed = gs.units_sold * int(UNIT_PRICE * COMPLETION_PCT)
            debt_to_repay = min(gs.debt, expected_deed)
            net_at_completion = gs.cash + expected_deed - debt_to_repay
            net_color = POSITIVE_COLOR if net_at_completion > 0 else NEGATIVE_COLOR
            
            comp_rows = [
                ("Deed In:", money(expected_deed), TEXT_SECONDARY),
                ("Debt Out:", money(debt_to_repay), NEGATIVE_COLOR if debt_to_repay > 0 else TEXT_SECONDARY),
                ("Net Cash:", money(net_at_completion), net_color),
            ]
            draw_card(screen, LEFT_X, current_y, LEFT_WIDTH, card_h, "🎯 COMPLETION", comp_rows, fonts)
            current_y += card_h + CARD_GAP
        
        # ==========================================
        # === CARD 8: RETURNS (compact, passive) ===
        # ==========================================
        if gs.equity_used > 0 and gs.month >= 6:  # Hide early game
            card_h = CARD_HEIGHTS["returns"]
            ret_rows = [
                ("Multiple:", f"{money_multiple:.2f}x", TEXT_PRIMARY),
                ("IRR:", f"{irr*100:.1f}%", TEXT_PRIMARY),
            ]
            draw_card(screen, LEFT_X, current_y, LEFT_WIDTH, card_h, "📈 RETURNS", ret_rows, fonts)
            current_y += card_h + CARD_GAP

        # === CONSTRUCTION VISUAL (right side, top) ===
        right_col_x = 580
        draw_building(screen, gs.progress, right_col_x, 140, 250, 140)
        
        # Actions panel (right side, below construction visual)
        actions_x = 920
        actions_w = 450
        pygame.draw.rect(screen, PANEL_BG, (actions_x, 110, actions_w, 675), border_radius=16)
        pygame.draw.rect(screen, PANEL_BORDER, (actions_x, 110, actions_w, 675), width=3, border_radius=16)
        screen.blit(mid.render("⚙️ Actions", True, TEXT_PRIMARY), (actions_x+20, 125))

        for b in buttons:
            b.draw(screen, font_body, mouse_pos)
        
        ai_toggle_btn.draw(screen, font_body, mouse_pos)
        next_month_btn.draw(screen, font_body, mouse_pos)
        
        # Show simulation controls and progress
        if mc_running and sim_progress:
            # Draw cancel button
            mc_cancel_btn.draw(screen, font_small, mouse_pos)
            
            # Progress bar
            prog_y = 730
            prog_w = 410
            prog_h = 20
            pygame.draw.rect(screen, (30, 40, 50), (940, prog_y, prog_w, prog_h), border_radius=8)
            
            pct = sim_progress.progress_pct()
            if pct > 0:
                fill_w = int(prog_w * pct)
                pygame.draw.rect(screen, PROGRESS_BAR, (940, prog_y, fill_w, prog_h), border_radius=8)
            
            # Progress text
            prog_text = font_small.render(f"{int(pct*100)}%", True, TEXT_PRIMARY)
            screen.blit(prog_text, (1145 - prog_text.get_width()//2, prog_y + 4))
            
            # Status line
            status_y = 755
            elapsed = sim_progress.elapsed_time()
            eta = sim_progress.eta()
            
            elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
            eta_str = f"{int(eta//60):02d}:{int(eta%60):02d}"
            
            status_text = f"Running {sim_progress.completed} / {sim_progress.total} — {elapsed_str} elapsed — ETA {eta_str}"
            status_surf = font_micro.render(status_text, True, TEXT_SECONDARY)
            screen.blit(status_surf, (950, status_y))
            
            # Live preview stats (if we have some results)
            if sim_progress.completed >= 10 and sim_progress.completed % 25 == 0:
                preview_stats = sim_progress.compute_stats()
                if preview_stats:
                    preview_y = 772
                    preview_lines = [
                        f"Win: {preview_stats['win_rate']*100:.1f}% | IRR: {preview_stats['mean_irr']*100:.1f}% | MOIC: {preview_stats['mean_moic']:.2f}x"
                    ]
                    for line in preview_lines:
                        preview_surf = font_micro.render(line, True, POSITIVE_COLOR)
                        screen.blit(preview_surf, (950, preview_y))
                        preview_y += 12
        else:
            # Show MC buttons when not running
            mc_100_btn.draw(screen, font_small, mouse_pos)
            mc_1000_btn.draw(screen, font_small, mouse_pos)
            
            # Show analytics button if we have results
            if mc_stats and sim_progress and sim_progress.results:
                mc_analytics_btn.draw(screen, font_small, mouse_pos)
                # Show completed stats
                mc_text = font_small.render(f"✓ Win rate: {mc_stats['win_rate']*100:.1f}% | IRR: {mc_stats['mean_irr']*100:.1f}% | n={len(sim_progress.results)}", True, POSITIVE_COLOR)
                screen.blit(mc_text, (950, 730))
        
        # Active modifiers in actions panel
        mod_y = 595
        if gs.slow_build and gs.slow_build_months_left > 0:
            pygame.draw.rect(screen, (60, 80, 100), (945, mod_y, 400, 30), border_radius=8)
            screen.blit(font_small.render(f"🐌 Slow Build: {gs.slow_build_months_left} months left", True, TEXT_PRIMARY), (955, mod_y+7))
            mod_y += 35
        if gs.sales_push and gs.sales_push_months_left > 0:
            pygame.draw.rect(screen, (60, 100, 80), (945, mod_y, 400, 30), border_radius=8)
            screen.blit(font_small.render(f"📢 Sales Push: {gs.sales_push_months_left} months left", True, TEXT_PRIMARY), (955, mod_y+7))
            mod_y += 35
        
        # Active events in actions panel
        if gs.active_events:
            event_y = mod_y + 10
            screen.blit(mid.render("⚠️ Active Events", True, TEXT_PRIMARY), (945, event_y))
            event_y += 35
            for ev in gs.active_events[:3]:
                pygame.draw.rect(screen, (80, 50, 50), (945, event_y, 400, 28), border_radius=8)
                ev_text = f"{ev.name} ({ev.months_left}m)"
                screen.blit(font_small.render(ev_text, True, (255, 200, 200)), (955, event_y+6))
                event_y += 32

        # Game over overlay
        if gs.game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, 0))

            # Calculate IRR/MOIC
            irr, moic, equity_in, equity_out, irr_status = eng.calculate_irr_moic()

            # Result box (larger, scrollable)
            result_rect = pygame.Rect(200, 50, 1000, 700)
            pygame.draw.rect(screen, (30, 40, 55), result_rect, border_radius=20)
            pygame.draw.rect(screen, (100, 120, 160), result_rect, width=4, border_radius=20)

            result = "🎉 YOU WIN!" if gs.win else "💥 GAME OVER"
            result_color = POSITIVE_COLOR if gs.win else NEGATIVE_COLOR
            result_text = mid.render(result, True, result_color)
            screen.blit(result_text, (result_rect.centerx - result_text.get_width()//2, 70))

            # Performance section
            yy = 120
            screen.blit(font_title.render("PERFORMANCE", True, TEXT_PRIMARY), (220, yy))
            yy += 30
            
            sc = eng.score()
            perf_lines = [
                f"Final Score: {money(sc)}",
                f"Duration: {gs.month} months",
                f"Units Sold: {gs.units_sold}/{gs.units_total} ({gs.units_sold/gs.units_total*100:.0f}%)",
                f"Equity Used: {money(gs.equity_used)}/{money(TOTAL_EQUITY)}",
                f"Debt Drawn: {money(gs.debt)}/{money(gs.max_debt)}",
                f"Close Calls: {gs.near_miss_count}",
            ]
            for line in perf_lines:
                text = font_body.render(line, True, TEXT_SECONDARY)
                screen.blit(text, (230, yy))
                yy += 24
            
            # Returns section
            yy += 20
            screen.blit(font_title.render("RETURNS", True, TEXT_PRIMARY), (220, yy))
            yy += 30
            
            moic_color = POSITIVE_COLOR if moic > 1.0 else NEGATIVE_COLOR
            irr_color = POSITIVE_COLOR if irr > 0 else NEGATIVE_COLOR
            
            returns_lines = [
                (f"Money Multiple (MOIC): {moic:.2f}x", moic_color),
                (f"IRR (annual): {irr*100:.1f}%", irr_color),
                (f"Total Equity In: {money(int(equity_in))}", TEXT_SECONDARY),
                (f"Total Equity Out: {money(int(equity_out))}", TEXT_SECONDARY),
            ]
            for line, color in returns_lines:
                text = font_body.render(line, True, color)
                screen.blit(text, (230, yy))
                yy += 24
            
            # Monte Carlo results section (if available)
            if mc_stats:
                yy += 20
                screen.blit(font_title.render("MONTE CARLO SIMULATION", True, TEXT_PRIMARY), (220, yy))
                yy += 30
                
                mc_lines = [
                    f"Win Rate: {mc_stats['win_rate']*100:.1f}%",
                    "",
                    f"IRR - Mean: {mc_stats['mean_irr']*100:.1f}% | Median: {mc_stats['median_irr']*100:.1f}%",
                    f"IRR - P10: {mc_stats['p10_irr']*100:.1f}% | P50: {mc_stats['p50_irr']*100:.1f}% | P90: {mc_stats['p90_irr']*100:.1f}%",
                    "",
                    f"MOIC - Mean: {mc_stats['mean_moic']:.2f}x | Median: {mc_stats['median_moic']:.2f}x",
                    f"MOIC - P10: {mc_stats['p10_moic']:.2f}x | P50: {mc_stats['p50_moic']:.2f}x | P90: {mc_stats['p90_moic']:.2f}x",
                ]
                for line in mc_lines:
                    if line:
                        text = font_small.render(line, True, TEXT_SECONDARY)
                        screen.blit(text, (230, yy))
                    yy += 20
            
            # Stress summary section
            if any([gs.stresses.had_delay, gs.stresses.had_capex_overrun, gs.stresses.had_rate_shock,
                    gs.stresses.had_sales_shock, gs.stresses.had_event_cost, gs.stresses.bank_gate_blocked_months > 0]):
                yy += 20
                screen.blit(font_title.render("STRESSES IN THIS RUN", True, TEXT_PRIMARY), (220, yy))
                yy += 30
                
                stress_lines = []
                if gs.stresses.had_delay:
                    stress_lines.append(f"⚠ Construction Delay: {gs.stresses.delay_months} months")
                if gs.stresses.had_capex_overrun:
                    stress_lines.append(f"⚠ Capex Overrun: {money(int(gs.stresses.capex_overrun_amount))} ({gs.stresses.capex_overrun_pct:.1f}%)")
                if gs.stresses.had_rate_shock:
                    stress_lines.append(f"⚠ Rate Shock: max +{gs.stresses.max_rate_bps} bps for {gs.stresses.rate_shock_months} months")
                if gs.stresses.had_sales_shock:
                    stress_lines.append(f"⚠ Sales Shock: {gs.stresses.min_sales_multiplier*100:.1f}% velocity for {gs.stresses.sales_shock_months} months")
                if gs.stresses.had_event_cost:
                    stress_lines.append(f"⚠ Event Costs: {money(int(gs.stresses.total_event_cost))} ({gs.stresses.event_count} events)")
                if gs.stresses.bank_gate_blocked_months > 0:
                    stress_lines.append(f"⚠ Bank Gate Blocked: {gs.stresses.bank_gate_blocked_months} months")
                
                for line in stress_lines:
                    text = font_small.render(line, True, (255, 200, 100))
                    screen.blit(text, (230, yy))
                    yy += 22
            
            # Reason at bottom
            yy += 20
            reason_lines = wrap_text(gs.reason, font_small, 940)
            for line in reason_lines:
                text = font_small.render(line, True, TEXT_SECONDARY)
                screen.blit(text, (result_rect.centerx - text.get_width()//2, yy))
                yy += 18

            reset_btn.draw(screen, font_body, mouse_pos)

        pygame.display.flip()

    pygame.quit()

def wrap_text(text, font, max_width):
    words = text.split(" ")
    lines = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        if font.size(test)[0] <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

def draw_cash_chart(screen, cash_hist, x, y, w, h):
    # Simple outline box
    pygame.draw.rect(screen, (22, 22, 22), (x, y, w, h), border_radius=10)
    pygame.draw.rect(screen, (70, 70, 70), (x, y, w, h), width=2, border_radius=10)

    if len(cash_hist) < 2:
        return

    # scale
    mn = min(cash_hist)
    mx = max(cash_hist)
    if mn == mx:
        mx += 1

    # draw line
    pts = []
    n = len(cash_hist)
    for i, c in enumerate(cash_hist):
        px = x + 8 + int((w - 16) * (i / (n - 1)))
        t = (c - mn) / (mx - mn)
        py = y + h - 8 - int((h - 16) * t)
        pts.append((px, py))

    pygame.draw.lines(screen, (140, 140, 140), False, pts, 2)

    # labels
    f = pygame.font.SysFont("consolas", 14)
    screen.blit(f.render(f"Cash (min {money(mn)}, max {money(mx)})", True, (180, 180, 180)), (x+10, y+8))

if __name__ == "__main__":
    main()
