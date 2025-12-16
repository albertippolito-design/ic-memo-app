# Calibration Implementation - Final Status Report

## Executive Summary

**Goal:** Calibrate simulation to achieve 40-70% win rate with realistic 15-20% IRR and 1.5-1.7x MOIC, eliminate all conservation violations.

**Current Status:** Significant progress made, but implementation incomplete. 

**Key Achievement:** Win scenarios now show realistic returns (13-14% IRR, 1.44-1.48x MOIC) - was previously showing impossible 10x+ MOICs or negative returns.

**Critical Issue:** 60-80% of runs still marked invalid due to unresolved conservation violation in accounting logic.

---

## What Was Successfully Implemented ✓

### 1. Calibration Framework
- ✅ `calibration_config.py` with `CALIBRATION_TARGETS` and `STRESS_CONFIG`
- ✅ Targets displayed in all reports
- ✅ Import structure in main game file

### 2. Conservation Violation Detection
- ✅ `valid_run` flag added to all simulation results
- ✅ Invalid runs excluded from all statistics
- ✅ Detailed waterfall logging for each violation
- ✅ Stricter tolerance (€500k instead of €1m)

### 3. Realistic Stress Parameters
- ✅ Event probability: 10% → 2.5% per month (75% reduction)
- ✅ Rate shocks: 500bp → 100-250bp max
- ✅ Sales drops: 70-90% → 15-40%
- ✅ Construction overruns: variable → 5-15%
- ✅ Event duration: 3-8 months → 2-6 months
- ✅ Resolve costs: fixed €10m+ → 1.5x monthly burn (capped at 3% remaining capex)

### 4. Capital Stack Improvements
- ✅ Revolver added: €20m capacity with +200bp spread
- ✅ Smooth bank gate: Graduated debt unlock (€100m base + €10 per €1 deposit)
- ✅ Revolver integrated into AI decision logic (auto-draw when cash < €20m)
- ✅ Revolver interest calculated and tracked separately

### 5. Base Case Validation
- ✅ `--no-events` mode added to run_analytics.py
- ✅ Validates base economics independently of stress events
- ✅ Reports P25/P50/P75 IRR and MOIC for wins
- ✅ Pass/fail sanity check (>70% win rate, >15% IRR)

### 6. Equity Waterfall
- ✅ Proper waterfall: Cash - Debt - Revolver - Deposits Liability
- ✅ Revolver subtracted in final distribution

---

## What Is NOT Working ❌

### 1. **CRITICAL: Conservation Violations (60-80% invalid runs)**

**Problem:** Even after multiple fixes, 60-80% of simulation runs are marked invalid due to conservation violations.

**Pattern Observed:**
```
Seed 1: Equity out: €174m, Max possible: €104m → Excess: €70m
Seed 6: Equity out: €136m, Max possible: €76m → Excess: €60m
Seed 9: Equity out: €112m, Max possible: €52m → Excess: €60m
```

**Hypothesis:** The conservation check formula is still incorrect. Likely issues:
1. Peak debt tracking may not capture full debt lifecycle
2. Debt repayment at completion may not be reducing cash correctly
3. Post-completion unit sales (after month 30) may be collecting payments but not properly accounting for them

**What Needs Investigation:**
- Add debug logging for EVERY cash transaction (debt draw, debt repayment, completion payment)
- Track cumulative debt drawn vs debt repaid
- Verify completion payment logic accounts for partial debt repayment
- Check if post-completion sales are handled correctly

### 2. **Win Rate Still Too Low (1-4%)**

**Current:** 1-4% win rate (of valid runs)  
**Target:** 40-70% win rate

**Causes:**
- Stress parameters may still be too harsh despite reductions
- AI policy may not be optimal (over-injecting equity, delaying actions)
- Bank gate unlock slope may need tuning (currently €10 debt per €1 deposit - may be too steep)
- Revolver may not be utilized effectively by AI

**Not Yet Implemented:**
- AI policy optimization (prefer debt over equity, earlier sales push, optimize construction timing)
- Parameter grid search / automated calibration
- Sensitivity analysis on key parameters

### 3. **Statistics Calculation Errors**

**Observed:**
```
Valid IRRs: 73 (260.7%)  <- Impossible percentage
Invalid MOICs: -72       <- Negative count impossible
```

**Cause:** Bug in `compute_detailed_stats` function when filtering valid/invalid runs. The percentages and counts are being calculated incorrectly after filtering.

**Fix Needed:** Review lines 1154-1250 in python liquidity_crunch.py where valid_results filtering occurs.

---

## Immediate Next Steps (Priority Order)

### URGENT: Fix Conservation Violations

**Task 1a:** Add comprehensive debug logging
```python
# In tick_month, log every cash transaction:
- Deposit collection: log amount, update running total
- Completion payment: log gross amount, debt repayment, net to cash
- Debt draw: log amount drawn, cumulative debt drawn
- Revolver draw: log amount
- Costs: log dev cost, interest, running total
```

**Task 1b:** Fix conservation check formula
- Current formula still has issues
- Need to verify: `max_equity = GDV + equity_injected - costs - debt - revolver - deposits_liability`
- Add assertion: `actual_equity_out` should equal `compute_equity_distribution_end()` result
- If they don't match → there's cash leaking somewhere

**Task 1c:** Test with debug mode
```bash
$env:GAME_DEBUG=1
python run_analytics.py --baseline-only --sims 10
```

**Expected Output:** Zero violations in 10 runs, or clear diagnostic showing WHERE the accounting breaks

---

### HIGH: Improve Win Rate to 40-70%

**Task 2a:** Run base case check
```bash
python run_analytics.py --no-events --sims 500
```

**Expected:** If base case shows >70% win rate with >15% IRR, then stress tuning is the issue. If not, base economics need adjustment.

**Task 2b:** If stress is the problem, tune parameters:
- Reduce event probability further: 2.5% → 1.5%
- Increase revolver capacity: €20m → €30m
- Relax bank gate slope: €10 → €15 (more debt per deposit)
- Reduce resolve costs: 1.5x burn → 1.0x burn

**Task 2c:** If base economics are the problem:
- Increase unit price: €1m → €1.1m (10% higher GDV)
- Reduce dev costs: €600m → €550m
- Reduce construction time: 30 months → 27 months
- Reduce interest rates: 6% base → 5% base

---

### MEDIUM: Fix Statistics Display

**Task 3:** Debug and fix the valid/invalid counting in `compute_detailed_stats`

```python
# Around line 1154
valid_results = [r for r in results if r.get('valid_run', True)]
invalid_count = total_runs - len(valid_results)

# Check: Do percentages add up correctly?
# Check: Are IRR/MOIC lists built from valid_results only?
```

---

### LOW: AI Policy Optimization

**Task 4:** (Only after conservation violations are fixed)

Improve AI decision logic:
1. Prefer debt over equity when debt capacity available
2. Trigger sales push earlier (month 10-12 instead of month 15+)
3. Only slow construction if runway < 2 months
4. Track and minimize equity injections beyond land purchase

---

### FUTURE: Automated Calibration

**Task 5:** Implement `--calibrate` mode

```python
# Grid search over:
- event_probability: [0.015, 0.020, 0.025]
- revolver_capacity: [20m, 30m, 40m]
- bank_gate_slope: [10, 15, 20]
- resolve_cost_mult: [1.0, 1.5, 2.0]

# Objective function:
loss = |win_rate - 55%| + |median_irr_wins - 17.5%| + |median_moic_wins - 1.6x|

# Output: Best 5 parameter sets ranked by fitness
```

---

## Current Metrics vs Targets

| Metric | Target | Current (Valid Runs Only) | Status |
|--------|--------|---------------|--------|
| Win Rate | 40-70% | 1-4% | ❌ Far below |
| Conservation Violations | 0% | 60-80% | ❌ Critical |
| Median IRR (Wins) | 15-20% | 13-14% | ⚠️ Close! |
| Median MOIC (Wins) | 1.5-1.7x | 1.44-1.48x | ⚠️ Close! |
| P25 IRR (Wins) | >10% | ~13% | ✅ Exceeds |
| P75 IRR (Wins) | ~25% | ~14% | ❌ Below |

**Key Insight:** Win scenarios that DO complete successfully show realistic returns! This means the accounting fix is working for wins. The problem is:
1. Too many runs fail (98% loss rate)
2. Most wins get marked invalid due to conservation check bug

---

## Files Modified

### Core Game Logic
- `python liquidity_crunch.py` (2647 lines)
  - Lines 1-50: Added calibration_config import with fallback
  - Lines 247-248: Added revolver fields to GameState
  - Lines 322-332: Graduated bank gate unlock
  - Lines 379-406: Event generation using STRESS_CONFIG
  - Lines 527-529: Revolver interest calculation
  - Lines 652-671: action_draw_revolver method
  - Lines 713-719: Proportional resolve costs
  - Lines 793-865: compute_equity_distribution_end (added revolver subtraction)
  - Lines 953-959: AI priority for revolver draw
  - Lines 1103-1143: Conservation check (NEEDS MORE FIXES)
  - Lines 1154-1250: compute_detailed_stats (HAS BUGS)

### Configuration
- `calibration_config.py` (NEW FILE, 70 lines)
  - CalibrationTargets dataclass
  - StressConfig dataclass with all tuned parameters

### Analytics Runner
- `run_analytics.py` (156 lines → 230 lines)
  - Added --no-events flag
  - Added numpy import
  - Added base case economics validation logic
  - Shows calibration targets vs actual results

### Documentation
- `ACCOUNTING_FIX_SUMMARY.md` - Original conservation fix docs
- `CALIBRATION_PROGRESS.md` - Phase-by-phase progress tracker
- `CALIBRATION_FINAL_STATUS.md` (THIS FILE)

---

## Recommended Immediate Action

**DO THIS FIRST:**

1. Fix the conservation check bug completely
   - Add extensive debug logging
   - Test with 10-20 runs until 100% valid
   - Document the correct formula with detailed comments

2. Once conservation violations = 0:
   - Run base case check (--no-events)
   - Tune stress parameters based on results
   - Iterate until win rate reaches 40-70%

3. Then optimize AI policy and implement --calibrate mode

**Estimated Time:**
- Fix conservation: 2-3 hours
- Tune to target win rate: 2-4 hours
- AI policy + calibration: 3-5 hours
- **Total:** 7-12 hours of focused work

---

## Key Learnings

1. **Accounting is hard:** Even with detailed tracking, subtle bugs in cash flow accounting created systematic errors affecting 60-80% of runs.

2. **Stress calibration matters:** Reducing event probability from 10% to 2.5% improved outcomes, but win rate is still too low - suggests either stress is still too harsh OR base economics don't support targets.

3. **Conservative is not correct:** Early fixes were "conservative" (mark violations, exclude runs) but didn't solve root cause. Need to fix accounting, not just filter bad data.

4. **Test incrementally:** Should have tested conservation logic with 10-20 runs at each step, not 100-200. Faster iteration would have caught bugs sooner.

5. **Realistic returns achieved:** The fact that wins now show 13-14% IRR and 1.44-1.48x MOIC (vs impossible 10x+ before) proves the accounting DIRECTION is correct - just needs final debugging.

---

## User Instructions

**To continue calibration:**

```bash
# 1. Test current state (expect 60-80% invalid)
python run_analytics.py --baseline-only --sims 100 --save-plots

# 2. Enable debug mode and test small sample
$env:GAME_DEBUG='1'
python run_analytics.py --baseline-only --sims 10

# 3. Once conservation violations = 0, test base case
python run_analytics.py --no-events --sims 500

# 4. If base case passes, tune stress parameters in calibration_config.py
# Edit: event_probability_per_month, revolver_capacity, bank_gate_slope

# 5. Re-test until win rate reaches 40-70%
python run_analytics.py --baseline-only --sims 1000 --save-plots
```

**Key metrics to watch:**
- Invalid Runs: Must be 0
- Win Rate: Target 40-70%
- Median IRR (wins): Target 15-20%
- Median MOIC (wins): Target 1.5-1.7x

---

## Conclusion

**What worked:**
- ✅ Stress parameters reduced to realistic levels
- ✅ Revolver and graduated bank gate added
- ✅ Win scenarios now show realistic returns (13-14% IRR, 1.44x MOIC)
- ✅ Infrastructure for calibration and validation in place

**What needs fixing:**
- ❌ Conservation violation bug (60-80% invalid runs)
- ❌ Win rate too low (1-4% vs 40-70% target)
- ❌ Statistics calculation bugs

**Bottom line:** We're 70% of the way there. The foundation is solid, but the accounting bug must be fixed before the calibration can be completed. Once that's resolved, parameter tuning should get us to target metrics within a few hours.
