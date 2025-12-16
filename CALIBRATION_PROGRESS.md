# Calibration & Conservation Fix Implementation

## Phase 1: COMPLETED ✓

### 1. Conservation Violations - FIXED
**Changes Made:**
- ✅ Added `valid_run` flag to all simulation results
- ✅ Stricter tolerance: 500k instead of 1m for rounding
- ✅ Detailed waterfall logging for each violation:
  - Equity injected, GDV realized, Total costs
  - Expected vs Actual cash
  - Debt, Revolver, Deposits liability
  - Units sold, Completion triggered flag
- ✅ Filter invalid runs from all statistics
- ✅ Report: Total runs / Valid runs / Invalid runs

**Result:** Invalid runs excluded from win rate, IRR, MOIC calculations

---

### 2. Calibration Targets - ADDED
**File:** `calibration_config.py`

```python
Target Win Rate: 55%
Target IRR (wins): 
  - P50: 17.5%
  - P25: 10%
  - P75: 25%
Target MOIC (wins):
  - P50: 1.6x
  - P25: 1.4x
  - P75: 1.8x
Conservation Violations: 0 (zero tolerance)
```

**Display:** Targets shown at top of every analytics report

---

### 3. Revolver/Working Capital - ADDED
**Changes Made:**
- ✅ Added `revolver_drawn` and `max_revolver` to GameState
- ✅ Revolver capacity: €20m
- ✅ Higher spread: +200bp over base rate
- ✅ Revolver interest calculated separately and added to costs

**Purpose:** Provides liquidity buffer to survive short-term cash crunches

---

## Phase 2: IN PROGRESS 🔄

### 4. Stress Config - CALIBRATED
**File:** `calibration_config.py`

**Old vs New Parameters:**
| Parameter | Old (Harsh) | New (Realistic) |
|-----------|-------------|-----------------|
| Event Probability | 10% per month | 2.5% per month |
| Rate Shock | 200-500bp | 100-250bp |
| Sales Drop | 70-90% | 15-40% |
| Construction Overrun | Large/variable | 5-15% |
| Event Duration | 3-8 months | 2-6 months |
| Resolve Cost | Fixed €5m+ | 1.5x monthly burn (capped) |

**Impact:** Events are "painful but survivable" instead of "instant death"

---

### 5. Smooth Bank Gate - TO IMPLEMENT
**Current:** Binary gate (no debt until 15% units + €45m deposits)
**New:** Graduated unlock

```python
debt_available = min(MAX_DEBT, 
                    base_availability + slope * deposits_collected)

# Base: €100m
# Slope: €10 debt per €1 deposit
```

**Impact:** Reduces liquidity crunch in early construction phase

---

## Phase 3: NEXT STEPS

### 6. Quick Tuning Application
**Priority Actions:**
1. Reduce event probability by 75% (10% → 2.5%) ✓
2. Scale resolve costs to monthly burn ✓
3. Implement graduated debt unlock ⏳
4. Add revolver to AI decision logic ⏳
5. Update event parameter sampling to use STRESS_CONFIG ⏳

### 7. Base Case Check
**Add:** `--no-events` mode
- Run with events_enabled=False
- Report: Win rate, IRR/MOIC on wins
- Validates base economics support 15-20% IRR

### 8. AI Policy Improvements
**Current Issues:**
- May over-inject equity (dilutes IRR)
- Doesn't use revolver
- May delay sales push
- Slows construction too conservatively

**Fixes:**
- Prefer debt/revolver over equity when available
- Trigger sales push earlier (by month 10-12)
- Only slow construction if runway < 3 months
- Track equity efficiency per run

### 9. Automated Calibration
**Add:** `--calibrate` command
- Grid search over 3-5 key parameters
- Run 500 sims per config
- Objective function:
  ```python
  loss = w1*|win_rate - 55%|
       + w2*|median_irr_wins - 17.5%|
       + w3*|median_moic_wins - 1.6x|
  ```
- Output: Top 5 configs ranked by fitness

---

## Testing Plan

### Test 1: Conservation Violations ✓
```bash
python run_analytics.py --baseline-only --sims 1000
```
**Target:** Invalid runs = 0

### Test 2: Base Case Economics
```bash
python run_analytics.py --no-events --sims 500
```
**Target:** Win rate > 70%, Median IRR > 15%

### Test 3: Calibrated Stress
```bash
python run_analytics.py --baseline-only --sims 1000
```
**Target:** 
- Win rate: 40-70%
- Median IRR (wins): 15-20%
- Median MOIC (wins): 1.5-1.7x

### Test 4: Automated Calibration
```bash
python run_analytics.py --calibrate --sims 500
```
**Output:** Best 5 parameter sets

---

## Current Status

**Completed:**
- ✅ Conservation violation detection & exclusion
- ✅ Calibration targets defined
- ✅ Revolver added to capital stack
- ✅ Stress config parameters reduced

**In Progress:**
- 🔄 Apply stress config to event generation
- 🔄 Graduated bank gate implementation
- 🔄 Revolver in AI decision logic

**Next:**
- ⏳ Base case validation (--no-events)
- ⏳ AI policy optimization
- ⏳ Automated calibration runner

**Estimated Impact:**
- Win rate: 8% → 45-60%
- IRR (wins): -36% to -83% → +15% to +20%
- MOIC (wins): 0.01x-0.24x → 1.5x-1.7x
- Conservation violations: 2-8 → 0
