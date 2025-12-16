# Liquidity Crunch - Deployment Guide

## 📁 Project Structure

```
Code/
├── engine.py                    # ✅ Pure game logic (NO pygame, NO UI)
├── liquiditycrunchapp.py        # ✅ Streamlit web app (CLOUD DEPLOYMENT)
├── python liquidity_crunch.py   # 🖥️ Desktop pygame version (NOT DEPLOYED)
├── run_analytics.py             # 📊 Monte Carlo CLI (uses engine.py)
├── calibration_config.py        # ⚙️ Game constants
└── requirements.txt             # 📦 Dependencies (Streamlit Cloud compatible)
```

## 🎯 Architecture Overview

### Separation of Concerns

**engine.py** - Pure Logic Layer
- ✅ Zero UI dependencies
- ✅ Deterministic simulation
- ✅ Headless execution
- ✅ Used by BOTH desktop and web versions

**liquiditycrunchapp.py** - Streamlit UI Layer
- ✅ Cloud-deployable web interface
- ✅ Imports logic from engine.py
- ✅ No pygame dependencies
- ✅ Clean, responsive UI

**python liquidity_crunch.py** - Desktop UI Layer (Legacy)
- 🖥️ pygame-based desktop interface
- 🖥️ Imports logic from engine.py
- 🖥️ NOT used in cloud deployment
- 🖥️ Local development/testing only

## 🚀 Local Testing

### Test Streamlit App Locally

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Run Streamlit
streamlit run liquiditycrunchapp.py
```

Should open browser at: `http://localhost:8501`

### Test Engine Directly

```python
# Test in Python REPL
from engine import new_game, step_month, get_results

# Create game
eng = new_game(seed=42)

# Play a few months
for i in range(5):
    step_month(eng)
    print(f"Month {eng.gs.month}: Cash = €{eng.gs.cash/1e6:.1f}M")

# Get results
if eng.gs.game_over:
    results = get_results(eng)
    print(f"Win: {results['win']}, IRR: {results['irr']*100:.1f}%")
```

## 📦 Dependencies

### Production (requirements.txt)
```
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scipy>=1.11.0
```

### NOT Included (Desktop Only)
- ❌ pygame-ce (desktop only)
- ❌ OS-specific packages

## 🌐 Streamlit Cloud Deployment

### Step 1: Push to GitHub

```bash
git init
git add engine.py liquiditycrunchapp.py requirements.txt calibration_config.py
git commit -m "Add Liquidity Crunch web app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/liquidity-crunch.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repo: `YOUR_USERNAME/liquidity-crunch`
4. Set main file: `liquiditycrunchapp.py`
5. Click "Deploy"

### Step 3: Verify Deployment

- ✅ App loads without errors
- ✅ Can click "Next Month" and advance game
- ✅ Can raise debt, inject equity
- ✅ Monte Carlo runs successfully
- ✅ Results display correctly

## 🎮 Engine API Reference

### Core Functions

```python
from engine import new_game, step_month, is_finished, get_results, run_monte_carlo

# Create new game
engine = new_game(seed=42)

# Advance with action
engine = step_month(engine, action='raise_debt')
# Actions: 'raise_debt', 'inject_equity', 'sales_push', 'slow_build', 'draw_revolver'

# Check if done
if is_finished(engine):
    results = get_results(engine)
    print(results['win'], results['irr'], results['moic'])

# Run Monte Carlo
mc_results = run_monte_carlo(n=1000)
print(f"Win rate: {mc_results['win_rate']*100:.1f}%")
```

### Game State Access

```python
gs = engine.gs  # GameState object

# Key properties
gs.month          # Current month
gs.cash           # Current cash
gs.debt           # Debt outstanding
gs.units_sold     # Units sold
gs.progress       # Construction progress (0-100%)
gs.bank_gate_met  # Bank financing unlocked?
gs.active_events  # List of EventCard objects
gs.stresses       # StressRunMetrics object
```

## 🧪 Testing Checklist

### Local Testing (Before Deploy)

- [ ] `python -m py_compile engine.py` - No syntax errors
- [ ] `python -m py_compile liquiditycrunchapp.py` - No syntax errors
- [ ] `streamlit run liquiditycrunchapp.py` - Launches locally
- [ ] Play a game to completion
- [ ] Run Monte Carlo with 100 sims
- [ ] Check for any console errors

### Cloud Testing (After Deploy)

- [ ] App loads at Streamlit Cloud URL
- [ ] All buttons work (Next Month, Raise Debt, etc.)
- [ ] Game progresses without errors
- [ ] Can complete a full game (win or lose)
- [ ] Monte Carlo runs without timeout
- [ ] Charts render correctly
- [ ] No 404 or module import errors

## 🔧 Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'engine'`

**Solution:** Ensure `engine.py` is in the same directory as `liquiditycrunchapp.py`

### Monte Carlo Timeout

**Problem:** Monte Carlo with 2000 sims times out on Streamlit Cloud

**Solution:** Reduce default slider max to 1000 or add caching:

```python
@st.cache_data
def run_cached_mc(n, seed):
    return run_monte_carlo(n)
```

### Memory Issues

**Problem:** App crashes with large Monte Carlo runs

**Solution:** Streamlit Cloud has 1GB RAM limit. Keep simulations under 1000 runs.

## 📊 Feature Comparison

| Feature | Desktop (pygame) | Web (Streamlit) |
|---------|------------------|-----------------|
| Game Loop | ✅ Real-time | ✅ Turn-based |
| Graphics | ✅ Custom pygame | ✅ Streamlit widgets |
| Monte Carlo | ✅ Full charts | ✅ Basic charts |
| Deployment | ❌ Local only | ✅ Cloud hosted |
| Multiplayer | ❌ No | ❌ No |
| Mobile | ❌ No | ✅ Responsive |

## 🎯 Next Steps

1. ✅ Test locally: `streamlit run liquiditycrunchapp.py`
2. ✅ Push to GitHub
3. ✅ Deploy to Streamlit Cloud
4. ✅ Share URL with users
5. 📈 Collect feedback
6. 🚀 Iterate and improve

## 📝 Notes

- **Engine is fully deterministic**: Same seed = same outcome
- **No pygame in requirements.txt**: Cloud deployment is headless
- **Desktop version still works**: Uses same engine.py logic
- **Analytics CLI unchanged**: `run_analytics.py` still functional

## 🆘 Support

For issues, check:
1. Streamlit Cloud logs (hamburger menu → "Manage app" → "Logs")
2. Local console output when running `streamlit run`
3. GitHub repo for version mismatches

---

**Status:** ✅ Ready for deployment

**Last Updated:** December 16, 2025
