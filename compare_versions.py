"""Compare pygame vs engine.py execution step-by-step"""
import sys
sys.path.insert(0, '.')
import random

# Import both versions
pygame_game = __import__('python liquidity_crunch')
from engine import new_game, step_month

SEED = 42

# Run pygame version
print("=" * 80)
print("PYGAME VERSION")
print("=" * 80)
random.seed(SEED)
pygame_eng = pygame_game.Engine(settings=pygame_game.SimulationSettings.baseline())
pygame_eng._seed = SEED

print(f"\nMonth 0 (after land):")
print(f"  Cash: €{pygame_eng.gs.cash:,.0f}")
print(f"  Equity remaining: €{pygame_eng.gs.equity_remaining:,.0f}")
print(f"  Equity used: €{pygame_eng.gs.equity_used:,.0f}")

for i in range(1, 6):
    if pygame_eng.gs.game_over:
        break
    
    cash_before = pygame_eng.gs.cash
    pygame_eng.ai_decide_action()
    pygame_eng.tick_month()
    
    print(f"\nMonth {i}:")
    print(f"  Cash: €{pygame_eng.gs.cash:,.0f} (Δ={cash_before - pygame_eng.gs.cash:,.0f})")
    print(f"  Units sold: {pygame_eng.gs.units_sold}")
    print(f"  Deposits: €{pygame_eng.gs.deposits_collected:,.0f}")
    print(f"  Progress: {pygame_eng.gs.progress:.1f}%")
    print(f"  Equity used: €{pygame_eng.gs.equity_used:,.0f}")
    print(f"  Equity remaining: €{pygame_eng.gs.equity_remaining:,.0f}")
    print(f"  Revolver: €{pygame_eng.gs.revolver_drawn:,.0f}")
    print(f"  Last net CF: €{pygame_eng.gs.last_net_cf:,.0f}")

print("\n" + "=" * 80)
print("ENGINE.PY VERSION")
print("=" * 80)
random.seed(SEED)
engine_eng = new_game(seed=SEED)

print(f"\nMonth 0 (after land):")
print(f"  Cash: €{engine_eng.gs.cash:,.0f}")
print(f"  Equity remaining: €{engine_eng.gs.equity_remaining:,.0f}")
print(f"  Equity used: €{engine_eng.gs.equity_used:,.0f}")

for i in range(1, 6):
    if engine_eng.gs.game_over:
        break
    
    cash_before = engine_eng.gs.cash
    engine_eng.ai_decide_action()
    engine_eng.tick_month()
    
    print(f"\nMonth {i}:")
    print(f"  Cash: €{engine_eng.gs.cash:,.0f} (Δ={cash_before - engine_eng.gs.cash:,.0f})")
    print(f"  Units sold: {engine_eng.gs.units_sold}")
    print(f"  Deposits: €{engine_eng.gs.deposits_collected:,.0f}")
    print(f"  Progress: {engine_eng.gs.progress:.1f}%")
    print(f"  Equity used: €{engine_eng.gs.equity_used:,.0f}")
    print(f"  Equity remaining: €{engine_eng.gs.equity_remaining:,.0f}")
    print(f"  Revolver: €{engine_eng.gs.revolver_drawn:,.0f}")
    print(f"  Last net CF: €{engine_eng.gs.last_net_cf:,.0f}")

print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)
print(f"Pygame - Month {pygame_eng.gs.month}: Cash=€{pygame_eng.gs.cash:,.0f}, Win={not pygame_eng.gs.game_over or pygame_eng.gs.win}")
print(f"Engine - Month {engine_eng.gs.month}: Cash=€{engine_eng.gs.cash:,.0f}, Win={not engine_eng.gs.game_over or engine_eng.gs.win}")
