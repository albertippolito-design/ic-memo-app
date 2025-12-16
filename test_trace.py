from engine import new_game, step_month
import random

random.seed(42)
eng = new_game(seed=42)

print("=== MONTH-BY-MONTH TRACE (seed=42) ===\n")
print(f"Month 0 (after land purchase):")
print(f"  Cash: €{eng.gs.cash:,.0f}")
print(f"  Debt: €{eng.gs.debt:,.0f}")
print(f"  Progress: {eng.gs.progress:.1f}%")
print(f"  Units sold: {eng.gs.units_sold}")
print(f"  Bank gate: {eng.gs.bank_gate_met}")
print()

for i in range(1, 10):
    if eng.gs.game_over:
        print(f"GAME OVER: {eng.gs.reason}")
        break
    
    cash_before = eng.gs.cash
    
    # AI decides and executes action
    eng.ai_decide_action()
    
    # Advance month
    eng.tick_month()
    
    cash_after = eng.gs.cash
    burn = cash_before - cash_after
    
    print(f"Month {i}:")
    print(f"  Cash: €{cash_after:,.0f} (Δ={burn:,.0f})")
    print(f"  Last net CF: €{eng.gs.last_net_cf:,.0f}")
    print(f"  Debt: €{eng.gs.debt:,.0f}")
    print(f"  Equity used: €{eng.gs.equity_used:,.0f} / Remaining: €{eng.gs.equity_remaining:,.0f}")
    print(f"  Progress: {eng.gs.progress:.1f}%")
    print(f"  Units sold: {eng.gs.units_sold} (new: {len([u for u in eng.gs.unit_sales if u.sale_month == i])})")
    print(f"  Deposits collected: €{eng.gs.deposits_collected:,.0f}")
    print(f"  Bank gate: {eng.gs.bank_gate_met}")
    print(f"  Revolver: €{eng.gs.revolver_drawn:,.0f}")
    print(f"  Reason: {eng.gs.reason}")
    print()
