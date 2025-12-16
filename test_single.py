from engine import run_one_simulation

# Run a single simulation with detailed output
result = run_one_simulation(seed=42)

print("=== SINGLE SIMULATION RESULT (seed=42) ===\n")
print(f"Win: {result['win']}")
print(f"Failure reason: {result['failure_reason']}")
print(f"Duration: {result['duration']} months")
print(f"Final cash: €{result['final_cash']:,.0f}")
print(f"Final debt: €{result['final_debt']:,.0f}")
print(f"Units sold: {result['units_sold']}/{1000}")
print(f"Progress: {result['progress']:.1f}%")
print(f"Cash trough: €{result['cash_trough']:,.0f}")
print(f"IRR: {result['irr']*100:.1f}% ({result['irr_status']})")
print(f"MOIC: {result['moic']:.2f}x")
print(f"Equity in: €{result['equity_in']:,.0f}")
print(f"Equity out: €{result['equity_out']:,.0f}")
