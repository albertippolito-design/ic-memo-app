import sys
sys.path.insert(0, '.')
game = __import__('python liquidity_crunch')

# Run ONE simulation with seed 42
result = game.run_one_simulation(seed=42)

print("=== PYGAME VERSION RESULT (seed=42) ===\n")
print(f"Win: {result['win']}")
print(f"Failure reason: {result.get('failure_reason')}")
print(f"Duration: {result['duration']} months")
print(f"Final cash: €{result['final_cash']:,.0f}")
print(f"Units sold: {result['units_sold']}")
print(f"IRR: {result['irr']*100:.1f}%")
print(f"MOIC: {result['moic']:.2f}x")
