from engine import run_monte_carlo

print("Running 100 simulations...")
mc = run_monte_carlo(100)

print(f"\nResults:")
print(f"Win rate: {mc['win_rate']*100:.1f}%")
print(f"Total simulations: {len(mc['results'])}")

results = mc['results']
wins = sum(1 for r in results if r.get('win', False))
losses = sum(1 for r in results if not r.get('win', False))

print(f"Wins: {wins}")
print(f"Losses: {losses}")

# Show some failure reasons
if losses > 0:
    print("\nFailure reasons (first 10):")
    for i, r in enumerate(results[:10]):
        if not r.get('win', False):
            print(f"  Run {i}: {r.get('failure_reason', 'unknown')}")
