from engine import run_monte_carlo

print("Testing engine.py Monte Carlo...")
mc = run_monte_carlo(1000)

print(f"\n=== ENGINE.PY RESULTS (1000 sims) ===")
print(f"Win Rate: {mc['win_rate']*100:.1f}%")
print(f"Median IRR: {mc['median_irr']*100:.1f}%")
print(f"Median MOIC: {mc['median_moic']:.2f}x")

# Count failures
results = mc['results']
valid_results = [r for r in results if r['valid_run']]
losses = [r for r in valid_results if not r['win']]

from collections import Counter
failure_reasons = Counter([r['failure_reason'] for r in losses if r['failure_reason']])

print(f"\nTop Failure Reasons:")
for reason, count in failure_reasons.most_common(5):
    pct = count / len(losses) * 100 if losses else 0
    print(f"  {reason}: {count} ({pct:.1f}%)")
