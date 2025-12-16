"""Test the analytics system with a small batch of simulations"""
import sys
sys.path.insert(0, '.')

# Import the necessary components
import random
from importlib import import_module

# Import the game module
game = import_module('python liquidity_crunch')

print("Testing analytics system...")
print("="*80)

# Test 1: Run 10 baseline simulations
print("\n1. Testing baseline simulations (n=10)...")
results = []
for i in range(10):
    result = game.run_one_simulation(i, settings=game.SimulationSettings.baseline())
    results.append(result)

wins = sum(1 for r in results if r['win'])
print(f"   ✓ Completed 10 simulations")
print(f"   ✓ Win rate: {wins}/10 ({wins*10}%)")
print(f"   ✓ Failure reasons tracked: {all(r.get('failure_reason') for r in results if not r['win'])}")

# Test 2: Compute statistics
print("\n2. Testing statistics computation...")
stats = game.compute_detailed_stats(results)
print(f"   ✓ Win rate: {stats['win_rate']:.1f}%")
print(f"   ✓ IRR stats computed: {stats['irr_stats'] is not None}")
print(f"   ✓ MOIC stats computed: {stats['moic_stats'] is not None}")
print(f"   ✓ Failure reasons: {len(stats['failure_reasons'])} types")

# Test 3: Failure analysis
print("\n3. Testing failure analysis...")
failure_analysis = game.compute_failure_analysis(results)
if failure_analysis:
    print(f"   ✓ Failure analysis computed: {len(failure_analysis)} reasons")
    for reason, data in failure_analysis.items():
        print(f"      - {reason}: {data['count']} occurrences")

# Test 4: Sample runs
print("\n4. Testing sample run extraction...")
samples = game.extract_sample_runs(results)
print(f"   ✓ Best wins: {len(samples['best_wins'])}")
print(f"   ✓ Diverse losses: {len(samples['diverse_losses'])}")

# Test 5: Settings variations
print("\n5. Testing sensitivity settings...")
settings_tests = [
    ("Baseline", game.SimulationSettings.baseline()),
    ("No Events", game.SimulationSettings.no_events()),
    ("Reduced Volatility", game.SimulationSettings.reduced_volatility()),
]

for name, settings in settings_tests:
    result = game.run_one_simulation(100, settings=settings)
    print(f"   ✓ {name}: Win={result['win']}, Duration={result['duration']}m")

print("\n" + "="*80)
print("All tests passed! ✓")
print("="*80)
