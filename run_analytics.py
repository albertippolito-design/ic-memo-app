"""Run sensitivity analysis from command line"""
import sys
sys.path.insert(0, '.')
import os
import numpy as np

from importlib import import_module

# Set matplotlib backend before importing game module
import matplotlib
if '--show-charts' not in sys.argv:
    # Use non-GUI backend for headless operation
    matplotlib.use('Agg')
else:
    # Use TkAgg for interactive display
    try:
        matplotlib.use('TkAgg')
    except:
        matplotlib.use('Agg')
        print("Warning: TkAgg backend not available, using Agg (save-only mode)")

game = import_module('python liquidity_crunch')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Monte Carlo simulation with analytics')
    parser.add_argument('--sims', type=int, default=1000, help='Number of simulations per scenario (default: 1000)')
    parser.add_argument('--sensitivity', action='store_true', help='Run full sensitivity analysis')
    parser.add_argument('--baseline-only', action='store_true', help='Run baseline scenario only')
    parser.add_argument('--no-events', action='store_true', help='Run base case with no random events (validate base economics)')
    parser.add_argument('--single-run', action='store_true', help='Run single simulation with detailed dump')
    parser.add_argument('--seed', type=int, default=123, help='Seed for single-run mode (default: 123)')
    parser.add_argument('--dump', action='store_true', help='Print detailed sources & uses reconciliation')
    parser.add_argument('--show-charts', action='store_true', help='Show matplotlib charts (requires GUI)')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to PNG files instead of displaying')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory for saved plots (default: output)')
    
    args = parser.parse_args()
    
    # Create output directory if saving plots
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Plots will be saved to: {os.path.abspath(args.output_dir)}")
    
    if args.sensitivity:
        # Run full sensitivity comparison
        comparison_results = game.run_sensitivity_comparison(n=args.sims)
        
        if args.show_charts or args.save_plots:
            # Show/save charts for each scenario
            for i, result in enumerate(comparison_results):
                scenario_name = result['name'].replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
                if args.save_plots:
                    save_path = os.path.join(args.output_dir, f"scenario_{i+1}_{scenario_name}.png")
                    print(f"\nGenerating chart for: {result['name']}")
                    game.show_simulation_results(result['results'], save_path=save_path)
                else:
                    print(f"\nShowing charts for: {result['name']}")
                    game.show_simulation_results(result['results'])
    
    elif args.no_events:
        # Run base case with no events (validate base economics)
        print(f"\n{'='*80}")
        print("BASE CASE ECONOMICS CHECK (No Random Events)")
        print(f"{'='*80}")
        print(f"Running {args.sims} simulations with events disabled...")
        print("This validates that base economics can support 15-20% IRR targets.\n")
        
        results = []
        for i in range(args.sims):
            if i % 100 == 0:
                print(f"Progress: {i}/{args.sims}")
            settings = game.SimulationSettings(events_enabled=False)
            result = game.run_one_simulation(i, settings=settings)
            results.append(result)
        
        stats = game.compute_detailed_stats(results)
        
        # Calculate win-specific stats
        valid_wins = [r for r in results if r.get('valid_run', True) and r['win']]
        if valid_wins:
            win_irrs = [r['irr'] for r in valid_wins if r['irr_status'] == 'valid']
            win_moics = [r['moic'] for r in valid_wins]
            
            print(f"\n{'='*80}")
            print("BASE CASE RESULTS (No Events)")
            print(f"{'='*80}")
            print(f"Valid Runs: {stats['valid_runs']}")
            print(f"Invalid Runs: {stats['invalid_runs']}")
            print(f"Win Rate: {stats['win_rate']:.1f}%")
            
            if win_irrs:
                print(f"\nIRR Distribution (Wins Only, {len(win_irrs)} valid):")
                print(f"  P25: {np.percentile(win_irrs, 25)*100:.1f}%")
                print(f"  P50 (Median): {np.percentile(win_irrs, 50)*100:.1f}%")
                print(f"  P75: {np.percentile(win_irrs, 75)*100:.1f}%")
                print(f"  Mean: {np.mean(win_irrs)*100:.1f}%")
            
            if win_moics:
                print(f"\nMOIC Distribution (Wins, {len(win_moics)} runs):")
                print(f"  P25: {np.percentile(win_moics, 25):.2f}x")
                print(f"  P50 (Median): {np.percentile(win_moics, 50):.2f}x")
                print(f"  P75: {np.percentile(win_moics, 75):.2f}x")
                print(f"  Mean: {np.mean(win_moics):.2f}x")
            
            print(f"\n{'='*80}")
            print("BASE CASE SANITY CHECK:")
            print(f"{'='*80}")
            if stats['win_rate'] > 70 and win_irrs and np.median(win_irrs) > 0.15:
                print("✓ PASS: Base economics support 15%+ IRR with 70%+ win rate")
                print("  → Calibration issues are likely from event/stress tuning")
            else:
                print("✗ FAIL: Base economics cannot support targets")
                print("  → Need to adjust: GDV, costs, timing, or debt structure")
                if stats['win_rate'] <= 70:
                    print(f"  → Win rate too low: {stats['win_rate']:.1f}% (need >70%)")
                if win_irrs and np.median(win_irrs) <= 0.15:
                    print(f"  → IRR too low: {np.median(win_irrs)*100:.1f}% (need >15%)")
        
        if args.show_charts or args.save_plots:
            if args.save_plots:
                save_path = os.path.join(args.output_dir, "no_events_results.png")
                print(f"\nGenerating no-events chart...")
                game.show_simulation_results(results, save_path=save_path)
            else:
                game.show_simulation_results(results)
    
    elif args.baseline_only:
        # Run baseline only
        print(f"\nRunning {args.sims} baseline simulations...")
        results = []
        for i in range(args.sims):
            if i % 100 == 0:
                print(f"Progress: {i}/{args.sims}")
            result = game.run_one_simulation(i, settings=game.SimulationSettings.baseline())
            results.append(result)
        
        stats = game.compute_detailed_stats(results)
        print(f"\n{'='*80}")
        print("BASELINE RESULTS")
        print(f"{'='*80}")
        print(f"Total Runs: {stats['n']}")
        print(f"Win Rate: {stats['win_rate']:.1f}%")
        
        # Handle case where there are no valid IRR/MOIC results
        if stats['irr_stats'] and stats['irr_stats']['median'] is not None:
            print(f"Median IRR: {stats['irr_stats']['median']*100:.1f}%")
        else:
            print(f"Median IRR: N/A (no valid wins)")
            
        if stats['moic_stats'] and stats['moic_stats']['median'] is not None:
            print(f"Median MOIC: {stats['moic_stats']['median']:.2f}x")
        else:
            print(f"Median MOIC: N/A (no valid wins)")
        
        print(f"IRR > 0%: {stats['irr_above_0_pct']:.1f}%")
        print(f"IRR > 10%: {stats['irr_above_10_pct']:.1f}%")
        print(f"IRR > 15%: {stats['irr_above_15_pct']:.1f}%")
        print(f"\nTop Failure Reasons:")
        for reason, count in sorted(stats['failure_reasons'].items(), key=lambda x: x[1], reverse=True)[:5]:
            pct = count / stats['losses'] * 100 if stats['losses'] > 0 else 0
            print(f"  {reason}: {count} ({pct:.1f}%)")
        
        if args.show_charts or args.save_plots:
            if args.save_plots:
                save_path = os.path.join(args.output_dir, "baseline_simulation_results.png")
                print(f"\nGenerating baseline chart...")
                game.show_simulation_results(results, save_path=save_path)
            else:
                game.show_simulation_results(results)
    
    elif args.single_run:
        # Run single simulation with detailed dump
        print(f"\n{'='*80}")
        print(f"SINGLE RUN DEBUG DUMP (Seed: {args.seed})")
        print(f"{'='*80}\n")
        
        result = game.run_one_simulation(args.seed, settings=game.SimulationSettings.baseline())
        
        # Get the engine state from the result
        # We need to re-run to get the engine object
        import random
        random.seed(args.seed)
        eng = game.Engine(settings=game.SimulationSettings.baseline())
        
        # Run simulation
        while not eng.gs.game_over and eng.gs.month < 60:
            eng.tick_month()
            eng.ai_decide_action()
        
        gs = eng.gs
        
        # Print comprehensive stats
        print(f"OUTCOME: {'WIN' if gs.win else 'LOSS'}")
        print(f"Duration: {gs.month} months")
        print(f"Units Sold: {gs.units_sold} / {gs.units_total} ({gs.units_sold/gs.units_total*100:.1f}%)")
        print(f"Final Cash: €{gs.cash:,.0f}")
        print(f"Final Progress: {gs.progress:.1f}%")
        
        print(f"\n{'='*80}")
        print("SOURCES & USES RECONCILIATION")
        print(f"{'='*80}")
        
        sources, uses, delta = eng.reconcile_sources_uses()
        
        print("\nSOURCES (Cash In):")
        total_sources = 0
        for key, val in sources.items():
            print(f"  {key:25s}: €{val:15,.0f}")
            total_sources += val
        print(f"  {'TOTAL SOURCES':25s}: €{total_sources:15,.0f}")
        
        print("\nUSES (Cash Out):")
        total_uses = 0
        for key, val in uses.items():
            print(f"  {key:25s}: €{val:15,.0f}")
            total_uses += val
        print(f"  {'TOTAL USES':25s}: €{total_uses:15,.0f}")
        
        print(f"\nENDING CASH: €{gs.cash:,.0f}")
        print(f"EXPECTED CASH (Sources - Uses): €{total_sources - total_uses:,.0f}")
        print(f"\nRECONCILIATION DELTA: €{delta:,.0f}")
        
        if abs(delta) > 1000:
            print(f"\n{'*'*80}")
            print(f"*** RECONCILIATION FAILED: Delta = €{delta:,.0f} ***")
            print(f"{'*'*80}")
        else:
            print(f"\n✓ Reconciliation passed (delta < €1k)")
        
        # Check deposits
        print(f"\nDEPOSIT STATUS:")
        print(f"  Deposits collected (total): €{gs.deposits_collected_total:,.0f}")
        print(f"  Deposits liability (end): €{gs.deposits_liability:,.0f}")
        if gs.units_sold == gs.units_total and abs(gs.deposits_liability) > 1000:
            print(f"  *** WARNING: All units sold but deposits_liability ≠ 0 ***")
        
        # Check debt
        print(f"\nDEBT STATUS:")
        print(f"  Debt drawn (total): €{gs.debt_drawn_total:,.0f}")
        print(f"  Debt repaid (total): €{gs.debt_principal_repaid_total:,.0f}")
        print(f"  Debt outstanding (end): €{gs.debt:,.0f}")
        print(f"  Net debt proceeds: €{gs.debt_drawn_total - gs.debt_principal_repaid_total:,.0f}")
        
        # MOIC calc
        equity_out = eng.compute_equity_distribution_end()
        moic = equity_out / gs.equity_in_total if gs.equity_in_total > 0 else 0
        print(f"\nEQUITY RETURNS:")
        print(f"  Equity in (total): €{gs.equity_in_total:,.0f}")
        print(f"  Equity out (net): €{equity_out:,.0f}")
        print(f"  MOIC: {moic:.2f}x")
        print(f"  IRR: {result.get('irr', 0)*100:.1f}% ({result.get('irr_status', 'unknown')})")
        
        # Stress analysis for this run
        stresses = gs.stresses
        print(f"\n{'='*80}")
        print("STRESSES ENCOUNTERED IN THIS RUN")
        print(f"{'='*80}")
        if stresses.had_delay:
            print(f"  ⚠ Construction Delay: {stresses.delay_months} months")
        if stresses.had_capex_overrun:
            print(f"  ⚠ Capex Overrun: {stresses.capex_overrun_pct:.1f}% (€{stresses.capex_overrun_amount:,.0f})")
        if stresses.had_rate_shock:
            print(f"  ⚠ Rate Shock: {stresses.max_rate_bps} bps for {stresses.rate_shock_months} months")
        if stresses.had_sales_shock:
            print(f"  ⚠ Sales Shock: {stresses.min_sales_multiplier*100:.1f}% velocity for {stresses.sales_shock_months} months")
        if stresses.had_event_cost:
            print(f"  ⚠ Event Costs: €{stresses.total_event_cost:,.0f} from {stresses.event_count} events")
        if stresses.bank_gate_blocked_months > 0:
            print(f"  ⚠ Bank Gate Blocked: {stresses.bank_gate_blocked_months} months")
        
        if not any([stresses.had_delay, stresses.had_capex_overrun, stresses.had_rate_shock, 
                    stresses.had_sales_shock, stresses.had_event_cost, stresses.bank_gate_blocked_months > 0]):
            print("  ✓ No stresses encountered (clean run)")
        
        print(f"\n{'='*80}\n")
    
    else:
        print("Please specify --sensitivity or --baseline-only")
        parser.print_help()
