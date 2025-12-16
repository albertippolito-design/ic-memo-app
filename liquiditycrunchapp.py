"""
Liquidity Crunch - Streamlit Web App
====================================
Cloud-deployed version of the greenfield property development simulation game.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from engine import (
    new_game,
    step_month,
    is_finished,
    get_results,
    run_monte_carlo,
    Engine,
    TOTAL_EQUITY,
    TOTAL_UNITS,
    LAND_COST,
    DEV_COSTS,
    GDV,
    CONSTRUCTION_MONTHS,
)

# ==================== Page Config ====================

st.set_page_config(
    page_title="Liquidity Crunch",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Helper Functions ====================

def money(x):
    """Format money with M/k suffix"""
    if abs(x) >= 1_000_000:
        return f"€{x/1_000_000:.1f}M"
    elif abs(x) >= 1_000:
        return f"€{x/1_000:.0f}k"
    return f"€{x:,.0f}"


def format_pct(x):
    """Format percentage"""
    return f"{x*100:.1f}%"


# ==================== Session State Initialization ====================

if "engine" not in st.session_state:
    st.session_state.engine = new_game(seed=42)
    st.session_state.action_log = []

if "mc_results" not in st.session_state:
    st.session_state.mc_results = None

engine = st.session_state.engine
gs = engine.gs

# ==================== Sidebar ====================

with st.sidebar:
    st.title("💧 Liquidity Crunch")
    st.markdown("*Property Development Simulation*")
    
    st.divider()
    
    st.subheader("📊 Project Overview")
    st.markdown(f"""
    - **GDV:** {money(GDV)} ({TOTAL_UNITS} units @ €1M)
    - **Total Equity:** {money(TOTAL_EQUITY)}
    - **Dev Costs:** {money(DEV_COSTS)}
    - **Construction:** {CONSTRUCTION_MONTHS} months
    - **Target:** Complete & sell 70%+ units
    """)
    
    st.divider()
    
    if st.button("🔄 New Game", use_container_width=True):
        st.session_state.engine = new_game()
        st.session_state.action_log = []
        st.session_state.mc_results = None
        st.rerun()
    
    st.divider()
    
    st.caption("Built with Streamlit | Engine: engine.py")

# ==================== Main Content ====================

st.title("💧 Liquidity Crunch – Greenfield Developer Game")

if is_finished(engine):
    results = get_results(engine)
    
    if results['win']:
        st.success("🎉 YOU WIN!", icon="🎉")
    else:
        st.error("💀 YOU LOSE", icon="💀")
    
    st.divider()
    
    # Results display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Duration", f"{results['month']} months")
        st.metric("Final Cash", money(results['cash']))
    
    with col2:
        st.metric("Units Sold", f"{results['units_sold']}/{TOTAL_UNITS}")
        st.metric("Progress", f"{results['progress']:.0f}%")
    
    with col3:
        st.metric("IRR", format_pct(results['irr']) if results['irr_status'] == 'valid' else "N/A")
        st.metric("MOIC", f"{results['moic']:.2f}x")
    
    with col4:
        st.metric("Equity In", money(results['equity_in']))
        st.metric("Equity Out", money(results['equity_out']))
    
    st.divider()
    
    st.subheader("📝 Game Summary")
    st.write(f"**Outcome:** {results['reason']}")
    if results['failure_reason']:
        st.write(f"**Failure Reason:** {results['failure_reason']}")
    
    # Stress summary
    stresses = results['stresses']
    if any([stresses.had_delay, stresses.had_capex_overrun, stresses.had_rate_shock,
            stresses.had_sales_shock, stresses.had_event_cost, stresses.bank_gate_blocked_months > 0]):
        st.subheader("⚠️ Stresses in This Run")
        stress_items = []
        if stresses.had_delay:
            stress_items.append(f"- Construction Delay: {stresses.delay_months} months")
        if stresses.had_capex_overrun:
            stress_items.append(f"- Capex Overrun: {stresses.capex_overrun_pct:.1f}% ({money(stresses.capex_overrun_amount)})")
        if stresses.had_rate_shock:
            stress_items.append(f"- Rate Shock: max +{stresses.max_rate_bps} bps for {stresses.rate_shock_months} months")
        if stresses.had_sales_shock:
            stress_items.append(f"- Sales Shock: {stresses.min_sales_multiplier*100:.1f}% velocity for {stresses.sales_shock_months} months")
        if stresses.had_event_cost:
            stress_items.append(f"- Event Costs: {money(stresses.total_event_cost)} ({stresses.event_count} events)")
        if stresses.bank_gate_blocked_months > 0:
            stress_items.append(f"- Bank Gate Blocked: {stresses.bank_gate_blocked_months} months")
        
        st.markdown("\n".join(stress_items))

else:
    # Game in progress
    st.subheader("🎮 Game Controls")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("▶️ Next Month", use_container_width=True):
            step_month(engine)
            st.session_state.action_log.append(f"Month {gs.month}: Advance")
            st.rerun()
    
    with col2:
        if st.button("💰 Raise Debt", use_container_width=True):
            step_month(engine, action='raise_debt')
            st.session_state.action_log.append(f"Month {gs.month}: Raise Debt")
            st.rerun()
    
    with col3:
        if st.button("💉 Inject Equity", use_container_width=True):
            step_month(engine, action='inject_equity')
            st.session_state.action_log.append(f"Month {gs.month}: Inject Equity")
            st.rerun()
    
    with col4:
        if st.button("📈 Sales Push", use_container_width=True):
            step_month(engine, action='sales_push')
            st.session_state.action_log.append(f"Month {gs.month}: Sales Push")
            st.rerun()
    
    with col5:
        if st.button("🐌 Slow Build", use_container_width=True):
            step_month(engine, action='slow_build')
            st.session_state.action_log.append(f"Month {gs.month}: Slow Build")
            st.rerun()
    
    st.divider()
    
    # KPI Dashboard
    st.subheader("📊 Project Status")
    
    k1, k2, k3, k4 = st.columns(4)
    
    with k1:
        st.metric("Month", gs.month)
        st.metric("Phase", gs.phase)
    
    with k2:
        cash_color = "normal" if gs.cash > 30_000_000 else "off"
        st.metric("Cash", money(gs.cash), delta=money(gs.last_net_cf) if gs.month > 0 else None)
        st.metric("Cash Trough", money(gs.cash_trough))
    
    with k3:
        st.metric("Debt Outstanding", money(gs.debt))
        st.metric("Debt Capacity", f"{gs.debt/gs.max_debt*100:.0f}%")
    
    with k4:
        st.metric("Units Sold", f"{gs.units_sold}/{gs.units_total}")
        st.metric("Deposits Collected", money(gs.deposits_collected))
    
    # Second row of KPIs
    k5, k6, k7, k8 = st.columns(4)
    
    with k5:
        st.metric("Construction", f"{gs.progress:.1f}%")
        progress_bar = st.progress(gs.progress / 100.0)
    
    with k6:
        st.metric("Equity Used", money(gs.equity_used))
        st.metric("Equity Total", money(gs.equity_total))
    
    with k7:
        rate = engine.current_rate()
        st.metric("Interest Rate", format_pct(rate))
        st.metric("Revolver Drawn", money(gs.revolver_drawn))
    
    with k8:
        st.metric("Bank Gate", "✅ MET" if gs.bank_gate_met else "🔒 LOCKED")
        st.metric("Active Events", len(gs.active_events))
    
    st.divider()
    
    # Status message
    if gs.reason:
        st.info(gs.reason)
    
    # Active events
    if gs.active_events:
        st.subheader("⚠️ Active Events")
        for ev in gs.active_events:
            st.warning(f"**{ev.name}** ({ev.months_left} months left): {ev.description}")
    
    # Action log (last 10)
    if st.session_state.action_log:
        with st.expander("📜 Action Log (last 10)"):
            for log in st.session_state.action_log[-10:]:
                st.text(log)

# ==================== Monte Carlo Section ====================

st.divider()
st.header("🎲 Monte Carlo Simulation")

col1, col2 = st.columns([2, 1])

with col1:
    sims = st.slider("Number of simulations", 100, 2000, 500, step=100)

with col2:
    run_mc = st.button("▶️ Run Monte Carlo", use_container_width=True, type="primary")

if run_mc:
    with st.spinner(f"Running {sims} simulations..."):
        mc = run_monte_carlo(sims, config=None)
        st.session_state.mc_results = mc
    st.success(f"Completed {sims} simulations!")

if st.session_state.mc_results:
    mc = st.session_state.mc_results
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Win Rate", format_pct(mc['win_rate']))
    
    with col2:
        st.metric("Median IRR (wins)", format_pct(mc['median_irr']))
    
    with col3:
        st.metric("Median MOIC (wins)", f"{mc['median_moic']:.2f}x")
    
    st.divider()
    
    # Distributions
    st.subheader("📈 Return Distributions")
    
    results = mc['results']
    valid_results = [r for r in results if r['valid_run']]
    wins = [r for r in valid_results if r['win']]
    
    if not wins:
        st.warning(f"⚠️ No winning runs out of {len(valid_results)} simulations. Try increasing the number of simulations or check game balance.")
    else:
        # IRR histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # IRR plot
        irrs = [r['irr'] * 100 for r in wins if r['irr_status'] == 'valid']
        if irrs:
            ax1.hist(irrs, bins=30, color='#4ECDC4', edgecolor='black', alpha=0.7)
            ax1.axvline(np.median(irrs), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(irrs):.1f}%')
            ax1.set_xlabel('IRR (%)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('IRR Distribution (Wins Only)')
            ax1.legend()
            ax1.grid(alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No valid IRR data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('IRR Distribution (Wins Only)')
        
        # MOIC plot
        moics = [r['moic'] for r in wins]
        if moics:
            ax2.hist(moics, bins=30, color='#FF6B6B', edgecolor='black', alpha=0.7)
            ax2.axvline(np.median(moics), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(moics):.2f}x')
            ax2.set_xlabel('MOIC (x)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('MOIC Distribution (Wins Only)')
            ax2.legend()
            ax2.grid(alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No MOIC data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('MOIC Distribution (Wins Only)')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Detailed statistics table
        st.subheader("📊 Detailed Statistics")
        
        if irrs and moics:
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'P10', 'P25', 'P75', 'P90'],
                'IRR (%)': [
                    f"{np.mean(irrs):.1f}",
                    f"{np.median(irrs):.1f}",
                    f"{np.std(irrs):.1f}",
                    f"{np.percentile(irrs, 10):.1f}",
                    f"{np.percentile(irrs, 25):.1f}",
                    f"{np.percentile(irrs, 75):.1f}",
                    f"{np.percentile(irrs, 90):.1f}",
                ],
                'MOIC (x)': [
                    f"{np.mean(moics):.2f}",
                    f"{np.median(moics):.2f}",
                    f"{np.std(moics):.2f}",
                    f"{np.percentile(moics, 10):.2f}",
                    f"{np.percentile(moics, 25):.2f}",
                    f"{np.percentile(moics, 75):.2f}",
                    f"{np.percentile(moics, 90):.2f}",
                ]
            })
            
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        else:
            st.info("📊 Statistics will appear when there are winning runs with valid IRR data.")

# ==================== Footer ====================

st.divider()
st.caption("Liquidity Crunch v2.0 | Powered by Streamlit | Logic: engine.py (headless)")
