import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from scipy.stats import norm

def display_monte_carlo_graph():
    # --- Controls above the graph ---
    st.write("### Simulation Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        initial_price = st.number_input("Initial Price (S₀)", 50, 500, 100, step=10)  # Adjustable initial price
        volatility = st.slider("Volatility (σ)", 0.01, 1.0, 0.2, step=0.01)
        drift = st.slider("Drift (μ)", -0.1, 0.2, 0.05, step=0.01)

    with col2:
        interest_rate = st.slider("Interest Rate (r)", 
            min_value=0.001,  # Prevent zero values
            max_value=0.2,
            value=0.05,
            step=0.005)
        
        T = st.number_input("Time to Expiration (years)", 
            min_value=0.1,  # Prevent values too close to zero
            max_value=5.0,
            value=1.0,
            step=0.1)
        
        num_paths = st.number_input("Number of Simulation Paths", 
            min_value=1,
            max_value=1000,
            value=50,
            step=1)

    with col3:
        num_steps = st.slider("Time Steps", 
            min_value=50,
            max_value=500,
            value=100,
            step=10)
        
        strike = st.number_input("Strike Price", 
            min_value=1.0,  # Prevent zero/negative values
            max_value=150.0,
            value=100.0,
            step=10.0)

    st.write("---")  # Separator before the graph

    # Add validation checks before calculations
    if initial_price <= 0 or strike <= 0:
        st.error("Initial price and strike price must be greater than 0")
        return
        
    if volatility <= 0:
        st.error("Volatility must be greater than 0")
        return
        
    if T <= 0:
        st.error("Time to expiration must be greater than 0")
        return

    # Protect against division by zero or very small numbers
    try:
        dt = T / num_steps
        if dt <= 0:
            st.error("Invalid time step calculated. Please check your parameters.")
            return

        # Initialize simulation array
        paths = np.zeros((num_steps + 1, int(num_paths)))
        paths[0] = initial_price

        # Generate paths using geometric Brownian motion
        for t in range(1, num_steps + 1):
            z = np.random.standard_normal(int(num_paths))
            paths[t] = paths[t-1] * np.exp((drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z)

        # Monte Carlo option pricing
        call_payoffs = np.maximum(paths[-1] - strike, 0)
        mc_option_price = np.exp(-interest_rate * T) * np.mean(call_payoffs)

        # Black-Scholes option pricing - Add error handling
        d1 = (np.log(initial_price / strike) + (interest_rate + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)
        bs_option_price = initial_price * norm.cdf(d1) - strike * np.exp(-interest_rate * T) * norm.cdf(d2)

        # Check for NaN or infinite values
        if np.isnan(mc_option_price) or np.isinf(mc_option_price) or \
           np.isnan(bs_option_price) or np.isinf(bs_option_price):
            st.error("Invalid result calculated. Please adjust your parameters.")
            return

        # Display the computed option prices
        st.write("### Option Prices")
        st.write(f"**Monte Carlo Estimated Option Price:** {mc_option_price:.2f}")
        st.write(f"**Black-Scholes Option Price:** {bs_option_price:.2f}")

        # --- Visualization section ---
        df = pd.DataFrame(paths)
        df['time'] = np.linspace(0, T, num_steps+1)
        df_melted = df.melt(id_vars=['time'], var_name='Simulation', value_name='Price')

        # Compute mean price path
        mean_path = np.mean(paths, axis=1)
        df_mean = pd.DataFrame({'time': np.linspace(0, T, num_steps+1), 'Price': mean_path})

        # Paths chart
        paths_chart = alt.Chart(df_melted).mark_line(opacity=0.3).encode(
            x=alt.X("time:Q", title="Time (years)", scale=alt.Scale(nice=False)),
            y=alt.Y("Price:Q", title="Underlying Price", scale=alt.Scale(nice=False)),
            color=alt.Color("Simulation:N", legend=alt.Legend(title="Price Paths")),
            tooltip=alt.value(None)
        )

        # Overlay mean path in bold red
        mean_chart = alt.Chart(df_mean).mark_line(color="red", size=3).encode(
            x=alt.X("time:Q", title="Time (years)", scale=alt.Scale(nice=False)),
            y=alt.Y("Price:Q", title="Mean Underlying Price", scale=alt.Scale(nice=False)),
        )

        hover_chart = alt.Chart(df_melted).mark_line(
            strokeWidth=5,  # Much thicker line for hovering over
            opacity=0        # Completely transparent
        ).encode(
            x=alt.X("time:Q", title="Time (years)", scale=alt.Scale(nice=False)),
            y=alt.Y("Price:Q", title="Underlying Price", scale=alt.Scale(nice=False)),
        )

        # Combine charts
        chart = (paths_chart + hover_chart + mean_chart).properties(
            width=700,
            height=630,
            title="Monte Carlo Simulation: Colored Paths with Mean Price",
        )

        st.altair_chart(chart, use_container_width=True)

    except (ZeroDivisionError, ValueError, RuntimeWarning) as e:
        st.error(f"An error occurred during calculations. Please check your input parameters: {str(e)}")
        return
