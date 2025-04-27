import numpy as np
import pandas as pd
from scipy.stats import norm
import streamlit as st

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def generate_option_price_grid(K, T, r, S_center, sigma_center, option_type="call"):
    spot_prices = np.linspace(S_center * 0.8, S_center * 1.2, 10)
    volatilities = np.linspace(max(0.01, sigma_center * 0.5), sigma_center * 1.5, 10)
    
    grid_data = {}
    for sigma in volatilities:
        row = []
        for S in spot_prices:
            if option_type == "call":
                price = black_scholes_call(S, K, T, r, sigma)
            else:
                price = black_scholes_put(S, K, T, r, sigma)
            row.append(price)
        grid_data[f"σ={sigma:.2f}"] = row
    
    df = pd.DataFrame(grid_data, index=[f"S={s:.2f}" for s in spot_prices])
    return df

st.title("Black-Scholes Option Pricing")

with st.sidebar:
    S = st.number_input("S", min_value=0.01, value=100.0)
    K = st.number_input("K", min_value=0.01, value=100.0)
    T = st.number_input("T ~ years", min_value=0.01, value=1.0)
    r = st.number_input("r", min_value=0.0, value=0.05)
    sigma = st.number_input("Vol", min_value=0.01, value=0.2)

call_price = black_scholes_call(S, K, T, r, sigma)
put_price = black_scholes_put(S, K, T, r, sigma)

st.write(f" **Call Price**: ${call_price:.2f}")
st.write(f" **Put Price**: ${put_price:.2f}")

st.header("Price Grids, Spot vs. Vol")

# Two columns side-by-side
col1, col2 = st.columns(2)

with col1:
    st.subheader("Call")
    call_df = generate_option_price_grid(K, T, r, S, sigma, option_type="call")
    st.dataframe(call_df.style.background_gradient(cmap="viridis").format("${:.2f}"))

with col2:
    st.subheader("Put")
    put_df = generate_option_price_grid(K, T, r, S, sigma, option_type="put")
    st.dataframe(put_df.style.background_gradient(cmap="viridis").format("${:.2f}"))



# --- Heston Model Simulation and Pricing ---

st.header("Heston Model Option Pricing (takes a few seconds)")

rho = st.sidebar.number_input("Correlation (heston)", min_value=-1.0, max_value=1.0, value=0.15)
v0 = sigma ** 2  # Initial variance from implied vol
kappa = st.sidebar.number_input("Mean Reversion Rate (heston)", min_value=0.01, value=2.0)
theta = st.sidebar.number_input("Long Run Variance (heston)", min_value=0.0001, value=0.04)
xi = st.sidebar.number_input("Volatility of Variance (heston)", min_value=0.01, value=0.5)

def heston_price(S0, K, T, r, v0, rho, kappa, theta, xi, n_steps=1000, n_paths=2000):
    dt = T / n_steps
    S = np.full(n_paths, S0)
    v = np.full(n_paths, v0)
    
    rng = np.random.default_rng() 

    for _ in range(n_steps):
        Z1 = rng.uniform(0, 1, n_paths)
        Z2 = rng.uniform(0, 1, n_paths)
        #inverse transf.
        W1 = norm.ppf(Z1)
        W2 = rho * W1 + np.sqrt(1 - rho**2) * norm.ppf(Z2)
        
        v = np.maximum(v + kappa * (theta - v) * dt + xi * np.sqrt(v * dt) * W2, 0)
        S = S * np.exp((r - 0.5 * v) * dt + np.sqrt(v * dt) * W1)

    call_payoff = np.maximum(S - K, 0)
    put_payoff = np.maximum(K - S, 0)
    
    call_price = np.exp(-r * T) * np.mean(call_payoff)
    put_price = np.exp(-r * T) * np.mean(put_payoff)
    
    return call_price, put_price

heston_call_price, heston_put_price = heston_price(S, K, T, r, v0, rho, kappa, theta, xi)

st.write(f" **Call Price**: ${heston_call_price:.2f}")
st.write(f" **Put Price**: ${heston_put_price:.2f}")

# --- Heston Model Option Price Grids ---

st.subheader("Price Grids, Spot vs. Vol")

spot_prices = np.linspace(50, 150, 10)
volatilities = np.linspace(0.1, 0.5, 10)

call_grid = []
put_grid = []

for sigma_ in volatilities:
    call_row = []
    put_row = []
    for S_ in spot_prices:
        v0_grid = sigma_**2
        call_price_, put_price_ = heston_price(S_, K, T, r, v0_grid, rho, kappa, theta, xi, n_steps=500, n_paths=2000)
        call_row.append(call_price_)
        put_row.append(put_price_)
    call_grid.append(call_row)
    put_grid.append(put_row)

import pandas as pd

call_df = pd.DataFrame(call_grid, index=[f"σ={v:.2f}" for v in volatilities], columns=[f"S={s:.0f}" for s in spot_prices])
put_df = pd.DataFrame(put_grid, index=[f"σ={v:.2f}" for v in volatilities], columns=[f"S={s:.0f}" for s in spot_prices])

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Call**")
    st.dataframe(call_df.style.background_gradient(cmap="viridis").format("${:.2f}"))

with col2:
    st.markdown("**Put**")
    st.dataframe(put_df.style.background_gradient(cmap="plasma").format("${:.2f}"))

st.caption("Heston simulation uses 10,000 paths and 1,000 time steps for rho-correlated W_1, W_2.")



