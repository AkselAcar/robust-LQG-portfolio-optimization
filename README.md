# Robust LQG Portfolio Optimization

**Semester Project – Spring 2025**  
**Author:** Aksel Berk Acar  
**Supervisors:** Jakob Nylöf and Prof. Giancarlo Ferrari Trecate

---
## Overview

This project implements a **distributionally robust Linear Quadratic Gaussian (LQG) controller** for financial portfolio optimization under model uncertainty. The framework models financial assets using controlled Ornstein-Uhlenbeck processes, suitable for mean-reverting trading strategies. The approach addresses the limitations of traditional LQG controllers that assume known Gaussian noise statistics by incorporating Wasserstein ambiguity sets and semi-definite programming to handle distributional uncertainty in financial markets.

---

## Getting Started

1. Ensure up to date Python verison and that all required Python packages are installed:
- Python 3.7+
- Required packages: `numpy`, `matplotlib`, `cvxpy`, `scipy`
2. Run the `main.py` script. This will:
   - Compute LQG and robust controller gains,
   - Simulate trajectories,
   - Plot results and cost comparisons.
3. All parameter modifications should be made directly in `main.py`.

---

## Modifications

To change the **dimensions** of the system, modify these lines in `main.py`:

- Line 14: `n` → number of state dimensions
- Line 18: `mu` → target mean price vector
- Line 24: `theta` → mean reversion coefficients

Make sure matrix shapes are consistent with the new dimension.

---

## Project Structure

The main file (`main.py`) handles:

- Computing **LQR and Kalman gains** (nominal and robust)
- Solving the **SDP** for robust noise covariances (`save_sdp_results.py`)
- Simulating **nominal and robust** controllers
- Plotting:
  - State trajectories
  - Instantaneous & cumulative costs
  - Total cost per run
  - Noise sensitivity
- Managing saved SDP solutions with a hash check

---

## Hyperparameters

All key settings are located near the top of `main.py`:

- Time and discretization:  
  `time_horizon`, `dt`, `T`
- Covariances:  
  `sigma_x0`, `sigma_w`, `sigma_v`
- Cost matrices:  
  `Q`, `R`
- Robustness radii:  
  `rho_x0`, `rho_w`, `rho_v`

---

## Outputs

Plots are saved to the `plots/` directory (if you want to run this code locally please don't forget to create a `plots/` directory):

- **State Trajectories** – Mean and standard deviation bands
- **Instantaneous Cost** – Nominal vs. Robust
- **Cumulative Cost**
- **Total Cost per Simulation Run**
- **Cost vs. Noise Scaling Factor**

---

##  Notes

- The SDP is **solved offline** and cached unless parameters change.
- For low-dimensional systems (1D or 2D), computation is fast. It can take longer for higher dimensions! 
- Tuning `rho_*` parameters impacts robustness-performance trade-off. 
---


