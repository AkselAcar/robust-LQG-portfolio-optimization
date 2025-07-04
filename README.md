# Robust LQG Portfolio Optimization

**Semester Project – Spring 2025**  
**Author:** Aksel Berk Acar  
**Supervisors:** Jakob Nylöf and Prof. Giancarlo Ferrari Trecate

---

## Overview

This project implements a **robust Linear Quadratic Gaussian (LQG) controller** for portfolio optimization under uncertainty in both process and observation noise. The approach extends standard LQG control with distributional robustness via semi-definite programming (SDP).

---

## Getting Started

1. Ensure all required Python packages are installed (`numpy`, `matplotlib`, etc.).
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

Plots are saved to the `plots/` directory (if you want to run this code locally please don't forget to crerate a `plots/` directory):

- **State Trajectories** – Mean and standard deviation bands
- **Instantaneous Cost** – Nominal vs. Robust
- **Cumulative Cost**
- **Total Cost per Simulation Run**
- **Cost vs. Noise Scaling Factor**

---

##  Notes

- The SDP is **solved offline** and cached unless parameters change.
- For low-dimensional systems (1D or 2D), computation is fast. It can take longer for higher dimensions ! 
- Tuning `rho_*` parameters impacts robustness-performance trade-off. Putting `rho_w` high makes the variance explode in the somulations which is something to investigate. 

---


