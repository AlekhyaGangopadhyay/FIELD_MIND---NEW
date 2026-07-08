"""
FIELD-MIND Physics-Informed Synthetic Dataset Generator
=======================================================
Project  : FIELD-MIND — Offline Multimodal Agentic AI for Underground Mining
Authors  : IEM Kolkata (Subhadip Chandra, Ankita Ray Chowdhury)
Version  : 1.0
Date     : 2024

Sensors  : MQ-2, MQ-3, MQ-4, MQ-7, MQ-135, MQ-136, MG811, PM2.5 Dust

Physics models used
-------------------
1. Gaussian Plume Model       — background gas concentration field
2. Blast Event Model          — exponential decay spike injection
3. Sensor Drift Model         — Rs exponential aging + random walk
4. Ventilation Cycle          — sinusoidal airflow fluctuation
5. Shift-Start Diesel Pulse   — benzene / NOx at machinery start

Hazard thresholds (OSHA / MSHA standards)
-----------------------------------------
CH4  > 1000 ppm  (10% LEL)
CO   >   50 ppm  (OSHA TWA)
H2S  >   10 ppm  (MSHA action level)
CO2  > 5000 ppm  (poor ventilation)
Dust >  150 ug/m3 (silica hazard)
"""

import os
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SEED            = 42
N               = 50_000            # number of rows
SAMPLE_INTERVAL = "1s"              # 1-second sampling
START_TIME      = "2024-01-01 06:00:00"
script_dir      = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH     = os.path.join(script_dir, "data", "FIELDMIND_physics_dataset.csv")

rng = np.random.default_rng(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# TIMESTAMPS
# ─────────────────────────────────────────────────────────────────────────────
timestamps = pd.date_range(start=START_TIME, periods=N, freq=SAMPLE_INTERVAL)
t_arr = np.arange(N, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# BLAST EVENT SCHEDULE
# ~1 blast every 4000s ± 800s  (representative of active heading)
# ─────────────────────────────────────────────────────────────────────────────
blast_times = []
t = 2000
while t < N:
    blast_times.append(int(t))
    t += int(rng.normal(4000, 800))
blast_times = [b for b in blast_times if b < N]

blast_marker = np.zeros(N, dtype=int)
for bt in blast_times:
    blast_marker[bt] = 1


def blast_envelope(N, blast_times, peak, decay):
    """
    Exponential decay envelope after each blast event.

    Formula:
        env(t) = peak * exp(-(t - t_blast) / decay)   for t >= t_blast

    Parameters
    ----------
    peak  : float  — amplitude at t_blast
    decay : float  — time constant in seconds
                     fast decay (120s)  → dust, NOx clear quickly
                     slow decay (300s)  → CO afterdamp lingers
    """
    env = np.zeros(N)
    for bt in blast_times:
        window = min(decay * 8, N - bt)
        ta = np.arange(window)
        env[bt:bt + window] += peak * np.exp(-ta / decay)
    return env


blast_fast = blast_envelope(N, blast_times, peak=1.0, decay=120)   # dust, NOx, smoke
blast_slow = blast_envelope(N, blast_times, peak=1.0, decay=300)   # CO afterdamp, NH3


# ─────────────────────────────────────────────────────────────────────────────
# GAUSSIAN PLUME BACKGROUND
# ─────────────────────────────────────────────────────────────────────────────
def gaussian_plume_background(N, Q, u, sigma, t_arr, rng):
    """
    Simplified 1-D Gaussian plume model for underground tunnel gas dispersion.

    Full 3-D Gaussian plume:
        C(x,y,z) = Q / (2*pi*sigma_y*sigma_z*u)
                   * exp(-y^2 / (2*sigma_y^2))
                   * exp(-(z-H)^2 / (2*sigma_z^2))

    Simplified to time-domain at a fixed sensor location:
        C(t) = Q(t) / (2 * pi * sigma^2 * u(t))

    where:
        Q(t) = Q * (1 + 0.15 * sin(2*pi*t / T_cycle))
               — emission rate with 4-hour underground ventilation cycle
        u(t) = u + N(0, 0.1*u)
               — airflow velocity with 10% turbulence noise
        sigma = dispersion coefficient (m)  [gas-specific]

    Output is smoothed with a 60-second rolling mean to
    simulate spatial diffusion in the tunnel.

    Parameters
    ----------
    Q     : float — base emission rate (arbitrary source units)
    u     : float — mean airflow velocity (m/s)
    sigma : float — dispersion coefficient (m)
    """
    T_cycle = 14400  # 4-hour ventilation cycle (seconds)
    Q_t = Q * (1 + 0.15 * np.sin(2 * np.pi * t_arr / T_cycle))
    u_t = np.clip(u + rng.normal(0, u * 0.1, N), u * 0.3, u * 3.0)
    C = Q_t / (2 * np.pi * sigma ** 2 * u_t)
    C = pd.Series(C).rolling(60, min_periods=1, center=True).mean().to_numpy().copy()
    return C


# ─────────────────────────────────────────────────────────────────────────────
# SENSOR DRIFT MODEL
# ─────────────────────────────────────────────────────────────────────────────
def sensor_drift(N, tau, drift_magnitude, t_arr, rng):
    """
    Models MQ-series sensor aging as an exponential decay toward steady state,
    matching the warm-up and long-term drift behaviour described in MQ datasheets.

    Formula:
        d(t) = 1 + M * exp(-t / tau) + RandomWalk(t)

    where:
        M   = drift_magnitude  — fractional over-reading at t=0
                                 e.g. 0.15 means sensor reads 15% high initially
        tau = time constant (seconds) — how quickly drift settles
        RandomWalk(t) = cumsum(N(0, sigma_rw))
                      — slow stochastic aging noise on top of deterministic decay

    The multiplicative drift is applied to the physical gas signal:
        observed(t) = true_signal(t) * d(t)

    Drift parameters per sensor (tau in seconds):
        MQ-2   : tau=6 days,  M=0.15
        MQ-3   : tau=8 days,  M=0.10
        MQ-4   : tau=5 days,  M=0.18
        MQ-7   : tau=7 days,  M=0.13
        MQ-135 : tau=9 days,  M=0.09
        MQ-136 : tau=4 days,  M=0.20  (most sensitive, drifts fastest)
        MG811  : tau=10 days, M=0.08
        PM2.5  : tau=3 days,  M=0.22  (optical path degrades with dust)
    """
    drift = 1.0 + drift_magnitude * np.exp(-t_arr / tau)
    rw = np.cumsum(rng.normal(0, 0.00003, N))
    rw -= rw[0]
    return drift + rw


d_mq2   = sensor_drift(N, 50000 * 6,  0.15, t_arr, rng)
d_mq3   = sensor_drift(N, 50000 * 8,  0.10, t_arr, rng)
d_mq4   = sensor_drift(N, 50000 * 5,  0.18, t_arr, rng)
d_mq7   = sensor_drift(N, 50000 * 7,  0.13, t_arr, rng)
d_mq135 = sensor_drift(N, 50000 * 9,  0.09, t_arr, rng)
d_mq136 = sensor_drift(N, 50000 * 4,  0.20, t_arr, rng)
d_mg811 = sensor_drift(N, 50000 * 10, 0.08, t_arr, rng)
d_pm25  = sensor_drift(N, 50000 * 3,  0.22, t_arr, rng)


# ─────────────────────────────────────────────────────────────────────────────
# GAS SIGNAL GENERATION
# Each gas = physics background + event injection + sensor noise + drift
# ─────────────────────────────────────────────────────────────────────────────

# ── MQ-4 & MQ-2: CH4 (Methane) ───────────────────────────────────────────────
# Source: coal seam outgassing  |  Baseline: 100-200 ppm  |  Hazard: >1000 ppm
# Blast DEPLETES CH4 (forced ventilation flush after blast)
ch4_base = (gaussian_plume_background(N, 30000, 2.5, 10, t_arr, rng)
            + rng.normal(0, 30, N)
            - blast_slow * 100)
ch4_base = np.clip(ch4_base, 100, None)
MQ4_CH4_ppm = np.clip(ch4_base * d_mq4, 0, None)
MQ2_CH4_ppm = np.clip(ch4_base * 0.85 * d_mq2 + rng.normal(0, 15, N), 0, None)
# Note: MQ-2 reads ~85% of MQ-4 for CH4 due to different sensitivity curve

# ── MQ-7 & MQ-2: CO (Carbon Monoxide) ────────────────────────────────────────
# Source: blast afterdamp, diesel engines  |  Hazard: >50 ppm (OSHA TWA)
# CO spikes AFTER blast with slow decay (combustion byproduct lingers)
co_base = (gaussian_plume_background(N, 1500, 2.5, 7, t_arr, rng)
           + blast_slow * 80
           + rng.normal(0, 4, N))
co_base = np.clip(co_base, 0, None)
MQ7_CO_ppm = np.clip(co_base * d_mq7, 0, None)
MQ2_CO_ppm = np.clip(co_base * 0.92 * d_mq2 + rng.normal(0, 3, N), 0, None)

# ── MQ-2: LPG ─────────────────────────────────────────────────────────────────
# Source: equipment fuel storage  |  Slow sinusoidal variation (pressure/temp)
lpg_base = 110 + 25 * np.sin(2 * np.pi * t_arr / 7200) + rng.normal(0, 15, N)
MQ2_LPG_ppm = np.clip(lpg_base * d_mq2, 0, None)

# ── MQ-2: Smoke ───────────────────────────────────────────────────────────────
# Source: blast fumes, electrical fires  |  Correlated with blast_fast
smoke_base = 65 + blast_fast * 300 + rng.normal(0, 12, N)
MQ2_Smoke_ppm = np.clip(smoke_base * d_mq2, 0, None)

# ── MQ-3: Alcohol ─────────────────────────────────────────────────────────────
# Source: equipment cleaning solvents  |  Low level background
MQ3_Alcohol_ppm = np.clip((8 + rng.normal(0, 2, N)) * d_mq3, 0, None)

# ── MQ-3: Benzene ─────────────────────────────────────────────────────────────
# Source: diesel exhaust  |  Peaks at shift start when engines fire up
# Shift start = first 3600s of every 28800s (~8h) shift cycle
shift_flag = ((t_arr % 28800) < 3600).astype(float)
benzene_base = 4 + shift_flag * 3 + rng.normal(0, 1.5, N)
MQ3_Benzene_ppm = np.clip(benzene_base * d_mq3, 0, None)

# ── MQ-135: NH3 (Ammonia) ─────────────────────────────────────────────────────
# Source: ANFO explosive residue  |  Post-blast correlation with blast_slow
nh3_base = 7 + blast_slow * 40 + rng.normal(0, 2, N)
MQ135_NH3_ppm = np.clip(nh3_base * d_mq135, 0, None)

# ── MQ-135: NOx ───────────────────────────────────────────────────────────────
# Source: diesel engines + blast  |  Fast decay matches blast_fast
nox_base = 0.04 + blast_fast * 0.2 + rng.normal(0, 0.01, N)
MQ135_NOx_ppm = np.clip(nox_base * d_mq135, 0, None)

# ── MQ-135 & MG811: CO2 ──────────────────────────────────────────────────────
# Source: workers breathing + diesel + blasting  |  Hazard: >5000 ppm
# MQ-135 and MG811 are cross-validated — slight offset due to different
# electrochemical vs NDIR measurement principles
co2_base = (gaussian_plume_background(N, 200000, 2.5, 18, t_arr, rng)
            + 1400
            + rng.normal(0, 100, N))
co2_base = np.clip(co2_base, 400, None)
MQ135_CO2_ppm = np.clip(co2_base * d_mq135, 400, None)
MG811_CO2_ppm = np.clip(co2_base * 1.02 * d_mg811 + rng.normal(0, 50, N), 400, None)

# ── MQ-136: H2S (Hydrogen Sulphide) ──────────────────────────────────────────
# Source: sulphide ore zones  |  Hazard: >10 ppm (MSHA action level)
# Independent Gaussian plume (ore zone is spatially separate from blast)
h2s_base = (gaussian_plume_background(N, 500, 2.5, 5, t_arr, rng)
            + rng.normal(0, 0.8, N))
h2s_base = np.clip(h2s_base, 0, None)
MQ136_H2S_ppm = np.clip(h2s_base * d_mq136, 0, None)

# ── PM2.5 Dust Sensor ─────────────────────────────────────────────────────────
# Source: blast particulate + continuous ore dust  |  Hazard: >150 ug/m3
# Strongest blast correlation; fast clearing (particles settle)
# Also has slow hourly oscillation from ventilation air movement
dust_base = (35
             + blast_fast * 900
             + 12 * np.sin(2 * np.pi * t_arr / 3600)
             + rng.normal(0, 10, N))
PM25_Dust_ugm3 = np.clip(dust_base * d_pm25, 0, None)


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENTAL CONTEXT
# ─────────────────────────────────────────────────────────────────────────────
# Temperature: ~28°C underground with slow sinusoidal variation
# Humidity: ~76% with inverse relationship to temperature
Temp_C   = np.clip(28 + 2 * np.sin(2 * np.pi * t_arr / 50000)
                   + rng.normal(0, 0.5, N), 20, 40)
Humidity = np.clip(76 + 4 * np.sin(2 * np.pi * t_arr / 50000 + np.pi)
                   + rng.normal(0, 1, N), 55, 99)


# ─────────────────────────────────────────────────────────────────────────────
# HAZARD LABEL  (multi-condition compound logic per OSHA/MSHA)
# ─────────────────────────────────────────────────────────────────────────────
hazard = (
    (MQ4_CH4_ppm    > 1000) |   # 10% LEL for methane (explosion risk)
    (MQ7_CO_ppm     >   50) |   # OSHA TWA 8-hour limit
    (MQ136_H2S_ppm  >   10) |   # MSHA action level
    (MG811_CO2_ppm  > 5000) |   # Poor ventilation / asphyxiation risk
    (PM25_Dust_ugm3 >  150)     # Silica / coal dust inhalation hazard
).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLE DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame({
    'Timestamp':        timestamps,
    # MQ-2: LPG, CH4, CO, Smoke
    'MQ2_LPG_ppm':      np.round(MQ2_LPG_ppm,    3),
    'MQ2_CH4_ppm':      np.round(MQ2_CH4_ppm,    3),
    'MQ2_CO_ppm':       np.round(MQ2_CO_ppm,     3),
    'MQ2_Smoke_ppm':    np.round(MQ2_Smoke_ppm,  3),
    # MQ-3: Alcohol, Benzene
    'MQ3_Alcohol_ppm':  np.round(MQ3_Alcohol_ppm, 3),
    'MQ3_Benzene_ppm':  np.round(MQ3_Benzene_ppm, 3),
    # MQ-4: CH4 (dedicated methane sensor)
    'MQ4_CH4_ppm':      np.round(MQ4_CH4_ppm,    3),
    # MQ-7: CO (dedicated CO sensor)
    'MQ7_CO_ppm':       np.round(MQ7_CO_ppm,     3),
    # MQ-135: NH3, NOx, CO2
    'MQ135_NH3_ppm':    np.round(MQ135_NH3_ppm,  3),
    'MQ135_NOx_ppm':    np.round(MQ135_NOx_ppm,  4),
    'MQ135_CO2_ppm':    np.round(MQ135_CO2_ppm,  2),
    # MQ-136: H2S
    'MQ136_H2S_ppm':    np.round(MQ136_H2S_ppm,  4),
    # MG811: CO2 (NDIR dedicated CO2 sensor)
    'MG811_CO2_ppm':    np.round(MG811_CO2_ppm,  2),
    # PM2.5 Dust sensor
    'PM25_Dust_ugm3':   np.round(PM25_Dust_ugm3, 3),
    # Environmental context
    'Temp_C':           np.round(Temp_C,          2),
    'Humidity_pct':     np.round(Humidity,         2),
    # Event / label columns
    'Blast_Event':      blast_marker,
    'Hazard_Alert':     hazard,
})

df.to_csv(OUTPUT_PATH, index=False)

print(f"Dataset saved   : {OUTPUT_PATH}")
print(f"Shape           : {df.shape}")
print(f"Blast events    : {int(blast_marker.sum())}")
print(f"Hazard alerts   : {int(hazard.sum())} ({hazard.mean()*100:.2f}%)")
print(f"Time span       : {df['Timestamp'].iloc[0]} -> {df['Timestamp'].iloc[-1]}")
