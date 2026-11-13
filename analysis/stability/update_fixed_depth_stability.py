#!/usr/bin/env python3
"""
JoHRS Reviewer 2-2 / 2-4:
Update the existing Wedderburn-number and Lake-number event diagnostics so that they
use the corrected fixed-depth observation protocol and final event catalogs.

This is NOT a new physical model. It preserves the formulas, constants, simplified
morphometry, wind-height correction, thermocline definition, and event-scale
summary used in the original Figure-S.ipynb stability section.

Changes relative to the old notebook:
  1) events are NOT re-detected from old all_data;
  2) observed temperature profiles come from the R1-5 corrected fixed-depth archives;
  3) final event catalogs are:
       Shiode      = 91 strict events, 0.25--2.25 m, 11 depths
       Yamanokami  = 13 strict events, 0.40--2.20 m, 10 depths
       Shin        = 0 events, therefore not included in event stability plots;
  4) aligned CSVs are used only for meteorology (wind, air temperature, pressure).

Original physical settings retained:
  G = 9.81 m s-2
  rho_w_ref = 1000 kg m-3
  rho_air_default = 1.2 kg m-3
  Cd = 1.3e-3
  z0 = 0.0013 m
  wind observed at 2 m and converted to 10 m with neutral log profile
  thermocline depth = midpoint of adjacent sensors with maximum |dT/dz|
  fetch = sqrt(surface area) when site-specific effective fetch is unavailable
  simplified morphometry exactly as in the original notebook

Interpretation:
  Because morphometry/fetch are simplified and the profile outside retained sensor
  depths is approximated by nearest retained temperature, W and Lake number should be
  treated as relative stability diagnostics, not precise basin-scale estimates.

Dependencies:
  numpy, pandas, matplotlib
"""
from __future__ import annotations

import argparse
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 0. Constants: copied from original Figure-S.ipynb
# ============================================================

G = 9.81
RHO_W_REF = 1000.0
RHO_AIR_DEFAULT = 1.2
CD_DEFAULT = 1.3e-3
Z0_WATER = 0.0013
USE_WIND_10M_CORRECTION = True

try:
    integrate_trapz = np.trapezoid
except AttributeError:
    integrate_trapz = np.trapz

SITE_ORDER = ["yamanokami", "shiode"]
SITE_LABELS = {
    "yamanokami": "Yamanokami",
    "shiode": "Shiode",
}

# Original simplified morphometry retained.
SITE_CONFIGS = {
    "yamanokami": {
        "csv_name": "yamanokami_2024_aligned.csv",
        "max_depth_m": 2.40,
        "wind_height_m": 2.0,
        "H_from_bottom_m": [0, 0.5, 1.0, 1.5, 2.0, 2.4],
        "area_m2":         [0, 195, 295, 395, 495, 595],
        "fetch_m": None,
        "expected_depths_m": np.arange(0.40, 2.2001, 0.20),
        "expected_events": 13,
    },
    "shiode": {
        "csv_name": "shiode_2025_aligned.csv",
        "max_depth_m": 2.45,
        "wind_height_m": 2.0,
        "H_from_bottom_m": [0, 0.5, 1.0, 1.5, 2.0, 2.45],
        "area_m2":         [100, 500, 1000, 1500, 2000, 2380],
        "fetch_m": None,
        "expected_depths_m": np.arange(0.25, 2.2501, 0.20),
        "expected_events": 91,
    },
}

SHIODE_ARCHIVE = "Shiode_R1_5_step1A_return.zip"
YS_ARCHIVE = "Shin_Yamanokami_R1_5_events_return.zip"


def require(cond, msg):
    if not bool(cond):
        raise ValueError(msg)


def find_one(root: Path, filename: str, explicit: Path | None = None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        require(p.is_file(), f"Missing explicit input: {p}")
        return p

    here = Path(__file__).resolve().parent
    candidates = []
    for p in [here / "data" / filename, root / filename]:
        if p.is_file():
            candidates.append(p.resolve())

    if root.exists():
        candidates.extend(p.resolve() for p in root.rglob(filename))

    candidates = sorted(set(candidates))
    require(candidates, f"Cannot find {filename} below {root}")

    if len(candidates) > 1:
        print(f"NOTE: found {len(candidates)} copies of {filename}; using:")
        print(" ", candidates[0])

    return candidates[0]


def zip_map(z):
    return {n.replace("\\", "/"): n for n in z.namelist() if not n.endswith("/")}


def read_npz_from_zip(z, member):
    names = zip_map(z)
    require(member in names, f"Missing {member}")
    with np.load(io.BytesIO(z.read(names[member])), allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def read_csv_from_zip(z, member):
    names = zip_map(z)
    require(member in names, f"Missing {member}")
    return pd.read_csv(io.BytesIO(z.read(names[member])))


# ============================================================
# 1. Corrected fixed-depth observations + final events
# ============================================================

def load_corrected_site(site, shiode_zip, ys_zip):
    if site == "shiode":
        with zipfile.ZipFile(shiode_zip) as z:
            events = read_csv_from_zip(z, "events_B_fixed11_complete3h.csv")
            d = read_npz_from_zip(z, "fixed_depth_observation_reference.npz")

        time = pd.DatetimeIndex(pd.to_datetime(d["time"]))
        depths = np.asarray(d["depths"], float)
        temp = np.asarray(d["obs_data_quality"], float)
        valid = np.asarray(d["valid_mask"], bool)

    elif site == "yamanokami":
        with zipfile.ZipFile(ys_zip) as z:
            events = read_csv_from_zip(z, "yamanokami/core10/events_B.csv")
            d = read_npz_from_zip(z, "yamanokami/core10/observations_and_masks.npz")

        time = pd.DatetimeIndex(
            pd.to_datetime(d["time_ns"].astype("datetime64[ns]"))
        )
        depths = np.asarray(d["depths"], float)
        temp = np.asarray(d["temperature_raw"], float)
        valid = np.asarray(d["valid_data_only"], bool)

    else:
        raise ValueError(site)

    cfg = SITE_CONFIGS[site]
    require(len(events) == cfg["expected_events"],
            f"{site}: expected {cfg['expected_events']} final events, got {len(events)}")
    require(
        np.allclose(depths, cfg["expected_depths_m"], atol=1e-8, rtol=0),
        f"{site}: unexpected fixed depth set {depths.tolist()}",
    )
    require(
        temp.shape == valid.shape == (len(time), len(depths)),
        f"{site}: observation shape mismatch",
    )
    require(time.is_monotonic_increasing and time.is_unique,
            f"{site}: time axis must be sorted and unique")

    # Every final event must be fully valid over its stored 3-h window.
    for ev in events.itertuples(index=False):
        i = int(ev.start_index)
        j = int(ev.end_index)
        require(j - i == 18, f"{site}/{ev.event_id}: not a 19-point 3-h window")
        require(valid[i:j+1].all(),
                f"{site}/{ev.event_id}: final event is not fully valid")

    return {
        "site": site,
        "time": time,
        "depths": depths,
        "temp": temp,
        "valid": valid,
        "events": events.copy(),
    }


# ============================================================
# 2. Meteorology: same variables as original notebook
# ============================================================

def norm_col(s):
    return str(s).lower().replace(" ", "").replace("_", "").replace("-", "")


def find_column(df, candidates, required=True):
    candidates_n = [norm_col(c) for c in candidates]
    for col in df.columns:
        c = norm_col(col)
        if any(c == x or x in c for x in candidates_n):
            return col
    if required:
        raise ValueError(
            f"Could not find required column among {candidates}. "
            f"Available={list(df.columns)}"
        )
    return None


def load_meteorology(csv_path: Path, target_time: pd.DatetimeIndex):
    met = pd.read_csv(csv_path)

    time_col = find_column(met, ["time", "datetime", "date_time"], required=True)
    wind_col = find_column(
        met,
        ["wind", "windspeed", "wind_speed", "ws"],
        required=True,
    )
    air_col = find_column(
        met,
        ["air_temp", "airtemperature", "tair", "airtemp"],
        required=False,
    )
    pressure_col = find_column(
        met,
        ["pressure", "airpressure", "pres", "p_air"],
        required=False,
    )

    met = met.copy()
    met[time_col] = pd.to_datetime(met[time_col], errors="coerce")
    met = (
        met.dropna(subset=[time_col])
        .sort_values(time_col)
        .drop_duplicates(subset=[time_col], keep="last")
        .reset_index(drop=True)
    )

    base = pd.DataFrame({"time": target_time})
    cols = [time_col, wind_col]
    if air_col is not None:
        cols.append(air_col)
    if pressure_col is not None:
        cols.append(pressure_col)

    merged = base.merge(
        met[cols],
        left_on="time",
        right_on=time_col,
        how="left",
        validate="one_to_one",
    )

    wind = pd.to_numeric(merged[wind_col], errors="coerce").to_numpy(float)
    air = (
        pd.to_numeric(merged[air_col], errors="coerce").to_numpy(float)
        if air_col is not None else None
    )
    pressure = (
        pd.to_numeric(merged[pressure_col], errors="coerce").to_numpy(float)
        if pressure_col is not None else None
    )

    return {
        "wind": wind,
        "air_temp": air,
        "pressure": pressure,
        "columns": {
            "time": time_col,
            "wind": wind_col,
            "air_temp": air_col,
            "pressure": pressure_col,
        },
        "n_missing_wind": int((~np.isfinite(wind)).sum()),
        "n_missing_air_temp": (
            int((~np.isfinite(air)).sum()) if air is not None else len(target_time)
        ),
        "n_missing_pressure": (
            int((~np.isfinite(pressure)).sum()) if pressure is not None else len(target_time)
        ),
    }


# ============================================================
# 3. Original physical functions retained
# ============================================================

def water_density_freshwater(T_c):
    T = np.asarray(T_c, dtype=float)
    return 1000.0 * (
        1.0
        - ((T + 288.9414) / (508929.2 * (T + 68.12963)))
        * (T - 3.9863) ** 2
    )


def convert_wind_to_10m(Uz, z=2.0, z0=Z0_WATER):
    if not np.isfinite(Uz):
        return np.nan
    Uz = max(float(Uz), 0.0)
    if not USE_WIND_10M_CORRECTION:
        return Uz
    if abs(z - 10.0) < 1e-6:
        return Uz
    return Uz * np.log(10.0 / z0) / np.log(z / z0)


def air_density(air_temp_c=None, pressure_value=None):
    if air_temp_c is None or pressure_value is None:
        return RHO_AIR_DEFAULT
    if not np.isfinite(air_temp_c) or not np.isfinite(pressure_value):
        return RHO_AIR_DEFAULT

    p = float(pressure_value)
    if p < 200:
        p_pa = p * 1000.0
    elif p < 2000:
        p_pa = p * 100.0
    else:
        p_pa = p

    T_k = float(air_temp_c) + 273.15
    R_d = 287.05
    return p_pa / (R_d * T_k)


def water_friction_velocity(
    U_obs,
    wind_height_m=2.0,
    air_temp_c=None,
    pressure_value=None,
):
    if not np.isfinite(U_obs):
        return np.nan

    U10 = convert_wind_to_10m(U_obs, z=wind_height_m)
    rho_air = air_density(
        air_temp_c=air_temp_c,
        pressure_value=pressure_value,
    )
    tau = rho_air * CD_DEFAULT * U10**2
    return np.sqrt(tau / RHO_W_REF)


def build_bathymetry(cfg, dz=0.01):
    max_depth = cfg["max_depth_m"]
    H = np.asarray(cfg["H_from_bottom_m"], dtype=float)
    A = np.asarray(cfg["area_m2"], dtype=float)

    depth_from_surface = max_depth - H
    order = np.argsort(depth_from_surface)
    d = depth_from_surface[order]
    a = A[order]

    depth_grid = np.arange(0.0, max_depth + dz / 2, dz)
    area_grid = np.interp(depth_grid, d, a)
    area_grid = np.maximum(area_grid, 0.0)
    return depth_grid, area_grid


def volume_weighted_mean(values, depth_grid, area_grid, mask=None):
    values = np.asarray(values, dtype=float)

    if mask is None:
        mask = np.isfinite(values) & np.isfinite(area_grid)
    else:
        mask = mask & np.isfinite(values) & np.isfinite(area_grid)

    if np.sum(mask) < 2:
        return np.nan

    numerator = integrate_trapz(
        values[mask] * area_grid[mask],
        depth_grid[mask],
    )
    denominator = integrate_trapz(
        area_grid[mask],
        depth_grid[mask],
    )

    if denominator <= 0:
        return np.nan
    return numerator / denominator


def lake_volume_and_center(depth_grid, area_grid):
    V = integrate_trapz(area_grid, depth_grid)
    if V <= 0:
        return np.nan, np.nan
    z_cv = integrate_trapz(depth_grid * area_grid, depth_grid) / V
    return V, z_cv


def thermocline_depth_from_sensors(depths, temps, min_temp_range=0.10):
    depths = np.asarray(depths, dtype=float)
    temps = np.asarray(temps, dtype=float)

    valid = np.isfinite(depths) & np.isfinite(temps)
    depths = depths[valid]
    temps = temps[valid]

    if len(depths) < 3:
        return np.nan

    order = np.argsort(depths)
    depths = depths[order]
    temps = temps[order]

    if np.nanmax(temps) - np.nanmin(temps) < min_temp_range:
        return np.nan

    dz = np.diff(depths)
    dT = np.diff(temps)
    valid_dz = dz > 0

    if np.sum(valid_dz) == 0:
        return np.nan

    gradients = np.full_like(dT, np.nan, dtype=float)
    gradients[valid_dz] = np.abs(dT[valid_dz] / dz[valid_dz])

    if not np.isfinite(gradients).any():
        return np.nan

    idx = int(np.nanargmax(gradients))
    return float(0.5 * (depths[idx] + depths[idx + 1]))


def interpolate_temperature_to_grid(depths, temps, depth_grid):
    depths = np.asarray(depths, dtype=float)
    temps = np.asarray(temps, dtype=float)

    valid = np.isfinite(depths) & np.isfinite(temps)
    depths = depths[valid]
    temps = temps[valid]

    if len(depths) < 3:
        return None

    order = np.argsort(depths)
    depths = depths[order]
    temps = temps[order]

    # Original notebook behavior:
    # outside retained sensor range, use nearest retained observed temperature.
    return np.interp(
        depth_grid,
        depths,
        temps,
        left=temps[0],
        right=temps[-1],
    )


def calculate_stability_indices_one_time(
    temps,
    depths,
    U_obs,
    cfg,
    depth_grid,
    area_grid,
    air_temp_c=None,
    pressure_value=None,
):
    max_depth = cfg["max_depth_m"]
    A0 = max(cfg["area_m2"])
    fetch = (
        cfg["fetch_m"]
        if cfg["fetch_m"] is not None
        else np.sqrt(A0)
    )

    zT = thermocline_depth_from_sensors(depths, temps)
    T_grid = interpolate_temperature_to_grid(depths, temps, depth_grid)

    ustar = water_friction_velocity(
        U_obs=U_obs,
        wind_height_m=cfg["wind_height_m"],
        air_temp_c=air_temp_c,
        pressure_value=pressure_value,
    )

    out = {
        "thermocline_depth_m": zT,
        "u_star": ustar,
        "rho_epi": np.nan,
        "rho_hyp": np.nan,
        "delta_rho": np.nan,
        "schmidt_stability_J_m2": np.nan,
        "Wedderburn": np.nan,
        "Lake_number": np.nan,
    }

    if T_grid is None:
        return out

    rho_grid = water_density_freshwater(T_grid)

    V, z_cv = lake_volume_and_center(depth_grid, area_grid)
    rho_bar = volume_weighted_mean(rho_grid, depth_grid, area_grid)

    if np.isfinite(rho_bar) and np.isfinite(z_cv):
        St = (
            G / A0
            * integrate_trapz(
                (depth_grid - z_cv)
                * (rho_grid - rho_bar)
                * area_grid,
                depth_grid,
            )
        )
        out["schmidt_stability_J_m2"] = St
    else:
        St = np.nan

    if not np.isfinite(zT):
        return out
    if zT <= 0 or zT >= max_depth:
        return out
    if not np.isfinite(ustar) or ustar <= 0:
        return out

    epi_mask = depth_grid <= zT
    hyp_mask = depth_grid > zT

    rho_epi = volume_weighted_mean(
        rho_grid, depth_grid, area_grid, epi_mask
    )
    rho_hyp = volume_weighted_mean(
        rho_grid, depth_grid, area_grid, hyp_mask
    )
    delta_rho = rho_hyp - rho_epi

    out["rho_epi"] = rho_epi
    out["rho_hyp"] = rho_hyp
    out["delta_rho"] = delta_rho

    if not np.isfinite(delta_rho) or delta_rho <= 0:
        return out

    g_reduced = G * delta_rho / RHO_W_REF
    h_e = zT

    W = g_reduced * h_e**2 / (ustar**2 * fetch)
    out["Wedderburn"] = W

    denom = (
        rho_hyp
        * ustar**2
        * np.sqrt(A0)
        * (1.0 - z_cv / max_depth)
    )

    if np.isfinite(St) and St > 0 and denom > 0:
        LN = St * (1.0 - zT / max_depth) / denom
        out["Lake_number"] = LN

    return out


# ============================================================
# 4. Continuous corrected stability time series
# ============================================================

def calculate_corrected_continuous(site_data, met):
    site = site_data["site"]
    cfg = SITE_CONFIGS[site]
    time = site_data["time"]
    depths = site_data["depths"]
    temp = site_data["temp"]
    valid = site_data["valid"]

    depth_grid, area_grid = build_bathymetry(cfg, dz=0.01)

    rows = []

    for i in range(len(time)):
        profile_valid = bool(valid[i].all())

        if profile_valid:
            temps = temp[i]
        else:
            # Keep the fixed-depth rule. Do not silently compute with changing layers.
            temps = np.full(len(depths), np.nan)

        indices = calculate_stability_indices_one_time(
            temps=temps,
            depths=depths,
            U_obs=met["wind"][i],
            cfg=cfg,
            depth_grid=depth_grid,
            area_grid=area_grid,
            air_temp_c=(
                met["air_temp"][i]
                if met["air_temp"] is not None
                else None
            ),
            pressure_value=(
                met["pressure"][i]
                if met["pressure"] is not None
                else None
            ),
        )

        if profile_valid:
            std_obs = float(np.std(temp[i], ddof=0))
        else:
            std_obs = np.nan

        rows.append({
            "site": site,
            "site_label": SITE_LABELS[site],
            "time": time[i],
            "fixed_profile_complete": profile_valid,
            "STD_obs": std_obs,
            "wind_obs": met["wind"][i],
            "air_temp_obs": (
                met["air_temp"][i]
                if met["air_temp"] is not None
                else np.nan
            ),
            "pressure_obs": (
                met["pressure"][i]
                if met["pressure"] is not None
                else np.nan
            ),
            **indices,
        })

    return pd.DataFrame(rows)


# ============================================================
# 5. Final-event extraction
# ============================================================

def normalize_event_table(site_data):
    events = site_data["events"].copy()
    time = site_data["time"]

    needed = [
        "event_id",
        "start_index",
        "peak_index",
        "min_index",
        "end_index",
        "drop_degC",
    ]
    for col in needed:
        require(col in events.columns, f"{site_data['site']}: missing event column {col}")

    rows = []
    for ev in events.itertuples(index=False):
        i = int(ev.start_index)
        p = int(ev.peak_index)
        q = int(ev.min_index)
        j = int(ev.end_index)

        require(0 <= i <= p < q <= j < len(time),
                f"{site_data['site']}/{ev.event_id}: invalid event indices")

        rows.append({
            "event_id": str(ev.event_id),
            "event_start_time": time[i],
            "obs_peak_time": time[p],
            "obs_min_time": time[q],
            "event_end_time": time[j],
            "start_idx": i,
            "peak_idx": p,
            "min_idx": q,
            "end_idx": j,
            "obs_drop_abs": float(ev.drop_degC),
            "collapse_duration_h": (
                time[q] - time[p]
            ).total_seconds() / 3600.0,
        })

    return pd.DataFrame(rows)


def extract_event_stability(site_data, continuous_df):
    events = normalize_event_table(site_data)
    rows = []

    time = site_data["time"]

    for ev in events.itertuples(index=False):
        p = int(ev.peak_idx)
        q = int(ev.min_idx)
        i = int(ev.start_idx)

        # Original notebook pre3h window: from peak - 3 h through peak.
        peak_time = pd.Timestamp(time[p])
        pre_start_time = peak_time - pd.Timedelta(hours=3)

        pre = continuous_df[
            (continuous_df["time"] >= pre_start_time)
            & (continuous_df["time"] <= peak_time)
        ].copy()

        row = {
            "site": site_data["site"],
            "site_label": SITE_LABELS[site_data["site"]],
            "event_id": ev.event_id,
            "event_start_time": ev.event_start_time,
            "obs_peak_time": ev.obs_peak_time,
            "obs_min_time": ev.obs_min_time,
            "event_end_time": ev.event_end_time,
            "start_idx": i,
            "peak_idx": p,
            "min_idx": q,
            "end_idx": int(ev.end_idx),
            "obs_drop_abs": float(ev.obs_drop_abs),
            "collapse_duration_h": float(ev.collapse_duration_h),
            "pre3h_total_records": int(len(pre)),
            "pre3h_complete_profile_records": int(pre["fixed_profile_complete"].sum()),
        }

        for suffix, idx in [
            ("start", i),
            ("peak", p),
            ("min", q),
        ]:
            r = continuous_df.iloc[idx]
            for col in [
                "thermocline_depth_m",
                "u_star",
                "schmidt_stability_J_m2",
                "Wedderburn",
                "Lake_number",
            ]:
                row[f"{col}_{suffix}"] = r[col]

        for col in [
            "thermocline_depth_m",
            "u_star",
            "schmidt_stability_J_m2",
            "Wedderburn",
            "Lake_number",
        ]:
            row[f"{col}_pre3h_median"] = pre[col].median(skipna=True)
            row[f"{col}_pre3h_valid_n"] = int(pre[col].notna().sum())

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 6. Summaries + figures
# ============================================================

def q25(x):
    return x.quantile(0.25)


def q75(x):
    return x.quantile(0.75)


def summarize_events(event_df):
    rows = []
    for site in SITE_ORDER:
        sub = event_df[event_df["site"] == site].copy()

        row = {
            "site": site,
            "site_label": SITE_LABELS[site],
            "n_events": int(len(sub)),
            "valid_W_peak_n": int(sub["Wedderburn_peak"].notna().sum()),
            "valid_Lake_peak_n": int(sub["Lake_number_peak"].notna().sum()),
            "obs_drop_median": sub["obs_drop_abs"].median(),
            "obs_drop_Q25": sub["obs_drop_abs"].quantile(0.25),
            "obs_drop_Q75": sub["obs_drop_abs"].quantile(0.75),
        }

        for col in [
            "Wedderburn_peak",
            "Lake_number_peak",
            "Wedderburn_pre3h_median",
            "Lake_number_pre3h_median",
            "thermocline_depth_m_peak",
            "u_star_peak",
            "schmidt_stability_J_m2_peak",
        ]:
            row[f"{col}_median"] = sub[col].median()
            row[f"{col}_Q25"] = sub[col].quantile(0.25)
            row[f"{col}_Q75"] = sub[col].quantile(0.75)

        rows.append(row)

    return pd.DataFrame(rows)


def plot_one(event_df, column, ylabel, filename_base, out_dir):
    labels = []
    data = []

    for site in SITE_ORDER:
        vals = (
            event_df.loc[event_df["site"] == site, column]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(float)
        )
        vals = vals[vals > 0]
        if len(vals):
            labels.append(SITE_LABELS[site])
            data.append(vals)

    require(data, f"No valid data to plot for {column}")

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.axhline(1.0, linestyle="--", linewidth=1.0, color="black")
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    png = out_dir / f"{filename_base}.png"
    pdf = out_dir / f"{filename_base}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return png, pdf


def write_report(summary, met_meta, out_dir):
    lines = [
        "# Fixed-depth event stability update",
        "",
        "This run preserves the original Figure-S.ipynb stability formulas and updates only",
        "the temperature-depth protocol and event lists.",
        "",
        "## Final event catalogs",
        "",
        "- Shiode: 91 strict events; fixed 0.25–2.25 m, 11 depths.",
        "- Yamanokami: 13 strict events; fixed 0.40–2.20 m, 10 depths.",
        "- Shin: 0 original-threshold events, so no event stability boxplot is needed.",
        "",
        "## Important interpretation boundary",
        "",
        "Wedderburn and Lake numbers remain relative diagnostics because effective fetch and",
        "morphometry are simplified. Temperature outside the retained sensor range is filled",
        "with the nearest retained observed temperature only for the density/morphometry",
        "integration, exactly following the original notebook approach. For Yamanokami this",
        "means the 0–0.40 m interval is represented by the 0.40 m observed temperature.",
        "",
        "Do not use these results as proof of complete vertical mixing or a unique mechanism.",
        "",
        "## Summary",
        "",
    ]

    for r in summary.itertuples(index=False):
        lines += [
            f"### {r.site_label}",
            f"- events: {r.n_events}",
            f"- valid W at peak: {r.valid_W_peak_n}",
            f"- valid Lake number at peak: {r.valid_Lake_peak_n}",
            f"- median W at peak: {r.Wedderburn_peak_median:.6g}"
              if np.isfinite(r.Wedderburn_peak_median) else "- median W at peak: NA",
            f"- median Lake number at peak: {r.Lake_number_peak_median:.6g}"
              if np.isfinite(r.Lake_number_peak_median) else "- median Lake number at peak: NA",
            f"- median Schmidt stability at peak: {r.schmidt_stability_J_m2_peak_median:.6g} J m-2"
              if np.isfinite(r.schmidt_stability_J_m2_peak_median)
              else "- median Schmidt stability at peak: NA",
            "",
        ]

    lines += [
        "## Meteorological inputs",
        "",
    ]
    for site in SITE_ORDER:
        x = met_meta[site]
        lines += [
            f"### {SITE_LABELS[site]}",
            f"- aligned CSV: {x['csv_path']}",
            f"- columns: {x['columns']}",
            f"- missing wind records on corrected temperature timeline: {x['n_missing_wind']}",
            "",
        ]

    (out_dir / "REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 7. Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--shiode-events", type=Path, default=None)
    ap.add_argument("--yama-shin-events", type=Path, default=None)
    ap.add_argument("--shiode-csv", type=Path, default=None)
    ap.add_argument("--yamanokami-csv", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    require(root.is_dir(), f"Root does not exist: {root}")

    shiode_zip = find_one(root, SHIODE_ARCHIVE, args.shiode_events)
    ys_zip = find_one(root, YS_ARCHIVE, args.yama_shin_events)

    csv_paths = {
        "shiode": find_one(
            root,
            SITE_CONFIGS["shiode"]["csv_name"],
            args.shiode_csv,
        ),
        "yamanokami": find_one(
            root,
            SITE_CONFIGS["yamanokami"]["csv_name"],
            args.yamanokami_csv,
        ),
    }

    out_dir = (
        args.out.expanduser().resolve()
        if args.out
        else root / "revision_R2_4" / "fixed_depth_stability"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_continuous = []
    all_events = []
    met_meta = {}

    for site in SITE_ORDER:
        print("\n============================================")
        print("Processing", SITE_LABELS[site])
        print("============================================")

        site_data = load_corrected_site(site, shiode_zip, ys_zip)
        met = load_meteorology(csv_paths[site], site_data["time"])

        met_meta[site] = {
            "csv_path": str(csv_paths[site]),
            "columns": met["columns"],
            "n_missing_wind": met["n_missing_wind"],
            "n_missing_air_temp": met["n_missing_air_temp"],
            "n_missing_pressure": met["n_missing_pressure"],
        }

        continuous = calculate_corrected_continuous(site_data, met)
        events = extract_event_stability(site_data, continuous)

        require(
            len(events) == SITE_CONFIGS[site]["expected_events"],
            f"{site}: event extraction count changed",
        )

        all_continuous.append(continuous)
        all_events.append(events)

    continuous_all = pd.concat(all_continuous, ignore_index=True)
    event_all = pd.concat(all_events, ignore_index=True)
    summary = summarize_events(event_all)

    continuous_all.to_csv(
        out_dir / "TableS_continuous_stability_fixeddepth.csv",
        index=False,
        encoding="utf-8-sig",
    )
    event_all.to_csv(
        out_dir / "TableS_event_stability_fixeddepth.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        out_dir / "TableS_event_stability_summary_fixeddepth.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_one(
        event_all,
        "Wedderburn_peak",
        "Wedderburn number at observed STD peak",
        "Figure_D1_event_wedderburn_fixeddepth",
        out_dir,
    )
    plot_one(
        event_all,
        "Lake_number_peak",
        "Lake number at observed STD peak",
        "Figure_D2_event_lake_number_fixeddepth",
        out_dir,
    )

    write_report(summary, met_meta, out_dir)

    settings = {
        "constants": {
            "G": G,
            "RHO_W_REF": RHO_W_REF,
            "RHO_AIR_DEFAULT": RHO_AIR_DEFAULT,
            "CD_DEFAULT": CD_DEFAULT,
            "Z0_WATER": Z0_WATER,
            "USE_WIND_10M_CORRECTION": USE_WIND_10M_CORRECTION,
        },
        "site_configs": {
            site: {
                k: (
                    v.tolist()
                    if isinstance(v, np.ndarray)
                    else v
                )
                for k, v in cfg.items()
            }
            for site, cfg in SITE_CONFIGS.items()
        },
        "temperature_source": {
            "shiode": SHIODE_ARCHIVE,
            "yamanokami": YS_ARCHIVE,
        },
        "meteorology_source": {
            site: str(csv_paths[site])
            for site in SITE_ORDER
        },
        "note": (
            "Same stability formulas/morphometry as original Figure-S.ipynb; "
            "updated fixed-depth observations and final event catalogs."
        ),
    }
    (out_dir / "settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return_zip = out_dir / "JoHRS_R2_4_stability_return.zip"
    with zipfile.ZipFile(return_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in [
            "REPORT_zh.md",
            "TableS_event_stability_fixeddepth.csv",
            "TableS_event_stability_summary_fixeddepth.csv",
            "Figure_D1_event_wedderburn_fixeddepth.png",
            "Figure_D1_event_wedderburn_fixeddepth.pdf",
            "Figure_D2_event_lake_number_fixeddepth.png",
            "Figure_D2_event_lake_number_fixeddepth.pdf",
            "settings.json",
        ]:
            p = out_dir / name
            if p.exists():
                z.write(p, p.name)

    print("\nCOMPLETE")
    print(summary.to_string(index=False))
    print("\nUPLOAD THIS FILE:")
    print(" ", return_zip)


if __name__ == "__main__":
    main()
