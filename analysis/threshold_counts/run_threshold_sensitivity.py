#!/usr/bin/env python3
"""
JoHRS Reviewer 2-3: threshold sensitivity of event counts AND four-model drop ratios.

This script extends the existing event-count sensitivity analysis. It does NOT train
models, run inference, run GLM, select members, or tune thresholds.

Threshold settings are exactly the manuscript's existing sensitivity settings:
-50%, -25%, Original, +25%, +50%, applied simultaneously to:
  STD(start) threshold = 0.50 degC
  3-h drop threshold  = 0.25 degC

Corrected fixed-depth observation protocol:
  Shiode      : 0.25--2.25 m, 11 depths, complete 3-h windows
  Yamanokami  : 0.40--2.20 m, 10 depths, complete 3-h windows
  Shin        : 0.25--3.45 m, 17 depths, complete 3-h windows

Primary model event metric follows the manuscript:
  1) average the fixed ensemble members' temperature predictions first;
  2) calculate vertical STD at each time;
  3) within each OBSERVATION-defined event window:
       model amplitude = max(model STD) - min(model STD)
       drop ratio = model amplitude / observed amplitude
  The model extrema are not required to occur in the correct temporal order.
  Temporal direction is recorded separately.

Outputs include event counts, drop-ratio distributions, median absolute ratio error
(|drop ratio - 1|), and a threshold-wise ranking based on that error.

Dependencies:
  numpy, pandas, matplotlib
No tabulate dependency.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import tarfile
import traceback
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERSION = "1.0.0"
WINDOW_STEPS = 18
STEP_MINUTES = 10
BASE_STD_THRESHOLD = 0.50
BASE_DROP_THRESHOLD = 0.25

SETTINGS = [
    ("-50%", 0.50),
    ("-25%", 0.75),
    ("Original", 1.00),
    ("+25%", 1.25),
    ("+50%", 1.50),
]
MODEL_ORDER = ["GLM", "LSTM", "rSSM", "PGNN"]
SITE_ORDER = ["Shiode", "Yamanokami", "Shin"]

EXPECTED_BASELINE_COUNTS = {
    "Shiode": 91,
    "Yamanokami": 13,
    "Shin": 0,
}

SHIODE_DEPTHS = np.arange(0.25, 2.2501, 0.20)
YAMA_DEPTHS = np.arange(0.40, 2.2001, 0.20)
SHIN_DEPTHS = np.arange(0.25, 3.4501, 0.20)

EXPECTED_FILES = {
    "shiode_obs": "Shiode_R1_5_step1A_return.zip",
    "shiode_models": "Shiode_R1_5_step1B_return.zip",
    "obs_yama_shin": "Shin_Yamanokami_R1_5_events_return.zip",
    "yama_ml": "Yamanokami_ML_original_batch_20261024_170442_830097.zip",
    "yama_pgnn_added": "Yamanokami_PGNN_added_seeds_20261024_173009.zip",
    "yama_glm": "GLM_yamanokami_return.tar.gz",
    "shin_ml": "Shin_ML_retrain_return.tar.gz",
    "shin_glm": "GLM_shin_test_return.tar.gz",
}

# Frozen membership already established in previous revision work.
YAMA_LSTM = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)
YAMA_RSSM = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)
YAMA_PGNN = (3, 4, 5, 6, 7, 9, 14, 15, 16, 20)


def require(cond, msg):
    if not bool(cond):
        raise ValueError(msg)


def normalize_zip_name(name):
    return name.replace("\\", "/")


def zip_map(z):
    out = {}
    for info in z.infolist():
        if info.is_dir():
            continue
        key = normalize_zip_name(info.filename)
        require(key not in out, f"Duplicate normalized ZIP path: {key}")
        out[key] = info.filename
    return out


def read_npz_bytes(blob):
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def as_time(values):
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.integer):
        out = arr.astype("datetime64[ns]")
    else:
        out = pd.to_datetime(arr, errors="raise").to_numpy(dtype="datetime64[ns]")
    require(len(out) > 1, "Time axis too short.")
    require(np.all(np.diff(out.astype("int64")) == 10 * 60 * 10**9),
            "Expected regular 10-minute time grid.")
    return out


def exact_time_index(source_time, target_time):
    src = as_time(source_time)
    tgt = as_time(target_time)
    ix = np.searchsorted(src, tgt)
    require((ix < len(src)).all(), "Target time exceeds source time range.")
    require(np.array_equal(src[ix], tgt), "Missing exact target timestamps.")
    return ix


def exact_depth_index(source_depths, target_depths, atol=1e-5):
    src = np.asarray(source_depths, float)
    tgt = np.asarray(target_depths, float)
    idx = np.array([int(np.argmin(np.abs(src - d))) for d in tgt], dtype=int)
    require(len(set(idx.tolist())) == len(idx), "Duplicate depth mapping.")
    require(np.allclose(src[idx], tgt, rtol=0, atol=atol),
            f"Cannot exactly map depths {tgt.tolist()} from {src.tolist()}.")
    return idx


def find_one(root, filename, explicit=None):
    if explicit:
        p = Path(explicit).expanduser().resolve()
        require(p.is_file(), f"Missing explicit input: {p}")
        return p

    candidates = []
    here = Path(__file__).resolve().parent
    for p in [here / "data" / filename, root / filename]:
        if p.is_file():
            candidates.append(p.resolve())
    if root.exists():
        for p in root.rglob(filename):
            candidates.append(p.resolve())

    candidates = sorted(set(candidates))
    require(candidates, f"Cannot find {filename} below {root}.")
    # Duplicate copies are okay only when byte-identical.
    sizes = {(p.stat().st_size) for p in candidates}
    if len(candidates) > 1:
        # avoid hashing huge archives unless needed; same filename/size is still ambiguous
        print(f"NOTE: found {len(candidates)} copies of {filename}; using first:")
        for p in candidates:
            print("  ", p)
    return candidates[0]


def row_std(temp):
    temp = np.asarray(temp, float)
    require(temp.ndim == 2, "Temperature must be [time, depth].")
    require(np.isfinite(temp).all(), "Nonfinite values in strict event temperature window.")
    return np.std(temp, axis=1, ddof=0)


def forward_complete(profile_complete):
    x = np.asarray(profile_complete, bool)
    width = WINDOW_STEPS + 1
    out = np.zeros(len(x), dtype=bool)
    if len(x) >= width:
        c = np.r_[0, np.cumsum(x.astype(np.int64))]
        out[:len(x)-width+1] = (c[width:] - c[:-width]) == width
    return out


def detect_events(std_series, eligible_start, std_thr, drop_thr):
    """
    Exact event logic used in the earlier R1-5 scripts:
      - start STD >= std_thr
      - inside next 3h, global max-min >= drop_thr
      - observed minimum occurs after observed maximum
      - strict completeness gate is applied BEFORE greedy merge
      - greedy merge: next accepted start > previous start + 18 samples
    """
    sd = np.asarray(std_series, float)
    eligible = np.asarray(eligible_start, bool)
    require(sd.shape == eligible.shape, "Event eligibility shape mismatch.")

    cand = np.zeros(len(sd), dtype=bool)
    details = {}

    for i in range(len(sd)):
        if not eligible[i]:
            continue
        if not np.isfinite(sd[i]) or sd[i] < std_thr:
            continue

        j = i + WINDOW_STEPS
        if j >= len(sd):
            continue
        w = sd[i:j+1]
        if not np.isfinite(w).all():
            continue

        p = int(np.argmax(w))
        q = int(np.argmin(w))
        drop = float(w[p] - w[q])

        if drop >= drop_thr and q > p:
            cand[i] = True
            details[i] = (i+p, i+q, j, drop)

    starts = []
    last = -10**12
    for i in np.flatnonzero(cand):
        i = int(i)
        if i > last + WINDOW_STEPS:
            starts.append(i)
            last = i

    events = []
    for eid, i in enumerate(starts, 1):
        p, q, j, drop = details[i]
        events.append({
            "event_id": eid,
            "start_index": i,
            "peak_index": p,
            "min_index": q,
            "end_index": j,
            "obs_amplitude_degC": drop,
        })
    return pd.DataFrame(events), cand


@dataclass
class SiteData:
    site: str
    time: np.ndarray
    depths: np.ndarray
    obs_temp: np.ndarray
    obs_valid: np.ndarray
    obs_std: np.ndarray
    complete_start: np.ndarray
    model_std: dict
    common_model_valid: np.ndarray


def ensemble_std_from_temperature_members(pred_members, valid_members):
    """
    First average temperatures across frozen members, then calculate vertical STD.
    Also require all fixed members to have finite prediction coverage at a cell.
    """
    require(len(pred_members) == len(valid_members) and len(pred_members) > 0,
            "Member arrays missing.")
    stack = np.stack(pred_members, axis=0)
    vstack = np.stack(valid_members, axis=0).astype(bool)
    cell_valid_all = np.all(vstack, axis=0) & np.all(np.isfinite(stack), axis=0)

    mean_temp = np.mean(stack, axis=0)
    series = np.full(mean_temp.shape[0], np.nan)
    complete_time = np.all(cell_valid_all, axis=1)
    if complete_time.any():
        series[complete_time] = np.std(mean_temp[complete_time], axis=1, ddof=0)
    return series, cell_valid_all


# -----------------------------
# SHIODE: use already-generated fixed11 ensemble structure series
# -----------------------------
def load_shiode(obs_zip, model_zip):
    with zipfile.ZipFile(obs_zip) as z:
        names = zip_map(z)
        d = read_npz_bytes(z.read(names["fixed_depth_observation_reference.npz"]))

    time = as_time(d["time"])
    depths = np.asarray(d["depths"], float)
    require(np.allclose(depths, SHIODE_DEPTHS, atol=1e-8, rtol=0), "Shiode depths changed.")
    obs_temp = np.asarray(d["obs_data_quality"], float)
    obs_valid = np.asarray(d["valid_mask"], bool)
    complete_profile = np.asarray(d["strict_complete_profile"], bool)
    complete_start = np.asarray(d["strict_window_start_eligible"], bool)
    require(np.array_equal(complete_profile, obs_valid.all(axis=1)), "Shiode completeness mismatch.")
    require(np.array_equal(complete_start, forward_complete(complete_profile)), "Shiode 3h gate mismatch.")

    obs_std = np.full(len(time), np.nan)
    ok = complete_profile
    obs_std[ok] = np.std(obs_temp[ok], axis=1, ddof=0)

    with zipfile.ZipFile(model_zip) as z:
        names = zip_map(z)
        s = read_npz_bytes(z.read(names["ensemble_structure_series.npz"]))
        m = read_npz_bytes(z.read(names["common_scoring_masks.npz"]))

    require(np.array_equal(as_time(s["time"]), time), "Shiode model-series time mismatch.")
    require(np.array_equal(as_time(m["time"]), time), "Shiode mask time mismatch.")

    model_std = {model: np.asarray(s[f"{model}_STD_fixed11"], float) for model in MODEL_ORDER}
    # common complete prediction support from previous R1-5 analysis
    if "strict_common_time" in m:
        common_time = np.asarray(m["strict_common_time"], bool)
    elif "B_common_time" in m:
        common_time = np.asarray(m["B_common_time"], bool)
    else:
        # derive conservatively: all four ensemble STD series finite
        common_time = np.logical_and.reduce([np.isfinite(model_std[x]) for x in MODEL_ORDER])

    common_model_valid = np.repeat(common_time[:, None], len(depths), axis=1)

    return SiteData(
        "Shiode", time, depths, obs_temp, obs_valid, obs_std,
        complete_start, model_std, common_model_valid
    )


# -----------------------------
# YAMANOKAMI
# -----------------------------
def yama_ml_member_from_npz(d, target_time, target_depths, obs_temp, obs_valid):
    for k in ("time", "pred_sel", "obs_sel", "counts", "matched_obs_depths"):
        require(k in d, f"Yamanokami ML NPZ missing {k}")
    ti = exact_time_index(d["time"], target_time)
    di = exact_depth_index(d["matched_obs_depths"], target_depths)

    pred = np.asarray(d["pred_sel"], float)[ti][:, di]
    counts = np.asarray(d["counts"], float)
    # counts may be in model-depth coordinates; previous audited NPZs use matched model columns.
    if counts.ndim == 2 and counts.shape[1] == np.asarray(d["pred_sel"]).shape[1]:
        cov = counts[ti][:, di] > 0
    else:
        # fallback used only when counts' second axis differs; prediction finite is still required
        cov = np.isfinite(pred)
    valid = cov & np.isfinite(pred)

    archived_obs = np.asarray(d["obs_sel"], float)[ti][:, di]
    use = obs_valid & np.isfinite(archived_obs)
    require(use.any(), "No Yamanokami ML/observation overlap.")
    require(np.max(np.abs(archived_obs[use] - obs_temp[use])) <= 2e-5,
            "Yamanokami ML observation reference mismatch.")
    return pred, valid


def load_yamanokami(events_zip, ml_zip, pgnn_added_zip, glm_tar):
    with zipfile.ZipFile(events_zip) as z:
        names = zip_map(z)
        d = read_npz_bytes(z.read(names["yamanokami/core10/observations_and_masks.npz"]))

    time = as_time(d["time_ns"])
    depths = np.asarray(d["depths"], float)
    require(np.allclose(depths, YAMA_DEPTHS, atol=1e-8, rtol=0), "Yamanokami core10 changed.")
    obs_temp = np.asarray(d["temperature_raw"], float)
    obs_valid = np.asarray(d["valid_data_only"], bool)
    complete_profile = np.asarray(d["complete_profile"], bool)
    complete_start = np.asarray(d["complete_3h_start"], bool)
    require(np.array_equal(complete_profile, obs_valid.all(axis=1)), "Yamanokami profile gate mismatch.")
    require(np.array_equal(complete_start, forward_complete(complete_profile)), "Yamanokami 3h gate mismatch.")

    obs_std = np.full(len(time), np.nan)
    obs_std[complete_profile] = np.std(obs_temp[complete_profile], axis=1, ddof=0)

    pred_by_model = defaultdict(list)
    valid_by_model = defaultdict(list)

    # Main archive
    with zipfile.ZipFile(ml_zip) as z:
        names = zip_map(z)
        audit = pd.read_csv(io.BytesIO(z.read(names["_audit/members.csv"])))
        for r in audit.itertuples(index=False):
            model_raw = str(r.model)
            seed = int(r.actual_seed)
            keep = (
                (model_raw == "lstm" and seed in YAMA_LSTM)
                or (model_raw == "rssm" and seed in YAMA_RSSM)
                or (model_raw == "pgnn" and seed in YAMA_PGNN and seed not in (14, 15, 16, 20))
            )
            if not keep:
                continue
            model = {"lstm":"LSTM","rssm":"rSSM","pgnn":"PGNN"}[model_raw]
            key = "evidence/" + str(r.prediction).replace("\\", "/")
            require(key in names, f"Missing Yamanokami prediction {key}")
            pred, valid = yama_ml_member_from_npz(
                read_npz_bytes(z.read(names[key])), time, depths, obs_temp, obs_valid
            )
            pred_by_model[model].append(pred)
            valid_by_model[model].append(valid)

    # Added final PGNN members
    with zipfile.ZipFile(pgnn_added_zip) as z:
        names = zip_map(z)
        for seed in (14, 15, 16, 20):
            key = f"seed{seed}/lstm_overlap_avg_predictions.npz"
            require(key in names, f"Missing Yamanokami added PGNN {key}")
            pred, valid = yama_ml_member_from_npz(
                read_npz_bytes(z.read(names[key])), time, depths, obs_temp, obs_valid
            )
            pred_by_model["PGNN"].append(pred)
            valid_by_model["PGNN"].append(valid)

    for model in ("LSTM", "rSSM", "PGNN"):
        require(len(pred_by_model[model]) == 10, f"Yamanokami {model}: expected 10 fixed members.")

    # GLM
    glm_preds = []
    glm_valid = []
    with tarfile.open(glm_tar, "r:gz") as t:
        members = {m.name:m for m in t.getmembers() if m.isfile()}
        def suffix(s):
            hits = [n for n in members if n == s or n.endswith("/"+s)]
            require(len(hits) == 1, f"Cannot uniquely find Yamanokami GLM {s}")
            return hits[0]
        def read(name):
            f = t.extractfile(members[name])
            require(f is not None, f"Cannot read {name}")
            return f.read()

        summary = json.loads(read(suffix("summary.json")).decode("utf-8-sig"))
        runs = list(summary["frozen_top10"])
        require(len(runs) == 10, "Yamanokami GLM frozen ensemble not 10.")

        ref = read_npz_bytes(read(suffix("reference/test_observation_reference.npz")))
        ti = exact_time_index(ref["time"], time)
        di = exact_depth_index(ref["depths"], depths)
        robs = np.asarray(ref["obs"], float)[ti][:, di]
        use = obs_valid & np.isfinite(robs)
        require(use.any() and np.max(np.abs(robs[use] - obs_temp[use])) <= 2e-5,
                "Yamanokami GLM observation reference mismatch.")

        for run in runs:
            dd = read_npz_bytes(read(suffix(f"predictions/{run}.npz")))
            ti = exact_time_index(dd["time"], time)
            di = exact_depth_index(dd["depths"], depths)
            p = np.asarray(dd["pred"], float)[ti][:, di]
            glm_preds.append(p)
            glm_valid.append(np.isfinite(p))

    pred_by_model["GLM"] = glm_preds
    valid_by_model["GLM"] = glm_valid

    model_std = {}
    common_cell = np.ones(obs_temp.shape, dtype=bool)
    for model in MODEL_ORDER:
        series, cell_valid = ensemble_std_from_temperature_members(
            pred_by_model[model], valid_by_model[model]
        )
        model_std[model] = series
        common_cell &= cell_valid

    return SiteData(
        "Yamanokami", time, depths, obs_temp, obs_valid, obs_std,
        complete_start, model_std, common_cell
    )


# -----------------------------
# SHIN
# -----------------------------
def load_shin(events_zip, ml_tar, glm_tar):
    with zipfile.ZipFile(events_zip) as z:
        names = zip_map(z)
        d = read_npz_bytes(z.read(names["shin/original17/observations_and_masks.npz"]))

    time = as_time(d["time_ns"])
    depths = np.asarray(d["depths"], float)
    require(np.allclose(depths, SHIN_DEPTHS, atol=1e-8, rtol=0), "Shin fixed17 changed.")
    obs_temp = np.asarray(d["temperature_raw"], float)
    obs_valid = np.asarray(d["valid_data_only"], bool)
    complete_profile = np.asarray(d["complete_profile"], bool)
    complete_start = np.asarray(d["complete_3h_start"], bool)
    require(np.array_equal(complete_profile, obs_valid.all(axis=1)), "Shin profile gate mismatch.")
    require(np.array_equal(complete_start, forward_complete(complete_profile)), "Shin 3h gate mismatch.")

    obs_std = np.full(len(time), np.nan)
    obs_std[complete_profile] = np.std(obs_temp[complete_profile], axis=1, ddof=0)

    pred_by_model = defaultdict(list)
    valid_by_model = defaultdict(list)

    # Corrected retrained ML outputs
    with tarfile.open(ml_tar, "r:gz") as t:
        members = {m.name:m for m in t.getmembers() if m.isfile()}

        # Read frozen member registries rather than guessing member names.
        for model_dir, model_label in [("lstm","LSTM"),("rssm","rSSM"),("pgnn","PGNN")]:
            registry_name = f"models/{model_dir}/frozen_members.json"
            require(registry_name in members, f"Missing Shin registry {registry_name}")
            reg = json.loads(t.extractfile(members[registry_name]).read().decode("utf-8-sig"))
            # Support either list or dict containers.
            if isinstance(reg, dict):
                rows = reg.get("members", reg.get("frozen_members", []))
            else:
                rows = reg
            require(len(rows) == 10, f"Shin {model_label}: expected 10 frozen members.")

            # Locate each member prediction by member_id / actual_seed in NPZ metadata.
            candidates = [
                n for n in members
                if n.startswith(f"models/{model_dir}/member_") and n.endswith("/test_predictions.npz")
            ]
            require(len(candidates) == 10, f"Shin {model_label}: expected 10 test prediction NPZs.")

            for name in sorted(candidates):
                f = t.extractfile(members[name]); require(f is not None, f"Cannot read {name}")
                dd = read_npz_bytes(f.read())
                ti = exact_time_index(dd["time"], time)
                di = exact_depth_index(dd["depths"], depths)
                p = np.asarray(dd["pred"], float)[ti][:, di]
                cov = np.asarray(dd["prediction_coverage"], bool)[ti][:, di]
                valid = cov & np.isfinite(p)

                archived_obs = np.asarray(dd["obs"], float)[ti][:, di]
                use = obs_valid & np.isfinite(archived_obs)
                require(use.any() and np.max(np.abs(archived_obs[use]-obs_temp[use])) <= 2e-5,
                        f"Shin {model_label} observation reference mismatch.")

                pred_by_model[model_label].append(p)
                valid_by_model[model_label].append(valid)

    for model in ("LSTM","rSSM","PGNN"):
        require(len(pred_by_model[model]) == 10, f"Shin {model}: expected 10 members.")

    # Frozen GLM top10
    with tarfile.open(glm_tar, "r:gz") as t:
        b = t.extractfile(t.getmember("test_predictions.npz")).read()
        g = read_npz_bytes(b)

    gtime = as_time(g["time"])
    ti = exact_time_index(gtime, time)
    di = exact_depth_index(g["depths"], depths)
    ids = [str(x) for x in g["run_ids"].tolist()]
    selected = [str(x) for x in g["new_top10"].tolist()]
    require(len(selected) == 10 and len(set(selected)) == 10, "Shin GLM top10 invalid.")
    run_index = [ids.index(x) for x in selected]
    gp = np.asarray(g["pred"], float)[run_index][:, ti][:, :, di]  # member,time,depth

    # Numpy advanced indexing above can reorder axes; normalize robustly.
    if gp.shape != (10, len(time), len(depths)):
        gp = np.asarray(g["pred"], float)[run_index][:, ti, :][:, :, di]
    require(gp.shape == (10, len(time), len(depths)), f"Unexpected Shin GLM shape {gp.shape}")

    gob = np.asarray(g["obs"], float)[ti][:, di]
    use = obs_valid & np.isfinite(gob)
    require(use.any() and np.max(np.abs(gob[use]-obs_temp[use])) <= 2e-5,
            "Shin GLM observation reference mismatch.")

    pred_by_model["GLM"] = [gp[k] for k in range(10)]
    valid_by_model["GLM"] = [np.isfinite(gp[k]) for k in range(10)]

    model_std = {}
    common_cell = np.ones(obs_temp.shape, dtype=bool)
    for model in MODEL_ORDER:
        series, cell_valid = ensemble_std_from_temperature_members(
            pred_by_model[model], valid_by_model[model]
        )
        model_std[model] = series
        common_cell &= cell_valid

    return SiteData(
        "Shin", time, depths, obs_temp, obs_valid, obs_std,
        complete_start, model_std, common_cell
    )


def event_model_metrics(site_data, events, setting_label):
    rows = []
    for e in events.itertuples(index=False):
        i = int(e.start_index); j = int(e.end_index)
        obs_amp = float(e.obs_amplitude_degC)

        # Require common prediction support for all models over the whole event window.
        common_ok = bool(site_data.common_model_valid[i:j+1].all())

        for model in MODEL_ORDER:
            base = {
                "site": site_data.site,
                "threshold_setting": setting_label,
                "event_id": int(e.event_id),
                "start_time": str(pd.Timestamp(site_data.time[i])),
                "obs_peak_time": str(pd.Timestamp(site_data.time[int(e.peak_index)])),
                "obs_min_time": str(pd.Timestamp(site_data.time[int(e.min_index)])),
                "obs_amplitude_degC": obs_amp,
                "common_model_coverage": common_ok,
                "model": model,
            }
            if not common_ok:
                base.update({
                    "model_amplitude_degC": np.nan,
                    "drop_ratio": np.nan,
                    "absolute_ratio_error": np.nan,
                    "absolute_amplitude_error_degC": np.nan,
                    "model_min_after_max": np.nan,
                    "signed_change_at_observed_extrema_ratio": np.nan,
                })
                rows.append(base)
                continue

            s = np.asarray(site_data.model_std[model][i:j+1], float)
            require(np.isfinite(s).all(),
                    f"{site_data.site}/{model}: finite common coverage but nonfinite STD.")

            p = int(np.argmax(s))
            q = int(np.argmin(s))
            amp = float(s[p] - s[q])
            ratio = float(amp / obs_amp)

            op = int(e.peak_index) - i
            oq = int(e.min_index) - i
            signed = float((s[op] - s[oq]) / obs_amp)

            base.update({
                "model_amplitude_degC": amp,
                "drop_ratio": ratio,
                "absolute_ratio_error": abs(ratio - 1.0),
                "absolute_amplitude_error_degC": abs(amp - obs_amp),
                "model_min_after_max": bool(q > p),
                "signed_change_at_observed_extrema_ratio": signed,
            })
            rows.append(base)
    return rows


def summarize_metrics(metrics):
    df = pd.DataFrame(metrics)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    valid = df[df["common_model_coverage"] & np.isfinite(df["drop_ratio"])].copy()
    out = []
    for (site, setting, model), g in valid.groupby(
        ["site","threshold_setting","model"], sort=False
    ):
        x = g["drop_ratio"].to_numpy(float)
        a = g["absolute_ratio_error"].to_numpy(float)
        ae = g["absolute_amplitude_error_degC"].to_numpy(float)
        out.append({
            "site": site,
            "threshold_setting": setting,
            "model": model,
            "N_events": len(g),
            "drop_ratio_mean": float(np.mean(x)),
            "drop_ratio_median": float(np.median(x)),
            "drop_ratio_q25": float(np.quantile(x, 0.25)),
            "drop_ratio_q75": float(np.quantile(x, 0.75)),
            "absolute_ratio_error_mean": float(np.mean(a)),
            "absolute_ratio_error_median": float(np.median(a)),
            "absolute_amplitude_error_degC_mean": float(np.mean(ae)),
            "absolute_amplitude_error_degC_median": float(np.median(ae)),
            "fraction_drop_ratio_below_1": float(np.mean(x < 1.0)),
            "fraction_model_min_after_max": float(np.mean(g["model_min_after_max"].astype(bool))),
            "signed_change_at_observed_extrema_ratio_mean":
                float(np.mean(g["signed_change_at_observed_extrema_ratio"].to_numpy(float))),
        })
    summary = pd.DataFrame(out)

    rank_rows = []
    setting_order = [x[0] for x in SETTINGS]
    for (site, setting), g in summary.groupby(["site","threshold_setting"], sort=False):
        gg = g.sort_values(
            ["absolute_ratio_error_median","absolute_amplitude_error_degC_median","model"]
        ).reset_index(drop=True)
        for rank, r in enumerate(gg.itertuples(index=False), 1):
            rank_rows.append({
                "site": site,
                "threshold_setting": setting,
                "rank_by_median_abs_ratio_error": rank,
                "model": r.model,
                "median_abs_ratio_error": r.absolute_ratio_error_median,
                "median_drop_ratio": r.drop_ratio_median,
                "N_events": r.N_events,
            })
    ranks = pd.DataFrame(rank_rows)

    # Compare each threshold's rank order against Original, without claiming significance.
    if not ranks.empty:
        compare = []
        for site in SITE_ORDER:
            rr = ranks[ranks.site == site]
            base = rr[rr.threshold_setting == "Original"].sort_values(
                "rank_by_median_abs_ratio_error"
            )["model"].tolist()
            for setting in setting_order:
                cur = rr[rr.threshold_setting == setting].sort_values(
                    "rank_by_median_abs_ratio_error"
                )["model"].tolist()
                compare.append({
                    "site": site,
                    "threshold_setting": setting,
                    "original_rank_order": " > ".join(base) if base else "",
                    "current_rank_order": " > ".join(cur) if cur else "",
                    "same_rank_order_as_original": bool(base and cur and base == cur),
                })
        rank_compare = pd.DataFrame(compare)
    else:
        rank_compare = pd.DataFrame()

    return summary, ranks, rank_compare


def plot_event_counts(counts, out):
    order = [x[0] for x in SETTINGS]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for site in SITE_ORDER:
        g = counts[counts.site == site].set_index("threshold_setting").reindex(order)
        ax.plot(order, g["detected_events"], marker="o", label=site)
    ax.set_xlabel("Threshold change")
    ax.set_ylabel("Number of detected events")
    ax.set_title("Fixed-depth event-count sensitivity")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "Figure_C1_fixed_depth_event_counts.png", dpi=300)
    plt.close(fig)


def plot_drop_ratio_site(metrics, site, out):
    order = [x[0] for x in SETTINGS]
    df = metrics[
        (metrics.site == site)
        & metrics.common_model_coverage
        & np.isfinite(metrics.drop_ratio)
    ].copy()

    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    positions = []
    data = []
    labels = []
    width = 0.16

    for si, setting in enumerate(order):
        center = si + 1
        for mi, model in enumerate(MODEL_ORDER):
            vals = df[(df.threshold_setting == setting) & (df.model == model)]["drop_ratio"].to_numpy(float)
            if len(vals) == 0:
                continue
            pos = center + (mi - 1.5) * width
            positions.append(pos)
            data.append(vals)
            labels.append((setting, model))

    if data:
        bp = ax.boxplot(
            data, positions=positions, widths=width*0.85,
            showfliers=False, patch_artist=False,
            medianprops={"linewidth": 1.5},
        )
        # Rebuild a simple legend using invisible line handles.
        for mi, model in enumerate(MODEL_ORDER):
            ax.plot([], [], label=model)
        ax.legend(ncol=4, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No evaluable events", transform=ax.transAxes,
                ha="center", va="center")

    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xticks(range(1, len(order)+1), order)
    ax.set_xlabel("Threshold change")
    ax.set_ylabel("Drop ratio")
    ax.set_title(f"{site}: drop-ratio sensitivity")
    ax.grid(axis="y", alpha=0.20)
    fig.tight_layout()
    fig.savefig(out / f"Figure_C2_drop_ratio_sensitivity_{site.lower()}.png", dpi=300)
    plt.close(fig)


def manual_report(counts, summary, ranks, rank_compare, out):
    lines = [
        "# R2-3 阈值敏感性：固定深度后的事件数量与四模型drop ratio",
        "",
        "本文件由用户本地脚本生成。没有重新训练、推理、运行GLM、重新选择模型成员或调阈值。",
        "",
        "## 阈值",
        "",
        "| 设置 | STD起点阈值 (°C) | 3h下降阈值 (°C) |",
        "|---|---:|---:|",
    ]
    for label, f in SETTINGS:
        lines.append(f"| {label} | {BASE_STD_THRESHOLD*f:.4f} | {BASE_DROP_THRESHOLD*f:.4f} |")

    lines += ["", "## 事件数量", "",
              "| Site | -50% | -25% | Original | +25% | +50% |",
              "|---|---:|---:|---:|---:|---:|"]
    order = [x[0] for x in SETTINGS]
    for site in SITE_ORDER:
        g = counts[counts.site == site].set_index("threshold_setting")
        vals = [str(int(g.loc[x, "detected_events"])) if x in g.index else "NA" for x in order]
        lines.append("| " + site + " | " + " | ".join(vals) + " |")

    lines += ["", "## 模型汇总（仅共同可评价事件）", ""]
    if summary.empty:
        lines.append("没有可评价事件。")
    else:
        for site in SITE_ORDER:
            lines += [f"### {site}", "",
                      "| Threshold | Model | N | mean ratio | median ratio | median |ratio-1| | mean abs amp error (°C) |",
                      "|---|---|---:|---:|---:|---:|---:|"]
            gsite = summary[summary.site == site]
            for setting in order:
                for model in MODEL_ORDER:
                    g = gsite[(gsite.threshold_setting == setting) & (gsite.model == model)]
                    if g.empty:
                        lines.append(f"| {setting} | {model} | 0 | NA | NA | NA | NA |")
                    else:
                        r = g.iloc[0]
                        lines.append(
                            f"| {setting} | {model} | {int(r.N_events)} | "
                            f"{r.drop_ratio_mean:.4f} | {r.drop_ratio_median:.4f} | "
                            f"{r.absolute_ratio_error_median:.4f} | "
                            f"{r.absolute_amplitude_error_degC_mean:.4f} |"
                        )
            lines.append("")

    lines += [
        "## 排名解释",
        "",
        "排名按每组事件的 median |drop ratio - 1| 从小到大生成；它只表示幅度比距离1的程度，",
        "不是显著性检验，也不是事件发生/时间预测能力。drop ratio本身越大并不一定越好。",
        "",
        "## 解释边界",
        "",
        "- 事件由观测独立识别，同一观测窗口应用于四个模型。",
        "- 这里检验的是阈值改变后事件幅度评价是否稳健，不评价模型独立预测事件发生或时间。",
        "- 固定深度和完整3小时窗口规则与R1-5修正保持一致。",
        "- Shiode使用0.25–2.25 m固定11层；Yamanokami使用0.40–2.20 m固定10层；Shin使用0.25–3.45 m固定17层。",
        "- 若某一阈值下事件太少，分布和排名只作描述，不应过度解释。",
        "- 若排名发生变化，应如实报告，而不是调整阈值或筛选事件以恢复原排序。",
        "",
    ]
    (out / "REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--out", type=Path, default=None)
    for key in EXPECTED_FILES:
        ap.add_argument("--" + key.replace("_","-"), dest=key, type=Path, default=None)
    args = ap.parse_args()

    root = args.root.resolve()
    require(root.is_dir(), f"Root does not exist: {root}")

    inputs = {}
    for key, fn in EXPECTED_FILES.items():
        inputs[key] = find_one(root, fn, getattr(args, key))
        print(f"{key:18s}: {inputs[key]}", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out.resolve() if args.out else root / "revision_R2_3" / f"threshold_sensitivity_{stamp}"
    require(not out.exists(), f"Output already exists: {out}")
    out.mkdir(parents=True)

    try:
        sites = {
            "Shiode": load_shiode(inputs["shiode_obs"], inputs["shiode_models"]),
            "Yamanokami": load_yamanokami(
                inputs["obs_yama_shin"], inputs["yama_ml"],
                inputs["yama_pgnn_added"], inputs["yama_glm"]
            ),
            "Shin": load_shin(
                inputs["obs_yama_shin"], inputs["shin_ml"], inputs["shin_glm"]
            ),
        }

        count_rows = []
        event_rows_all = []
        metric_rows = []

        for site_name in SITE_ORDER:
            s = sites[site_name]

            for label, factor in SETTINGS:
                std_thr = BASE_STD_THRESHOLD * factor
                drop_thr = BASE_DROP_THRESHOLD * factor

                events, cand = detect_events(
                    s.obs_std, s.complete_start, std_thr, drop_thr
                )

                # Baseline consistency assertion.
                if label == "Original":
                    require(
                        len(events) == EXPECTED_BASELINE_COUNTS[site_name],
                        f"{site_name}: Original threshold produced {len(events)} events; "
                        f"expected {EXPECTED_BASELINE_COUNTS[site_name]}. "
                        "Stop: input/version mismatch."
                    )

                # Common model coverage count.
                common_evaluable = 0
                for e in events.itertuples(index=False):
                    i, j = int(e.start_index), int(e.end_index)
                    if s.common_model_valid[i:j+1].all():
                        common_evaluable += 1

                count_rows.append({
                    "site": site_name,
                    "threshold_setting": label,
                    "factor": factor,
                    "STD_start_threshold_degC": std_thr,
                    "drop_threshold_degC": drop_thr,
                    "detected_events": len(events),
                    "events_with_common_four_model_coverage": common_evaluable,
                    "candidate_starts_before_merge": int(cand.sum()),
                })

                for e in events.itertuples(index=False):
                    event_rows_all.append({
                        "site": site_name,
                        "threshold_setting": label,
                        "event_id": int(e.event_id),
                        "start_time": str(pd.Timestamp(s.time[int(e.start_index)])),
                        "peak_time": str(pd.Timestamp(s.time[int(e.peak_index)])),
                        "min_time": str(pd.Timestamp(s.time[int(e.min_index)])),
                        "end_time": str(pd.Timestamp(s.time[int(e.end_index)])),
                        "obs_amplitude_degC": float(e.obs_amplitude_degC),
                    })

                metric_rows.extend(event_model_metrics(s, events, label))

        counts = pd.DataFrame(count_rows)
        event_catalog = pd.DataFrame(event_rows_all)
        metrics = pd.DataFrame(metric_rows)
        summary, ranks, rank_compare = summarize_metrics(metrics)

        counts.to_csv(out / "threshold_event_counts.csv", index=False, encoding="utf-8-sig")
        event_catalog.to_csv(out / "threshold_event_catalogs.csv", index=False, encoding="utf-8-sig")
        metrics.to_csv(out / "threshold_event_metrics_ensemble.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(out / "threshold_model_summary.csv", index=False, encoding="utf-8-sig")
        ranks.to_csv(out / "threshold_model_ranking.csv", index=False, encoding="utf-8-sig")
        rank_compare.to_csv(out / "threshold_rank_consistency.csv", index=False, encoding="utf-8-sig")

        plot_event_counts(counts, out)
        for site in SITE_ORDER:
            plot_drop_ratio_site(metrics, site, out)

        manual_report(counts, summary, ranks, rank_compare, out)

        settings = {
            "version": VERSION,
            "threshold_settings": [
                {
                    "label": label,
                    "factor": factor,
                    "STD_start_threshold_degC": BASE_STD_THRESHOLD*factor,
                    "drop_threshold_degC": BASE_DROP_THRESHOLD*factor,
                }
                for label, factor in SETTINGS
            ],
            "fixed_depths_m": {
                "Shiode": SHIODE_DEPTHS.tolist(),
                "Yamanokami": YAMA_DEPTHS.tolist(),
                "Shin": SHIN_DEPTHS.tolist(),
            },
            "window_steps": WINDOW_STEPS,
            "window_minutes": WINDOW_STEPS*STEP_MINUTES,
            "STD_ddof": 0,
            "primary_model_aggregation": "fixed-member temperature mean, then vertical STD",
            "drop_ratio_definition": "model max(STD)-min(STD) / observed max(STD)-min(STD) within observation-defined event window",
            "baseline_expected_event_counts": EXPECTED_BASELINE_COUNTS,
            "no_training": True,
            "no_inference": True,
            "no_GLM_run": True,
        }
        (out / "settings.json").write_text(
            json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())

        ret = out / "JoHRS_R2_3_threshold_sensitivity_return.zip"
        with zipfile.ZipFile(ret, "x", compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.iterdir()):
                if p.is_file() and p != ret:
                    z.write(p, p.name)

        print("\nCOMPLETE")
        print("READ:", out / "REPORT_zh.md")
        print("UPLOAD:", ret)

    except Exception:
        (out / "FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("\nFAILED. See:", out / "FAILED.txt")
        raise


if __name__ == "__main__":
    main()
