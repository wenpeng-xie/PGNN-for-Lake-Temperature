#!/usr/bin/env python3
"""
Revised JoHRS Figure 2 v2.

Updates from v1:
* outputs SVG in addition to PNG/PDF
* valid-depth summary text is moved OUTSIDE the plotting area
  and placed at the upper-right of each panel-(b) axes
* colorbar uses a manual axis so it is easier to control
* still keeps only panel (a) and panel (b)
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Keep SVG text as editable text rather than converting it to paths.
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

STEP = pd.Timedelta(minutes=10)
QUALITY_FLAGS = ("flag_nan", "flag_range", "flag_spike", "flag_stuck")


@dataclass(frozen=True)
class PeriodConfig:
    name: str
    site: str
    role: str
    filename: str
    start: str
    end: str
    target_depths: tuple[float, ...]
    alternate_offset_m: float = 0.0
    boundaries: tuple[tuple[str, str], ...] = ()


YAMA_CORE10 = tuple(np.round(np.arange(0.40, 2.20 + 0.001, 0.20), 2))
SHIODE_FIXED11 = tuple(np.round(np.arange(0.25, 2.25 + 0.001, 0.20), 2))
SHIN_FIXED17 = tuple(np.round(np.arange(0.25, 3.45 + 0.001, 0.20), 2))

PERIODS = (
    PeriodConfig(
        name="Yamanokami 2024",
        site="Yamanokami",
        role="Independent test",
        filename="yamanokami_2024_aligned_qc_long.csv",
        start="2024-02-20 10:00:00",
        end="2024-10-14 00:00:00",
        target_depths=YAMA_CORE10,
    ),
    PeriodConfig(
        name="Yamanokami 2025",
        site="Yamanokami",
        role="Development",
        filename="yamanokami_2025_aligned_qc_long.csv",
        start="2025-03-01 00:00:00",
        end="2025-11-23 00:00:00",
        target_depths=YAMA_CORE10,
        alternate_offset_m=+0.05,
    ),
    PeriodConfig(
        name="Shiode 2024",
        site="Shiode",
        role="Development",
        filename="shiode_2024_aligned_qc_long.csv",
        start="2024-02-21 13:00:00",
        end="2024-10-14 00:00:00",
        target_depths=SHIODE_FIXED11,
        alternate_offset_m=-0.05,
    ),
    PeriodConfig(
        name="Shiode 2025",
        site="Shiode",
        role="Independent test",
        filename="shiode_2025_aligned_qc_long.csv",
        start="2025-05-01 00:00:00",
        end="2025-11-23 00:00:00",
        target_depths=SHIODE_FIXED11,
    ),
    PeriodConfig(
        name="Shin 2025",
        site="Shin",
        role="Train / validation / test",
        filename="shin_2025_aligned_qc_long.csv",
        start="2025-03-01 00:00:00",
        end="2025-10-26 09:30:00",
        target_depths=SHIN_FIXED17,
        boundaries=(("2025-07-13 01:20:00", "Validation"), ("2025-08-15 13:50:00", "Test")),
    ),
)


@dataclass
class PeriodData:
    config: PeriodConfig
    source_path: Path
    time: pd.DatetimeIndex
    temperature: pd.DataFrame
    valid: pd.DataFrame
    n_valid: pd.Series
    complete: pd.Series
    std_complete: pd.Series
    mapping: dict[float, float]


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce').fillna(0).ne(0)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({'true', 't', '1', 'yes', 'y'})


def _require_columns(frame: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}. Available columns: {list(frame.columns)}")


def _match_source_depths(available: np.ndarray, target_depths: tuple[float, ...], alternate_offset_m: float, atol: float = 1e-6) -> dict[float, float]:
    available = np.asarray(sorted(set(float(x) for x in available)), dtype=float)
    mapping: dict[float, float] = {}
    for target in target_depths:
        exact = available[np.isclose(available, target, atol=atol, rtol=0)]
        if len(exact) == 1:
            mapping[float(target)] = float(exact[0])
            continue
        if len(exact) > 1:
            raise ValueError(f"Ambiguous exact source depths for target {target}: {exact}")
        alternate = target + alternate_offset_m
        hits = available[np.isclose(available, alternate, atol=atol, rtol=0)]
        if len(hits) == 1:
            mapping[float(target)] = float(hits[0])
            continue
        raise ValueError(
            f"Cannot map target depth {target:.2f} m. Tried exact {target:.2f} and historical label {alternate:.2f}. "
            f"Available depths: {available.tolist()}"
        )
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(f"One source depth was mapped more than once: {mapping}")
    return mapping


def load_period(root: Path, config: PeriodConfig) -> PeriodData:
    path = root / config.filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing required Figure 2 input:\n  {path}")
    usecols = ['time', 'depth_m', 'temp', *QUALITY_FLAGS]
    header = pd.read_csv(path, nrows=0)
    _require_columns(header, usecols, path)
    raw = pd.read_csv(path, usecols=usecols)
    raw['time'] = pd.to_datetime(raw['time'], errors='coerce')
    raw['depth_m'] = pd.to_numeric(raw['depth_m'], errors='coerce')
    raw['temp'] = pd.to_numeric(raw['temp'], errors='coerce')
    raw = raw.dropna(subset=['time', 'depth_m'])
    start = pd.Timestamp(config.start)
    end = pd.Timestamp(config.end)
    raw = raw[(raw['time'] >= start) & (raw['time'] <= end)].copy()
    if raw.empty:
        raise ValueError(f"{path.name}: no rows inside {start} to {end}")
    mapping = _match_source_depths(raw['depth_m'].dropna().unique(), config.target_depths, config.alternate_offset_m)
    raw = raw[raw['depth_m'].isin(set(mapping.values()))].copy()
    bad = pd.Series(False, index=raw.index)
    for flag in QUALITY_FLAGS:
        bad |= _as_bool(raw[flag])
    raw['valid_data_only'] = raw['temp'].notna() & ~bad
    duplicate_count = int(raw.duplicated(['time', 'depth_m'], keep=False).sum())
    if duplicate_count:
        warnings.warn(f"{config.name}: {duplicate_count} duplicate time-depth rows found; temperature is averaged and validity requires all duplicates valid.")
    temp_source = raw.pivot_table(index='time', columns='depth_m', values='temp', aggfunc='mean')
    valid_source = raw.pivot_table(index='time', columns='depth_m', values='valid_data_only', aggfunc='all')
    full_time = pd.date_range(start, end, freq=STEP)
    temp_source = temp_source.reindex(full_time)
    valid_source = valid_source.reindex(full_time, fill_value=False).astype(bool)
    temp = pd.DataFrame(index=full_time)
    valid = pd.DataFrame(index=full_time)
    for target in config.target_depths:
        source = mapping[float(target)]
        if source not in temp_source.columns:
            temp[target] = np.nan
            valid[target] = False
        else:
            temp[target] = pd.to_numeric(temp_source[source], errors='coerce')
            valid[target] = valid_source[source].astype(bool) & temp[target].notna()
    temp.columns = [float(c) for c in temp.columns]
    valid.columns = [float(c) for c in valid.columns]
    temp = temp.where(valid)
    n_valid = valid.sum(axis=1).astype(int)
    complete = valid.all(axis=1)
    std_values = np.std(temp.to_numpy(float), axis=1, ddof=0)
    std_values[~complete.to_numpy()] = np.nan
    std_complete = pd.Series(std_values, index=full_time, name='STD_complete')
    return PeriodData(config, path, full_time, temp, valid, n_valid, complete, std_complete, mapping)


def set_time_axis(ax, start: pd.Timestamp, end: pd.Timestamp, interval_months: int) -> None:
    ax.set_xlim(start, end)
    locator = mdates.MonthLocator(interval=interval_months)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=35, labelsize=8)


def draw_boundaries(ax, boundaries: tuple[tuple[str, str], ...], add_labels: bool) -> None:
    for when, label in boundaries:
        t = pd.Timestamp(when)
        ax.axvline(t, linestyle='--', linewidth=1.0, color='black', zorder=5)
        if add_labels:
            ax.text(t, 0.98, label, transform=ax.get_xaxis_transform(), rotation=90, va='top', ha='right', fontsize=7)


def plot_heatmap(ax, data: PeriodData, vmin: float, vmax: float, cmap):
    temp = np.ma.masked_invalid(data.temperature.to_numpy(float).T)
    mesh = ax.pcolormesh(data.time, np.asarray(data.temperature.columns, dtype=float), temp, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    ax.invert_yaxis()
    ax.set_ylabel('Depth (m)')
    ax.set_yticks(data.temperature.columns)
    ax.set_yticklabels([f'{d:.2f}' for d in data.temperature.columns], fontsize=7)
    draw_boundaries(ax, data.config.boundaries, add_labels=False)
    return mesh


def add_summary_text_outside(ax, data: PeriodData) -> None:
    n_fixed = data.temperature.shape[1]
    complete_pct = 100.0 * float(data.complete.mean())
    text = f'n={n_fixed}; complete={complete_pct:.1f}%'
    ax.text(1.0, 1.10, text, transform=ax.transAxes, va='bottom', ha='right', fontsize=8)


def plot_std_and_count(ax, data: PeriodData, show_valid_count: bool):
    n_fixed = data.temperature.shape[1]
    ax.plot(data.time, data.std_complete, linewidth=0.85, label='STD(t), complete profiles', zorder=4)
    ax.set_ylabel('STD(t) (°C)')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', linewidth=0.35, alpha=0.35)
    draw_boundaries(ax, data.config.boundaries, add_labels=True)
    add_summary_text_outside(ax, data)
    count_ax = None
    if show_valid_count:
        count_ax = ax.twinx()
        count_ax.step(data.time, data.n_valid, where='mid', linewidth=0.55, alpha=0.55, label='Valid fixed depths', zorder=2)
        count_ax.set_ylim(-0.25, n_fixed + 0.25)
        middle = int(round(n_fixed / 2))
        ticks = sorted(set([0, middle, n_fixed]))
        count_ax.set_yticks(ticks)
        count_ax.set_ylabel('Valid depths (n)', fontsize=8)
        count_ax.tick_params(axis='y', labelsize=7)
    return count_ax


def write_audit(outdir: Path, periods: list[PeriodData]) -> None:
    rows = []
    mapping_rows = []
    for d in periods:
        n_fixed = d.temperature.shape[1]
        n_time = len(d.time)
        n_complete = int(d.complete.sum())
        rows.append(dict(period=d.config.name, role=d.config.role, source_file=str(d.source_path), start=str(d.time[0]), end=str(d.time[-1]), n_time=n_time, n_fixed_depths=n_fixed, complete_profiles=n_complete, complete_fraction=n_complete / n_time, valid_depth_count_min=int(d.n_valid.min()), valid_depth_count_max=int(d.n_valid.max()), valid_temperature_cells=int(d.valid.to_numpy().sum()), expected_temperature_cells=int(d.valid.size)))
        for target, source in d.mapping.items():
            mapping_rows.append(dict(period=d.config.name, nominal_depth_m=target, source_depth_label_m=source, label_offset_m=source - target))
    pd.DataFrame(rows).to_csv(outdir / 'Figure2_period_completeness_audit.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(mapping_rows).to_csv(outdir / 'Figure2_depth_mapping_audit.csv', index=False, encoding='utf-8-sig')
    metadata = {
        'figure_layout': '5 rows x 2 columns; no third panel',
        'panel_a': 'fixed-depth observed temperature heatmap; missing/QC-excluded cells gray',
        'panel_b': 'population STD(t), complete profiles only; valid-depth count optional on secondary axis',
        'summary_text_location': 'outside panel b, upper-right of each subplot',
        'quality_flags_excluded': list(QUALITY_FLAGS),
        'ignored_for_independent_exclusion': ['flag_vert', 'flag_unstable', 'qc_bad'],
        'temperature_interpolation': 'none',
        'time_interpolation': 'none',
        'time_grid': 'continuous 10-minute index',
        'STD_ddof': 0,
    }
    (outdir / 'Figure2_method_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def add_manual_colorbar(fig, axes_leftcol, heatmap_mesh):
    tops = [ax.get_position().y1 for ax in axes_leftcol]
    bottoms = [ax.get_position().y0 for ax in axes_leftcol]
    rights = [ax.get_position().x1 for ax in axes_leftcol]
    cax_x0 = max(rights) + 0.010
    cax = fig.add_axes([cax_x0, min(bottoms), 0.012, max(tops) - min(bottoms)])
    cbar = fig.colorbar(heatmap_mesh, cax=cax)
    cbar.set_label('Water temperature (°C)')
    return cbar


def build_figure(root: Path, outdir: Path, dpi: int, tick_month_interval: int, show_valid_count: bool) -> None:
    periods = [load_period(root, config) for config in PERIODS]
    all_values = np.concatenate([d.temperature.to_numpy(float).ravel() for d in periods])
    all_values = all_values[np.isfinite(all_values)]
    if len(all_values) == 0:
        raise ValueError('No finite fixed-depth temperature observations were loaded.')
    vmin = float(np.nanpercentile(all_values, 2))
    vmax = float(np.nanpercentile(all_values, 98))
    cmap = plt.get_cmap('turbo').copy()
    cmap.set_bad('#d9d9d9')
    fig, axes = plt.subplots(nrows=len(periods), ncols=2, figsize=(15.6, 16.4), gridspec_kw={'width_ratios': [3.5, 2.25], 'hspace': 0.56, 'wspace': 0.40})
    fig.subplots_adjust(left=0.07, right=0.90, top=0.95, bottom=0.06)
    heatmap_mesh = None
    for row, d in enumerate(periods):
        ax_a, ax_b = axes[row]
        heatmap_mesh = plot_heatmap(ax_a, d, vmin, vmax, cmap)
        plot_std_and_count(ax_b, d, show_valid_count=show_valid_count)
        for ax in (ax_a, ax_b):
            set_time_axis(ax, d.time[0], d.time[-1], interval_months=tick_month_interval)
        ax_a.set_title(f'{d.config.name} ({d.config.role})', fontsize=10, loc='left', pad=8)
        ax_b.set_title(f'{d.config.name}', fontsize=10, loc='left', pad=8)
    axes[0, 0].text(0.0, 1.20, '(a) Observed temperature profiles at fixed depths', transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold', ha='left')
    axes[0, 1].text(0.0, 1.20, '(b) Complete-profile STD(t)', transform=axes[0, 1].transAxes, fontsize=12, fontweight='bold', ha='left')
    if heatmap_mesh is None:
        raise RuntimeError('No heatmap was generated')
    add_manual_colorbar(fig, [axes[i, 0] for i in range(len(periods))], heatmap_mesh)
    legend_handles = [Patch(facecolor='#d9d9d9', edgecolor='none', label='Missing or QC-excluded'), Line2D([0], [0], linewidth=1.2, label='STD(t), complete profiles only')]
    if show_valid_count:
        legend_handles.append(Line2D([0], [0], linewidth=0.8, alpha=0.6, label='Number of valid fixed depths (right axis)'))
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles), frameon=False, bbox_to_anchor=(0.5, 0.008), fontsize=8.5)
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / 'Figure2_fixed_depth_complete_profile_ab.png'
    pdf = outdir / 'Figure2_fixed_depth_complete_profile_ab.pdf'
    svg = outdir / 'Figure2_fixed_depth_complete_profile_ab.svg'
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    fig.savefig(svg)
    plt.close(fig)
    write_audit(outdir, periods)
    print('\nCOMPLETE')
    print(f'PNG: {png}')
    print(f'PDF: {pdf}')
    print(f'SVG: {svg}')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path.cwd(), help='Project root containing the five *_aligned_qc_long.csv files.')
    p.add_argument('--outdir', type=Path, default=None, help='Output directory. Default: <root>/revision_figures/Figure2_fixed_depth_ab_v2_1')
    p.add_argument('--dpi', type=int, default=500)
    p.add_argument('--tick-month-interval', type=int, default=2)
    p.add_argument('--hide-valid-depth-count', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Root directory does not exist: {root}')
    outdir = args.outdir.expanduser().resolve() if args.outdir is not None else root / 'revision_figures' / 'Figure2_fixed_depth_ab_v2_1'
    build_figure(root=root, outdir=outdir, dpi=args.dpi, tick_month_interval=args.tick_month_interval, show_valid_count=not args.hide_valid_depth_count)


if __name__ == '__main__':
    main()
