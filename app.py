"""
Bender Element Signal Interpretation App

Author:
    Mohammad Jawed Roshan
    Researcher in Geotechnical Engineering
    University of Minho, Portugal

Description:
    This application processes bender element test data to estimate shear wave
    velocity (Vs) and small-strain shear modulus (Gmax). Two interpretation
    approaches are implemented:

    1. Peak-to-peak method
    2. Normalized cross-correlation method

    The app supports both single-file and batch processing. Results and plots
    can be exported for reporting and further analysis.

Usage notes:
    - Upload CSV, Excel, or text files containing time, input, and output signals.
    - Select the appropriate columns and units.
    - Review the plots together with the reported values before final interpretation.
"""


import io
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import correlate, find_peaks


st.set_page_config(page_title="Bender Element Signal Interpretation App", layout="wide")


LOGO_FILENAME_CANDIDATES = [
    "header_logo.png",
    "logo.png",
    "logos.png",
    "header.png",
    "image.png",
]



@dataclass
class AnalysisResult:
    method: str
    travel_time_s: Optional[float]
    shear_wave_velocity_m_s: Optional[float]
    gmax_pa: Optional[float]
    gmax_mpa: Optional[float]
    arrival_time_s: Optional[float]
    notes: str
    reference_or_source_time_s: Optional[float] = None


@dataclass
class PeakToPeakDetails:
    input_peak_time_s: Optional[float]
    output_peak_time_s: Optional[float]
    input_peak_value: Optional[float]
    output_peak_value: Optional[float]


@dataclass
class FileAnalysisOutput:
    file_name: str
    results_df: pd.DataFrame
    peak_details: PeakToPeakDetails
    lags_s: Optional[np.ndarray]
    corr: Optional[np.ndarray]
    time_s: np.ndarray
    input_processed: np.ndarray
    output_processed: np.ndarray
    cleaned_df: pd.DataFrame



def get_script_dir() -> str:
    if "__file__" in globals():
        return os.path.dirname(os.path.abspath(__file__))
    return os.getcwd()



def find_first_existing_file(candidates: List[str]) -> Optional[str]:
    script_dir = Path(get_script_dir())

    for name in candidates:
        candidate = script_dir / name
        if candidate.exists():
            return str(candidate)

    app_files = {item.name.lower(): item for item in script_dir.iterdir() if item.is_file()}
    for name in candidates:
        matched = app_files.get(name.lower())
        if matched is not None:
            return str(matched)

    for folder_name in ["assets", "images", "img"]:
        folder = script_dir / folder_name
        if folder.exists() and folder.is_dir():
            folder_files = {item.name.lower(): item for item in folder.iterdir() if item.is_file()}
            for name in candidates:
                matched = folder_files.get(name.lower())
                if matched is not None:
                    return str(matched)

    return None



def find_logo_file() -> Optional[str]:
    return find_first_existing_file(LOGO_FILENAME_CANDIDATES)




def render_header() -> None:
    logo_path = find_logo_file()
    if logo_path is not None:
        st.image(logo_path, use_container_width=True)

    st.markdown(
        """
        <div style='text-align: center; margin-top: 0.25rem; margin-bottom: 0.75rem;'>
            <h1 style='margin-bottom: 0.2rem;'>Bender Element Signal Interpretation App</h1>
            <div style='font-size: 1.05rem;'>
                Developed by: Mohammad Jawed Roshan, António Gomes Correia, Ionut Dragos Moldovan, Miguel Azenha
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border:2px solid black;'>", unsafe_allow_html=True)




def safe_read_table(file_obj) -> pd.DataFrame:
    filename = getattr(file_obj, "name", "uploaded_file").lower()
    file_obj.seek(0)

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        try:
            return pd.read_excel(file_obj)
        except Exception as exc:
            raise ValueError(f"Could not read Excel file: {exc}")

    if filename.endswith(".txt"):
        separators = [None, "\t", ",", ";", "\\s+"]
        for sep in separators:
            file_obj.seek(0)
            try:
                if sep is None:
                    df = pd.read_csv(file_obj, sep=None, engine="python")
                else:
                    df = pd.read_csv(file_obj, sep=sep, engine="python")
                if df.shape[1] >= 2:
                    return df
            except Exception:
                continue
        raise ValueError(
            "Could not parse the text file automatically. Use a delimited text file with tab, comma, semicolon, or whitespace separators."
        )

    try:
        return pd.read_csv(file_obj)
    except Exception:
        file_obj.seek(0)
        return pd.read_csv(file_obj, engine="python")



def load_be_file(file_obj) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = safe_read_table(file_obj)
    df.columns = [str(c).strip() for c in df.columns]
    units_info: Dict[str, str] = {}

    if len(df) > 0:
        first_row = df.iloc[0].astype(str).tolist()
        non_numeric_count = 0
        for value in first_row:
            try:
                float(str(value).strip())
            except Exception:
                non_numeric_count += 1
        if non_numeric_count >= max(1, len(first_row) // 2):
            for col, unit in zip(df.columns, first_row):
                units_info[col] = str(unit)
            df = df.iloc[1:].reset_index(drop=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(how="all").reset_index(drop=True)
    return df, units_info



def get_sampling_info(time_values: np.ndarray, declared_unit: str) -> Tuple[np.ndarray, float]:
    time_values = np.asarray(time_values, dtype=float)

    if declared_unit == "ms":
        time_seconds = time_values * 1e-3
    elif declared_unit == "us":
        time_seconds = time_values * 1e-6
    elif declared_unit == "ns":
        time_seconds = time_values * 1e-9
    else:
        time_seconds = time_values.copy()

    diffs = np.diff(time_seconds)
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise ValueError("The time column does not contain a valid positive time step.")

    dt = float(np.median(diffs))
    return time_seconds, dt



def normalize_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    peak = np.nanmax(np.abs(signal))
    if not np.isfinite(peak) or peak == 0:
        return signal.copy()
    return signal / peak



def estimate_reference_time(time_s: np.ndarray, input_signal: np.ndarray, threshold_ratio: float = 0.1) -> float:
    abs_sig = np.abs(np.asarray(input_signal, dtype=float))
    peak = np.nanmax(abs_sig)
    if not np.isfinite(peak) or peak == 0:
        return float(time_s[0])

    threshold = threshold_ratio * peak
    idx = np.where(abs_sig >= threshold)[0]
    if len(idx) == 0:
        return float(time_s[0])
    return float(time_s[idx[0]])



def crop_window(time_s: np.ndarray, signal: np.ndarray, start_time_s: float, end_time_s: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    if end_time_s is None:
        mask = time_s >= start_time_s
    else:
        mask = (time_s >= start_time_s) & (time_s <= end_time_s)
    return time_s[mask], signal[mask]



def peak_to_peak_method(
    time_s: np.ndarray,
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    reference_time_s: float,
    search_delay_s: float,
    search_end_s: Optional[float],
    prominence_ratio: float,
    selected_output_peak_time_s: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float], str, PeakToPeakDetails]:
    input_signal = np.asarray(input_signal, dtype=float)
    output_signal = np.asarray(output_signal, dtype=float)

    input_peak = float(np.nanmax(input_signal))
    if not np.isfinite(input_peak) or input_peak <= 0:
        return None, None, "A positive input peak could not be identified.", PeakToPeakDetails(None, None, None, None)

    input_peaks, _ = find_peaks(input_signal, prominence=prominence_ratio * input_peak)
    if len(input_peaks) == 0:
        tx_idx = int(np.argmax(input_signal))
    else:
        valid_input_peaks = input_peaks[time_s[input_peaks] >= reference_time_s]
        if len(valid_input_peaks) > 0:
            tx_idx = int(valid_input_peaks[0])
        else:
            tx_idx = int(input_peaks[np.argmax(input_signal[input_peaks])])

    tx_peak_time_s = float(time_s[tx_idx])
    tx_peak_value = float(input_signal[tx_idx])

    start_time_s = tx_peak_time_s + search_delay_s
    t_out, s_out = crop_window(time_s, output_signal, start_time_s, search_end_s)
    if len(t_out) < 5:
        return tx_peak_time_s, None, "Insufficient data in the peak-to-peak search window.", PeakToPeakDetails(tx_peak_time_s, None, tx_peak_value, None)

    output_peak = float(np.nanmax(s_out))
    if not np.isfinite(output_peak) or output_peak <= 0:
        return tx_peak_time_s, None, "A positive output peak could not be identified.", PeakToPeakDetails(tx_peak_time_s, None, tx_peak_value, None)

    if selected_output_peak_time_s is not None:
        # Use the user-selected received/output peak. The nearest available sample
        # is used so the selected value remains valid even if the sampling interval
        # prevents an exact time match.
        if selected_output_peak_time_s < t_out[0] or selected_output_peak_time_s > t_out[-1]:
            return tx_peak_time_s, None, "The selected output peak is outside the peak-to-peak search window.", PeakToPeakDetails(tx_peak_time_s, None, tx_peak_value, None)
        rx_local_idx = int(np.argmin(np.abs(t_out - selected_output_peak_time_s)))
        note = "Travel time is measured from the transmitted positive peak time to the user-selected received/output peak."
    else:
        rx_local_idx = int(np.argmax(s_out))
        note = "Travel time is measured from the transmitted positive peak time to the highest positive point of the received signal within the search window."

    rx_peak_time_s = float(t_out[rx_local_idx])
    rx_peak_value = float(s_out[rx_local_idx])

    if rx_peak_time_s <= tx_peak_time_s:
        return tx_peak_time_s, None, "Peak-to-peak travel time is not positive.", PeakToPeakDetails(tx_peak_time_s, None, tx_peak_value, None)

    details = PeakToPeakDetails(tx_peak_time_s, rx_peak_time_s, tx_peak_value, rx_peak_value)
    return tx_peak_time_s, rx_peak_time_s, note, details


def get_peak_to_peak_candidates(
    file_obj,
    time_col: str,
    input_col: str,
    output_col: str,
    time_unit: str,
    prominence_ratio: float = 0.15,
    search_delay_s: float = 0.05e-3,
    search_end_s: Optional[float] = 3.0e-3,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, PeakToPeakDetails]:
    """Return candidate received/output peaks for manual peak-to-peak selection.

    The transmitted/input peak is detected first. Then all positive received/output
    peaks inside the same interpretation window are listed so the user can select
    the physically correct first arrival peak when it is not the maximum peak.
    """
    df, _ = load_be_file(file_obj)
    time_raw = df[time_col].to_numpy(dtype=float)
    input_raw = df[input_col].to_numpy(dtype=float)
    output_raw = df[output_col].to_numpy(dtype=float)

    valid_mask = np.isfinite(time_raw) & np.isfinite(input_raw) & np.isfinite(output_raw)
    time_raw = time_raw[valid_mask]
    input_raw = input_raw[valid_mask]
    output_raw = output_raw[valid_mask]

    time_s, _ = get_sampling_info(time_raw, time_unit)
    input_processed = input_raw - np.nanmean(input_raw)
    output_processed = output_raw - np.nanmean(output_raw)

    reference_time_s = estimate_reference_time(time_s, input_processed, threshold_ratio=0.10)

    input_peak = float(np.nanmax(input_processed))
    input_peaks, _ = find_peaks(input_processed, prominence=prominence_ratio * input_peak)
    if len(input_peaks) == 0:
        tx_idx = int(np.argmax(input_processed))
    else:
        valid_input_peaks = input_peaks[time_s[input_peaks] >= reference_time_s]
        tx_idx = int(valid_input_peaks[0]) if len(valid_input_peaks) > 0 else int(input_peaks[np.argmax(input_processed[input_peaks])])

    tx_peak_time_s = float(time_s[tx_idx])
    tx_peak_value = float(input_processed[tx_idx])

    start_time_s = tx_peak_time_s + search_delay_s
    mask = (time_s >= start_time_s) if search_end_s is None else ((time_s >= start_time_s) & (time_s <= search_end_s))
    t_out = time_s[mask]
    s_out = output_processed[mask]

    if len(t_out) < 5:
        empty = pd.DataFrame(columns=["Peak number", "Time (ms)", "Amplitude", "Candidate label"])
        return empty, time_s, input_processed, output_processed, PeakToPeakDetails(tx_peak_time_s, None, tx_peak_value, None)

    positive_peak = float(np.nanmax(s_out))
    if not np.isfinite(positive_peak) or positive_peak <= 0:
        empty = pd.DataFrame(columns=["Peak number", "Time (ms)", "Amplitude", "Candidate label"])
        return empty, time_s, input_processed, output_processed, PeakToPeakDetails(tx_peak_time_s, None, tx_peak_value, None)

    output_peaks_local, _ = find_peaks(s_out, prominence=prominence_ratio * positive_peak)
    output_peaks_local = [idx for idx in output_peaks_local if s_out[idx] > 0]

    if len(output_peaks_local) == 0:
        output_peaks_local = [int(np.argmax(s_out))]

    rows = []
    max_local_idx = int(np.argmax(s_out))
    first_peak_time_s = float(t_out[output_peaks_local[0]])
    max_peak_time_s = float(t_out[max_local_idx])

    for number, local_idx in enumerate(output_peaks_local, start=1):
        peak_time_s = float(t_out[local_idx])
        labels = []
        if np.isclose(peak_time_s, first_peak_time_s):
            labels.append("first detected positive peak")
        if np.isclose(peak_time_s, max_peak_time_s):
            labels.append("maximum positive peak")
        rows.append({
            "Peak number": number,
            "Time (ms)": peak_time_s * 1e3,
            "Amplitude": float(s_out[local_idx]),
            "Candidate label": "; ".join(labels) if labels else "candidate positive peak",
        })

    candidates_df = pd.DataFrame(rows)
    details = PeakToPeakDetails(tx_peak_time_s, None, tx_peak_value, None)
    return candidates_df, time_s, input_processed, output_processed, details


def make_peak_candidate_plot(
    time_s: np.ndarray,
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    input_details: PeakToPeakDetails,
    candidates_df: pd.DataFrame,
    zoom_end_ms: float = 3.0,
):
    fig = make_peak_to_peak_plot(time_s, input_signal, output_signal, input_details, zoom_end_ms)
    ax1 = fig.axes[0]
    ax2 = fig.axes[1]

    for _, row in candidates_df.iterrows():
        t_ms = float(row["Time (ms)"])
        amp = float(row["Amplitude"])
        number = int(row["Peak number"])
        ax2.plot(t_ms, amp, marker="s", linestyle="None", markersize=6, color="black", zorder=6)
        ax2.annotate(str(number), xy=(t_ms, amp), xytext=(4, 4), textcoords="offset points", fontsize=9, color="black")

    ax1.set_title("Candidate received/output peaks for manual peak-to-peak selection")
    fig.tight_layout()
    return fig



def cross_correlation_method(
    time_s: np.ndarray,
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    reference_time_s: float,
    search_delay_s: float,
    search_end_s: Optional[float],
) -> Tuple[Optional[float], str, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute normalized cross-correlation between the transmitted and received signals.

    The recorded bender element signals are discrete, so the delay is estimated using
    the discrete normalized cross-correlation. Mean removal is applied first, and each
    signal segment is scaled by its Euclidean norm so the correlation is not biased by
    amplitude differences between the two signals.
    """
    dt = float(np.median(np.diff(time_s)))


    input_start_s = reference_time_s
    input_end_s = search_end_s
    if input_end_s is not None and input_end_s <= input_start_s:
        input_end_s = None

    _, s_in = crop_window(time_s, input_signal, input_start_s, input_end_s)
    _, s_out = crop_window(time_s, output_signal, reference_time_s + search_delay_s, search_end_s)

    if len(s_in) < 5 or len(s_out) < 5:
        return None, "Insufficient data for cross-correlation.", None, None


    # Remove the DC component before correlation
    s_in = s_in - np.nanmean(s_in)
    s_out = s_out - np.nanmean(s_out)


    # Scale each signal window to unit energy
    norm_in = np.linalg.norm(s_in)
    norm_out = np.linalg.norm(s_out)
    if norm_in == 0 or norm_out == 0:
        return None, "Normalization failed due to zero signal energy.", None, None

    s_in = s_in / norm_in
    s_out = s_out / norm_out

    corr = correlate(s_out, s_in, mode="full")
    lags = np.arange(-len(s_in) + 1, len(s_out)) * dt


    # Only keep physically meaningful positive lags after the imposed delay
    positive_mask = lags >= search_delay_s
    if not np.any(positive_mask):
        return None, "No valid positive lag range for cross-correlation.", lags, corr

    valid_lags = lags[positive_mask]
    valid_corr = corr[positive_mask]
    best_idx = int(np.argmax(np.abs(valid_corr)))
    lag_s = float(valid_lags[best_idx])
    arrival_time_s = reference_time_s + lag_s

    if arrival_time_s <= reference_time_s:
        return None, "Cross-correlation returned a non-positive travel time.", lags, corr

    return arrival_time_s, "Arrival estimated using normalized cross-correlation.", lags, corr



def calculate_vs_and_gmax(
    arrival_time_s: Optional[float],
    reference_time_s: float,
    travel_length_m: float,
    density_kg_m3: float,
    method_name: str,
    notes: str,
    source_time_s: Optional[float] = None,
) -> AnalysisResult:
    if arrival_time_s is None:
        return AnalysisResult(method_name, None, None, None, None, None, notes, source_time_s)

    origin_time_s = reference_time_s if source_time_s is None else source_time_s
    travel_time_s = arrival_time_s - origin_time_s
    if travel_time_s <= 0:
        return AnalysisResult(method_name, travel_time_s, None, None, None, arrival_time_s, "Computed travel time is not positive.", origin_time_s)

    vs = travel_length_m / travel_time_s
    gmax_pa = density_kg_m3 * vs ** 2
    gmax_mpa = gmax_pa / 1e6
    return AnalysisResult(method_name, travel_time_s, vs, gmax_pa, gmax_mpa, arrival_time_s, notes, origin_time_s)



def build_results_table(results: Dict[str, AnalysisResult]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        rows.append(
            {
                "Method": name,
                "Arrival time (ms)": None if result.arrival_time_s is None else result.arrival_time_s * 1e3,
                "Travel time (ms)": None if result.travel_time_s is None else result.travel_time_s * 1e3,
                "Shear wave velocity Vs (m/s)": result.shear_wave_velocity_m_s,
                "Small-strain shear modulus Gmax (MPa)": result.gmax_mpa,
                "Notes": result.notes,
            }
        )
    return pd.DataFrame(rows)


def add_average_gmax_row(results_df: pd.DataFrame) -> pd.DataFrame:
    gmax_values = pd.to_numeric(results_df["Small-strain shear modulus Gmax (MPa)"], errors="coerce").dropna()
    vs_values = pd.to_numeric(results_df["Shear wave velocity Vs (m/s)"], errors="coerce").dropna()

    average_row = {
        "Method": "Average of methods",
        "Arrival time (ms)": np.nan,
        "Travel time (ms)": np.nan,
        "Shear wave velocity Vs (m/s)": vs_values.mean() if not vs_values.empty else np.nan,
        "Small-strain shear modulus Gmax (MPa)": gmax_values.mean() if not gmax_values.empty else np.nan,
        "Notes": "Arithmetic average based on the available interpretation methods.",
    }

    return pd.concat([results_df, pd.DataFrame([average_row])], ignore_index=True)



def make_peak_to_peak_plot(
    time_s: np.ndarray,
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    details: PeakToPeakDetails,
    zoom_end_ms: float,
):
    """
    Plot the peak-to-peak interpretation using dual y-axes.

    The transmitted/input signal is shown on the left axis in volts (V),
    while the received/output signal is shown on the right axis in millivolts (mV).
    Different colors are used for the two signals and their detected peaks to
    improve readability.
    """
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    time_ms = time_s * 1e3

    
    line_in, = ax1.plot(
        time_ms,
        input_signal,
        linewidth=1.2,
        color="blue",
        label="Input signal (V)",
    )
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Input signal (V)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(alpha=0.3)

    
    ax2 = ax1.twinx()
    line_out, = ax2.plot(
        time_ms,
        output_signal,
        linewidth=1.2,
        linestyle="--",
        color="red",
        label="Output signal (mV)",
    )
    ax2.set_ylabel("Output signal (mV)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

   
    peak_handles = []
    peak_labels = []

    if details.input_peak_time_s is not None and details.input_peak_value is not None:
        input_peak_handle, = ax1.plot(
            details.input_peak_time_s * 1e3,
            details.input_peak_value,
            marker="o",
            linestyle="None",
            markersize=8,
            color="blue",
            markeredgecolor="black",
            label="Input peak",
            zorder=5,
        )
        peak_handles.append(input_peak_handle)
        peak_labels.append("Input peak")

    if details.output_peak_time_s is not None and details.output_peak_value is not None:
        output_peak_handle, = ax2.plot(
            details.output_peak_time_s * 1e3,
            details.output_peak_value,
            marker="o",
            linestyle="None",
            markersize=8,
            color="red",
            markeredgecolor="black",
            label="Output peak",
            zorder=5,
        )
        peak_handles.append(output_peak_handle)
        peak_labels.append("Output peak")

    start_ms = 0.0
    if details.input_peak_time_s is not None:
        start_ms = max(0.0, details.input_peak_time_s * 1e3 - 0.2)

    ax1.set_xlim(start_ms, zoom_end_ms)
    ax1.set_title("Peak-to-peak interpretation")

    handles = [line_in, line_out] + peak_handles
    labels = ["Input signal (V)", "Output signal (mV)"] + peak_labels
    ax1.legend(handles, labels)

    fig.tight_layout()
    return fig


def make_cross_correlation_plot(lags_s: np.ndarray, corr: np.ndarray):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    positive_mask = lags_s >= 0
    ax.plot(lags_s[positive_mask] * 1e3, corr[positive_mask], linewidth=1.4, color="red")
    ax.set_xlim(left=0)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Cross-correlation")
    ax.set_title("Cross-correlation between input and output signals")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig



def fig_to_png_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()



def to_excel_bytes(results_df: pd.DataFrame, cleaned_data_map: Dict[str, pd.DataFrame]) -> Optional[bytes]:
    try:
        output_buffer = io.BytesIO()
        with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
            results_df.to_excel(writer, sheet_name="method_results", index=False)
            for file_name, cleaned_df in cleaned_data_map.items():
                safe_sheet = sanitize_sheet_name(f"data_{os.path.splitext(file_name)[0]}")
                cleaned_df.to_excel(writer, sheet_name=safe_sheet, index=False)
        return output_buffer.getvalue()
    except Exception:
        return None



def sanitize_sheet_name(name: str) -> str:
    cleaned = re.sub(r"[:\\/?*\[\]]", "_", name)
    return cleaned[:31] if cleaned else "Sheet1"



def sanitize_file_stem(file_name: str) -> str:
    stem = os.path.splitext(file_name)[0]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "uploaded_file"



def create_plot_zip(plot_files: Dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for file_name, data in plot_files.items():
            zip_file.writestr(file_name, data)
    buffer.seek(0)
    return buffer.getvalue()



def analyze_file(
    file_obj,
    time_col: str,
    input_col: str,
    output_col: str,
    time_unit: str,
    travel_length_mm: float,
    density_kg_m3: float,
    selected_output_peak_time_s: Optional[float] = None,
) -> FileAnalysisOutput:
    df, _ = load_be_file(file_obj)
    if df.empty:
        raise ValueError("No usable numeric data were found after cleaning the file.")

    time_raw = df[time_col].to_numpy(dtype=float)
    input_raw = df[input_col].to_numpy(dtype=float)
    output_raw = df[output_col].to_numpy(dtype=float)

    valid_mask = np.isfinite(time_raw) & np.isfinite(input_raw) & np.isfinite(output_raw)
    time_raw = time_raw[valid_mask]
    input_raw = input_raw[valid_mask]
    output_raw = output_raw[valid_mask]

    if len(time_raw) < 10:
        raise ValueError("The cleaned data contain too few valid rows for analysis.")

    time_s, _ = get_sampling_info(time_raw, time_unit)
    input_processed = input_raw - np.nanmean(input_raw)
    output_processed = output_raw - np.nanmean(output_raw)

    reference_time_s = estimate_reference_time(time_s, input_processed, threshold_ratio=0.10)
    search_delay_s = 0.05e-3
    search_end_s = 3.0e-3
    peak_prominence_ratio = 0.15
    # Convert specimen length from mm to m
    travel_length_m = travel_length_mm * 1e-3

    results: Dict[str, AnalysisResult] = {}

    tx_peak_time_s, rx_peak_time_s, note_p2p, peak_details = peak_to_peak_method(
        time_s,
        input_processed,
        output_processed,
        reference_time_s,
        search_delay_s,
        search_end_s,
        peak_prominence_ratio,
        selected_output_peak_time_s=selected_output_peak_time_s,
    )
    results["Peak-to-peak"] = calculate_vs_and_gmax(
        rx_peak_time_s,
        reference_time_s,
        travel_length_m,
        density_kg_m3,
        "Peak-to-peak",
        note_p2p,
        source_time_s=tx_peak_time_s,
    )

    arr_cc, note_cc, lags_s, corr = cross_correlation_method(
        time_s,
        input_processed,
        output_processed,
        reference_time_s,
        search_delay_s,
        search_end_s,
    )
    results["Cross-correlation"] = calculate_vs_and_gmax(
        arr_cc,
        reference_time_s,
        travel_length_m,
        density_kg_m3,
        "Cross-correlation",
        note_cc,
    )

    results_df = build_results_table(results)
    results_df = add_average_gmax_row(results_df)

    return FileAnalysisOutput(
        file_name=getattr(file_obj, "name", "uploaded_file"),
        results_df=results_df,
        peak_details=peak_details,
        lags_s=lags_s,
        corr=corr,
        time_s=time_s,
        input_processed=input_processed,
        output_processed=output_processed,
        cleaned_df=df,
    )


render_header()

st.markdown(
    """
    **How to use this app**

    1. Upload one or more bender element test files.
    2. Select the time, input, and output signal columns.
    3. Define the time unit, travel length, and bulk density.
    4. Run the analysis and review the plots and output tables.

    The app provides both peak-to-peak and normalized cross-correlation results.
    It also reports the average Gmax obtained from the available methods.
    """
)

st.markdown(
    "This app reads bender element CSV, Excel, and text files and estimates the shear wave velocity and small-strain shear modulus using peak-to-peak and cross-correlation, including the average Gmax from the available methods."
)

analysis_mode = st.radio("Select analysis mode", options=["Single file", "Multiple files"], horizontal=True)

if analysis_mode == "Single file":
    uploaded_file = st.file_uploader("Upload one bender element file", type=["csv", "xlsx", "xls", "txt"], accept_multiple_files=False)
    file_list = [uploaded_file] if uploaded_file is not None else []
else:
    uploaded_files = st.file_uploader("Upload multiple bender element files", type=["csv", "xlsx", "xls", "txt"], accept_multiple_files=True)
    file_list = uploaded_files if uploaded_files is not None else []

if not file_list:
    st.info("Upload file(s) to start the analysis.")
    st.stop()

try:
    preview_df, preview_units = load_be_file(file_list[0])
except Exception as exc:
    st.error(f"Could not read the uploaded file: {exc}")
    st.stop()

if preview_df.empty:
    st.error("No usable numeric data were found after cleaning the uploaded file.")
    st.stop()

st.subheader("Preview of cleaned data")
st.dataframe(preview_df.head(10), use_container_width=True)

with st.expander("Detected units row", expanded=False):
    if preview_units:
        st.write(preview_units)
    else:
        st.write("No units row was detected automatically.")

columns = list(preview_df.columns)
col1, col2, col3 = st.columns(3)
with col1:
    time_col = st.selectbox("Select the time column", options=columns, index=0)
with col2:
    input_col = st.selectbox("Select the transmitted/input signal column", options=columns, index=1 if len(columns) > 1 else 0)
with col3:
    output_col = st.selectbox("Select the received/output signal column", options=columns, index=2 if len(columns) > 2 else min(1, len(columns) - 1))

st.subheader("Test information")
meta1, meta2 = st.columns(2)
with meta1:
    time_unit = st.selectbox("Time unit in the file", options=["ms", "us", "ns", "s"], index=0)
    travel_length_mm = st.number_input("Travel length L (mm)", min_value=0.001, value=50.0, step=1.0, format="%.3f")
with meta2:
    density_kg_m3 = st.number_input("Bulk density ρ (kg/m³)", min_value=0.001, value=2000.0, step=10.0, format="%.3f")

selected_output_peak_time_s = None
manual_peak_candidates_df = None

if analysis_mode == "Single file":
    st.subheader("Peak-to-peak output peak selection")
    p2p_selection_mode = st.radio(
        "Received/output peak used for the peak-to-peak method",
        options=["Automatic: highest positive output peak", "Manual: select the output peak from candidates"],
        horizontal=False,
    )

    if p2p_selection_mode.startswith("Manual"):
        try:
            (
                manual_peak_candidates_df,
                candidate_time_s,
                candidate_input_signal,
                candidate_output_signal,
                candidate_input_details,
            ) = get_peak_to_peak_candidates(
                uploaded_file,
                time_col,
                input_col,
                output_col,
                time_unit,
            )

            if manual_peak_candidates_df.empty:
                st.warning("No candidate positive output peaks were detected in the current search window. The app will use the automatic interpretation.")
            else:
                st.write("Select the received/output peak that represents the first reliable arrival. This avoids using a later maximum peak when the first arrival is smaller.")
                st.dataframe(manual_peak_candidates_df, use_container_width=True)

                candidate_options = [
                    f"Peak {int(row['Peak number'])}: {row['Time (ms)']:.6f} ms, amplitude = {row['Amplitude']:.6g} ({row['Candidate label']})"
                    for _, row in manual_peak_candidates_df.iterrows()
                ]
                selected_candidate_label = st.selectbox("Select output peak for peak-to-peak calculation", options=candidate_options)
                selected_candidate_index = candidate_options.index(selected_candidate_label)
                selected_output_peak_time_s = float(manual_peak_candidates_df.iloc[selected_candidate_index]["Time (ms)"]) * 1e-3

                fig_candidates = make_peak_candidate_plot(
                    candidate_time_s,
                    candidate_input_signal,
                    candidate_output_signal,
                    candidate_input_details,
                    manual_peak_candidates_df,
                    3.0,
                )
                st.pyplot(fig_candidates)
                plt.close(fig_candidates)
        except Exception as exc:
            st.warning(f"Manual peak selection could not be prepared: {exc}")

run_analysis = st.button("Run analysis", type="primary")
if not run_analysis:
    st.stop()

analysis_outputs: List[FileAnalysisOutput] = []
all_results_tables: List[pd.DataFrame] = []
cleaned_data_map: Dict[str, pd.DataFrame] = {}
plot_files: Dict[str, bytes] = {}
errors: List[str] = []

for file_obj in file_list:
    try:
        analysis_output = analyze_file(
            file_obj,
            time_col,
            input_col,
            output_col,
            time_unit,
            travel_length_mm,
            density_kg_m3,
            selected_output_peak_time_s=selected_output_peak_time_s if analysis_mode == "Single file" else None,
        )

        results_df = analysis_output.results_df.copy()
        results_df.insert(0, "File", analysis_output.file_name)
        analysis_output.results_df = results_df

        all_results_tables.append(results_df)
        analysis_outputs.append(analysis_output)
        cleaned_data_map[analysis_output.file_name] = analysis_output.cleaned_df

        # Prepare file-safe names for exported plots
        safe_stem = sanitize_file_stem(analysis_output.file_name)

        fig_peak = make_peak_to_peak_plot(
            analysis_output.time_s,
            analysis_output.input_processed,
            analysis_output.output_processed,
            analysis_output.peak_details,
            3.0,
        )
        plot_files[f"{safe_stem}_peak_to_peak.png"] = fig_to_png_bytes(fig_peak)
        plt.close(fig_peak)

        if analysis_output.lags_s is not None and analysis_output.corr is not None:
            fig_cc = make_cross_correlation_plot(analysis_output.lags_s, analysis_output.corr)
            plot_files[f"{safe_stem}_cross_correlation.png"] = fig_to_png_bytes(fig_cc)
            plt.close(fig_cc)
    except Exception as exc:
        errors.append(f"{getattr(file_obj, 'name', 'uploaded_file')}: {exc}")

if errors:
    for message in errors:
        st.error(message)

if not all_results_tables:
    st.stop()

results_table_all = pd.concat(all_results_tables, ignore_index=True)

# Round numerical results to 3 decimal places for clarity
numeric_cols = [
    "Arrival time (ms)",
    "Travel time (ms)",
    "Shear wave velocity Vs (m/s)",
    "Small-strain shear modulus Gmax (MPa)",
]
for col in numeric_cols:
    if col in results_table_all.columns:
        results_table_all[col] = pd.to_numeric(results_table_all[col], errors="coerce").round(3)


st.subheader("Results table")
st.dataframe(results_table_all, use_container_width=True)

if analysis_mode == "Single file":
    st.subheader("Plots")
    single_output = analysis_outputs[0]
    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        fig_peak_display = make_peak_to_peak_plot(
            single_output.time_s,
            single_output.input_processed,
            single_output.output_processed,
            single_output.peak_details,
            3.0,
        )
        st.pyplot(fig_peak_display)
        peak_plot_bytes = fig_to_png_bytes(fig_peak_display)
        plt.close(fig_peak_display)

    with plot_col2:
        cc_plot_bytes = None
        if single_output.lags_s is not None and single_output.corr is not None:
            fig_cc_display = make_cross_correlation_plot(single_output.lags_s, single_output.corr)
            st.pyplot(fig_cc_display)
            cc_plot_bytes = fig_to_png_bytes(fig_cc_display)
            plt.close(fig_cc_display)
        else:
            st.info("Cross-correlation plot is unavailable.")
else:
    st.subheader("Batch plots")
    for analysis_output in analysis_outputs:
        with st.expander(f"Plots for {analysis_output.file_name}", expanded=False):
            batch_col1, batch_col2 = st.columns(2)
            with batch_col1:
                fig_peak_display = make_peak_to_peak_plot(
                    analysis_output.time_s,
                    analysis_output.input_processed,
                    analysis_output.output_processed,
                    analysis_output.peak_details,
                    3.0,
                )
                st.pyplot(fig_peak_display)
                plt.close(fig_peak_display)
            with batch_col2:
                if analysis_output.lags_s is not None and analysis_output.corr is not None:
                    fig_cc_display = make_cross_correlation_plot(analysis_output.lags_s, analysis_output.corr)
                    st.pyplot(fig_cc_display)
                    plt.close(fig_cc_display)
                else:
                    st.info("Cross-correlation plot is unavailable.")
    peak_plot_bytes = None
    cc_plot_bytes = None

excel_bytes = to_excel_bytes(results_table_all, cleaned_data_map)
if excel_bytes is not None:
    st.download_button(
        label="Download results as Excel",
        data=excel_bytes,
        file_name="bender_element_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.info("Excel export is unavailable in the current environment. Install openpyxl to enable Excel download.")

csv_bytes = results_table_all.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download method results as CSV",
    data=csv_bytes,
    file_name="bender_element_results.csv",
    mime="text/csv",
)

if analysis_mode == "Single file":
    if peak_plot_bytes is not None:
        st.download_button(
            label="Download peak-to-peak plot (PNG)",
            data=peak_plot_bytes,
            file_name="peak_to_peak_plot.png",
            mime="image/png",
        )

    if cc_plot_bytes is not None:
        st.download_button(
            label="Download cross-correlation plot (PNG)",
            data=cc_plot_bytes,
            file_name="cross_correlation_plot.png",
            mime="image/png",
        )
else:
    if plot_files:
        plot_zip_bytes = create_plot_zip(plot_files)
        st.download_button(
            label="Download all batch plots (ZIP)",
            data=plot_zip_bytes,
            file_name="bender_element_batch_plots.zip",
            mime="application/zip",
        )
