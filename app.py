import streamlit as st
import numpy as np
import pandas as pd
import parselmouth
from textgrid import TextGrid
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
from io import BytesIO
import zipfile
from datetime import datetime

plt.rcParams["figure.dpi"] = 120


# =========================
# Defaults (match your notebook)
# =========================
SEG_TIER_NAME_DEFAULT = "MAU"
ORT_TIER_NAME_DEFAULT = "ORT-MAU"

T_LABELS_DEFAULT = ("t", "t?", "t_?", "tʔ", "?", "Q?")

SAMPA_VOWELS_DEFAULT = {
    "i", "i:", "I",
    "e", "e:", "E",
    "{", "V", "A",
    "Q", "O", "o", "o:",
    "U", "u", "u:",
    "@", "3", "3:",
    "2", "9",
    "Y", "y"
}

SAMPA_SONORANTS_DEFAULT = {
    "m", "n", "N",
    "l", "l_", "L",
    "r", "r\\", "4",
    "j", "w"
}

FRAME_STEP_DEFAULT = 0.005       # 5 ms
WINDOW_LENGTH_DEFAULT = 0.030    # 30 ms window
F0_FLOOR_DEFAULT = 50
F0_CEIL_DEFAULT = 500

YMIN_DEFAULT = 0.0
YMAX_DEFAULT = 1.0

NORM_T_MAX_DEFAULT = 10.0

BASELINE_FRAC_DEFAULT = 0.2
DROP_MARGIN_DEFAULT = 0.03


# =========================
# TextGrid helpers
# =========================
def get_tier(tg: TextGrid, tier_name: str):
    target = (tier_name or "").strip().lower()
    for tier in tg.tiers:
        if tier.name and tier.name.strip().lower() == target:
            return tier
    available = [repr(t.name) for t in tg.tiers]
    raise ValueError(f"Tier {tier_name!r} not found. Available tiers: {', '.join(available)}")

def normalize_sampa(lab: str) -> str:
    lab = (lab or "").strip()
    for ch in ('"', "%", ":", "'", " "):
        lab = lab.replace(ch, "")
    return lab

def is_vowel(label: str, vowels: set) -> bool:
    return normalize_sampa(label) in vowels

def is_sonorant(label: str, sonorants: set) -> bool:
    return normalize_sampa(label) in sonorants

def get_overlapping_labels(tg: TextGrid, tier_name: str, start: float, end: float) -> str:
    tier = get_tier(tg, tier_name)
    labs = []
    for intv in tier.intervals:
        if intv.maxTime > start and intv.minTime < end:
            lab = (intv.mark or "").strip()
            if lab:
                labs.append(lab)
    return " ".join(labs)

def ort_full_text(tg: TextGrid, tier_name: str) -> str:
    tier = get_tier(tg, tier_name)
    toks = []
    for intv in tier.intervals:
        lab = (intv.mark or "").strip()
        if lab:
            toks.append(lab)
    return " ".join(toks)

def ort_intervals_rows(tg: TextGrid, tier_name: str, file_name: str):
    tier = get_tier(tg, tier_name)
    rows = []
    for k, intv in enumerate(tier.intervals, start=1):
        rows.append({
            "file_name": file_name,
            "ort_interval_index": k,
            "ort_interval_start_s": float(intv.minTime),
            "ort_interval_end_s": float(intv.maxTime),
            "ort_interval_label": (intv.mark or "").strip(),
        })
    return rows


# =========================
# Token detection: V–t–V and V–t–C[+son]
# =========================
def find_vt_like_tokens(tg: TextGrid,
                        seg_tier_name: str,
                        t_labels,
                        vowels: set,
                        sonorants: set):
    tier = get_tier(tg, seg_tier_name)
    ints = tier.intervals
    tokens = []

    for i, seg in enumerate(ints):
        if seg.mark not in t_labels:
            continue
        if i == 0 or i == len(ints) - 1:
            continue

        left = ints[i - 1]
        right = ints[i + 1]

        # left must be vowel
        if not is_vowel(left.mark, vowels):
            continue

        # right can be vowel or sonorant
        if is_vowel(right.mark, vowels):
            pattern = "V_t_V"
        elif is_sonorant(right.mark, sonorants):
            pattern = "V_t_Cson"
        else:
            continue

        tokens.append({
            "prev_label": left.mark,
            "t_label": seg.mark,
            "next_label": right.mark,
            "start": float(left.minTime),
            "t_start": float(seg.minTime),
            "t_end": float(seg.maxTime),
            "end": float(right.maxTime),
            "pattern": pattern,
        })
    return tokens


# =========================
# Acoustic tracks
# =========================
def peak_autocorr_track(sound: parselmouth.Sound,
                        start: float,
                        end: float,
                        frame_step: float,
                        window_length: float,
                        f0_floor: float,
                        f0_ceil: float):
    """
    Peak autocorr per frame (0..1), peak searched only over pitch-period lags:
      lag in [1/f0_ceil .. 1/f0_floor]
    Returns: times_abs, peaks
    """
    sr = sound.sampling_frequency
    snd = sound.extract_part(from_time=start, to_time=end, preserve_times=True)
    x = snd.values[0]
    t0 = snd.xmin

    half = int(window_length * sr / 2)

    min_lag = 1.0 / f0_ceil
    max_lag = 1.0 / f0_floor
    lo = max(1, int(min_lag * sr))
    hi = max(lo + 1, int(max_lag * sr))

    times = np.arange(start + window_length / 2,
                      end - window_length / 2,
                      frame_step)

    peaks = []

    for ctime in times:
        c = int((ctime - t0) * sr)
        s = c - half
        e = c + half

        if s < 0 or e > len(x):
            peaks.append(np.nan)
            continue

        seg = x[s:e]
        seg = seg - np.mean(seg)
        if np.allclose(seg, 0):
            peaks.append(np.nan)
            continue

        segw = seg * np.hanning(len(seg))

        acf = np.correlate(segw, segw, mode="full")
        acf = acf[len(acf) // 2:]

        if acf[0] == 0:
            peaks.append(np.nan)
            continue

        acf = acf / acf[0]

        hi2 = min(hi, len(acf) - 1)
        lo2 = min(lo, hi2 - 1)
        if hi2 <= lo2:
            peaks.append(np.nan)
            continue

        region = acf[lo2:hi2]
        peaks.append(float(np.nanmax(region)))

    return np.array(times), np.array(peaks)


def praat_voicing_strength(sound: parselmouth.Sound,
                           start: float,
                           end: float,
                           time_step: float,
                           f0_floor: float,
                           f0_ceil: float):
    """
    Praat autocorrelation pitch 'strength' (0..1) per frame.
    """
    snd = sound.extract_part(from_time=start, to_time=end, preserve_times=True)
    pitch = snd.to_pitch_ac(time_step=time_step, pitch_floor=f0_floor, pitch_ceiling=f0_ceil)
    times = np.array(pitch.xs())
    strength = np.array(pitch.selected_array["strength"])
    return times, strength


def normalize_time(times_abs: np.ndarray, start: float, end: float, tmax: float):
    dur = end - start
    if dur <= 0:
        return np.full_like(times_abs, np.nan, dtype=float)
    return tmax * (times_abs - start) / dur


def detect_decline_recovery(times_abs: np.ndarray,
                            track: np.ndarray,
                            t_start: float,
                            t_end: float,
                            token_start: float,
                            token_end: float,
                            baseline_frac: float,
                            drop_margin: float):
    """
    Automatic dip window:
      - baseline from early+late portions of VtX
      - threshold = baseline - drop_margin
      - find minimum inside /t/
      - expand left/right while below threshold
    """
    m = ~np.isnan(times_abs) & ~np.isnan(track)
    t = times_abs[m]
    y = track[m]
    if t.size < 10:
        return None, None

    dur = token_end - token_start
    b = baseline_frac * dur
    base_mask = ((t >= token_start) & (t <= token_start + b)) | ((t >= token_end - b) & (t <= token_end))
    base_vals = y[base_mask]

    if base_vals.size < 5:
        ys = np.sort(y)
        base_vals = ys[int(0.8 * len(ys)):] if len(ys) >= 10 else ys
        if base_vals.size < 5:
            return None, None

    baseline = float(np.nanmean(base_vals))
    thr = baseline - drop_margin

    t_mask = (t >= t_start) & (t <= t_end)
    if not np.any(t_mask):
        return None, None
    idxs = np.where(t_mask)[0]
    idx_min = idxs[np.nanargmin(y[t_mask])]

    i = idx_min
    j = idx_min
    while i > 0 and y[i] < thr:
        i -= 1
    while j < len(y) - 1 and y[j] < thr:
        j += 1

    dip_start = float(t[i])
    dip_end = float(t[j])

    dip_start = max(dip_start, token_start)
    dip_end = min(dip_end, token_end)
    if dip_end <= dip_start:
        return None, None
    return dip_start, dip_end


# =========================
# Plotting (PNG)
# =========================
def make_token_figure(sound: parselmouth.Sound,
                      token: dict,
                      t_ac: np.ndarray, peak: np.ndarray,
                      t_pr: np.ndarray, strength: np.ndarray,
                      dip_start, dip_end,
                      y_min: float, y_max: float,
                      title: str):
    snd = sound.extract_part(from_time=token["start"], to_time=token["end"], preserve_times=True)
    x = snd.values[0]
    t_abs = np.linspace(snd.xmin, snd.xmax, len(x))
    t_rel = t_abs - token["start"]

    ac_rel = t_ac - token["start"]
    pr_rel = t_pr - token["start"]

    t_t_start = token["t_start"] - token["start"]
    t_t_end = token["t_end"] - token["start"]

    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    ax[0].plot(t_rel, x)
    ax[0].axvspan(t_t_start, t_t_end, alpha=0.2)
    if dip_start is not None and dip_end is not None:
        ax[0].axvspan(dip_start - token["start"], dip_end - token["start"], alpha=0.15)
    ax[0].set_ylabel("Amplitude")
    ax[0].set_title(title)

    ax[1].plot(ac_rel, peak, label="Peak autocorr (pitch-lag)")
    ax[1].plot(pr_rel, strength, label="Praat voicing strength")
    ax[1].axvspan(t_t_start, t_t_end, alpha=0.2)
    if dip_start is not None and dip_end is not None:
        ax[1].axvspan(dip_start - token["start"], dip_end - token["start"], alpha=0.15)

    ax[1].set_ylim(y_min, y_max)
    ax[1].set_xlabel("Time (s, relative to V1 onset)")
    ax[1].set_ylabel("0..1")
    ax[1].legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


# =========================
# UI
# =========================
st.set_page_config(page_title="Autocorr glottalization export", layout="wide")
st.title("Autocorrelation export: V–t–(V / C[+son]) glottalization")

st.write(
    "Upload **pairs** of `*.wav` + `*.TextGrid` with the **same filename stem** (e.g., `item01.wav` + `item01.TextGrid`). "
    "The app detects **V–t–V** and **V–t–C[+sonorant]** tokens (from MAU by default), computes autocorrelation-based measures, "
    "creates one PNG per token, and exports CSVs + a ZIP for the whole upload."
)

with st.sidebar:
    st.header("Settings")

    seg_tier_name = st.text_input("Segment tier name", value=SEG_TIER_NAME_DEFAULT)
    ort_tier_name = st.text_input("Orthographic tier name", value=ORT_TIER_NAME_DEFAULT)

    t_labels = st.text_input("T labels (comma-separated)", value=",".join(T_LABELS_DEFAULT))
    t_labels = tuple([x.strip() for x in t_labels.split(",") if x.strip()])

    st.subheader("Inventories (SAMPA)")
    vowels_text = st.text_area("Vowels (space-separated)", value=" ".join(sorted(SAMPA_VOWELS_DEFAULT)))
    son_text = st.text_area("Sonorants (space-separated)", value=" ".join(sorted(SAMPA_SONORANTS_DEFAULT)))

    vowels = set([x.strip() for x in vowels_text.split() if x.strip()])
    sonorants = set([x.strip() for x in son_text.split() if x.strip()])

    st.subheader("Analysis parameters")
    frame_step = st.number_input("Frame step (s)", value=float(FRAME_STEP_DEFAULT), min_value=0.001, max_value=0.05, step=0.001, format="%.3f")
    window_length = st.number_input("Window length (s)", value=float(WINDOW_LENGTH_DEFAULT), min_value=0.010, max_value=0.080, step=0.005, format="%.3f")
    f0_floor = st.number_input("F0 floor (Hz)", value=int(F0_FLOOR_DEFAULT), min_value=20, max_value=200)
    f0_ceil = st.number_input("F0 ceiling (Hz)", value=int(F0_CEIL_DEFAULT), min_value=200, max_value=1000)

    st.subheader("Dip detection")
    baseline_frac = st.number_input("Baseline fraction", value=float(BASELINE_FRAC_DEFAULT), min_value=0.05, max_value=0.5, step=0.05, format="%.2f")
    drop_margin = st.number_input("Drop margin", value=float(DROP_MARGIN_DEFAULT), min_value=0.0, max_value=0.3, step=0.01, format="%.2f")

    st.subheader("Plot y-axis")
    y_min = st.number_input("y-min", value=float(YMIN_DEFAULT), min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
    y_max = st.number_input("y-max", value=float(YMAX_DEFAULT), min_value=0.0, max_value=1.0, step=0.05, format="%.2f")

    st.subheader("Time normalization")
    norm_tmax = st.number_input("Normalized time max", value=float(NORM_T_MAX_DEFAULT), min_value=1.0, max_value=100.0, step=1.0, format="%.0f")

    st.subheader("Exports")
    export_frames = st.checkbox("Also export frame-level CSV (can be large)", value=True)
    export_ort_intervals = st.checkbox("Also export ORT-MAU intervals CSV", value=True)


uploaded_files = st.file_uploader(
    "Upload .wav and .TextGrid files (multiple)",
    type=["wav", "WAV", "TextGrid", "textgrid"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Waiting for files…")
    st.stop()

# Save uploads to a temp folder and pair by stem
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)

    file_map = {}  # stem -> {"wav": path, "tg": path}

    for up in uploaded_files:
        name = up.name
        suffix = Path(name).suffix.lower()
        stem = Path(name).stem
        dest = tmpdir / name
        dest.write_bytes(up.read())

        entry = file_map.setdefault(stem, {})
        if suffix == ".wav":
            entry["wav"] = dest
        elif suffix == ".textgrid":
            entry["tg"] = dest

    pairs = [(stem, d["wav"], d["tg"]) for stem, d in file_map.items() if "wav" in d and "tg" in d]
    missing = [stem for stem, d in file_map.items() if not ("wav" in d and "tg" in d)]

    if missing:
        st.warning(f"Missing pairs for stems: {', '.join(sorted(missing))}")

    if not pairs:
        st.error("No complete wav+TextGrid pairs found.")
        st.stop()

    st.success(f"Found {len(pairs)} complete pairs.")

    token_rows = []
    frame_rows = []
    ort_rows = []

    # PNGs in memory for zipping
    png_items = []  # list of (zip_path, bytes)

    for stem, wav_path, tg_path in sorted(pairs, key=lambda x: x[0]):
        st.markdown(f"### File: `{stem}`")

        sound = parselmouth.Sound(str(wav_path))
        tg = TextGrid.fromFile(str(tg_path))

        # file metadata (lightweight)
        sr = float(sound.sampling_frequency)
        wav_dur = float(sound.xmax - sound.xmin)

        try:
            ort_full = ort_full_text(tg, ort_tier_name)
        except Exception:
            ort_full = ""

        if export_ort_intervals:
            try:
                ort_rows.extend(ort_intervals_rows(tg, ort_tier_name, stem))
            except Exception:
                pass

        # detect tokens
        try:
            tokens = find_vt_like_tokens(tg, seg_tier_name, t_labels, vowels, sonorants)
        except Exception as e:
            st.error(f"Token detection failed for {stem}: {e}")
            continue

        st.write(f"Tokens found: **{len(tokens)}**")
        if not tokens:
            continue

        for tok_i, token in enumerate(tokens, start=1):
            t_ac, peak = peak_autocorr_track(
                sound, token["start"], token["end"],
                frame_step=frame_step,
                window_length=window_length,
                f0_floor=f0_floor,
                f0_ceil=f0_ceil
            )
            t_pr, strength = praat_voicing_strength(
                sound, token["start"], token["end"],
                time_step=frame_step,
                f0_floor=f0_floor,
                f0_ceil=f0_ceil
            )

            # interpolate strength to autocorr times (for frame-level export)
            strength_i = np.interp(t_ac, t_pr, strength, left=np.nan, right=np.nan)

            # ORT overlap for token span
            try:
                ort_tok = get_overlapping_labels(tg, ort_tier_name, token["start"], token["end"])
            except Exception:
                ort_tok = ""

            # dip window
            dip_start, dip_end = detect_decline_recovery(
                t_ac, peak,
                token["t_start"], token["t_end"],
                token["start"], token["end"],
                baseline_frac=baseline_frac,
                drop_margin=drop_margin
            )

            # summaries
            peak_mean = float(np.nanmean(peak))
            peak_min = float(np.nanmin(peak))
            peak_max = float(np.nanmax(peak))

            # normalized time 0..norm_tmax
            t_norm = normalize_time(t_ac, token["start"], token["end"], tmax=float(norm_tmax))

            # PNG
            title = f"{stem} | tok {tok_i}: {token['prev_label']}–{token['t_label']}–{token['next_label']} ({token['pattern']})"
            fig = make_token_figure(sound, token, t_ac, peak, t_pr, strength, dip_start, dip_end, y_min, y_max, title)
            png_bytes = fig_to_png_bytes(fig)

            png_name = f"{stem}_tok{tok_i:03d}.png"
            png_items.append((f"images/{stem}/{png_name}", png_bytes))

            # show preview (optional but nice)
            with st.expander(f"Preview {png_name}", expanded=False):
                st.image(png_bytes)

            # token-level row
            token_rows.append({
                "file_name": stem,
                "sampling_rate_hz": sr,
                "wav_duration_s": wav_dur,

                "token_index": tok_i,
                "pattern": token["pattern"],

                "prev_label": token["prev_label"],
                "t_label": token["t_label"],
                "next_label": token["next_label"],

                "vtX_start_s": float(token["start"]),
                "t_start_s": float(token["t_start"]),
                "t_end_s": float(token["t_end"]),
                "vtX_end_s": float(token["end"]),
                "vtX_dur_s": float(token["end"] - token["start"]),
                "t_dur_s": float(token["t_end"] - token["t_start"]),

                "dip_start_s": float(dip_start) if dip_start is not None else np.nan,
                "dip_end_s": float(dip_end) if dip_end is not None else np.nan,

                "peak_autocorr_mean": peak_mean,
                "peak_autocorr_min": peak_min,
                "peak_autocorr_max": peak_max,

                "ort_mau_full_text": ort_full,
                "ort_mau_token_text": ort_tok,

                "png_path_in_zip": f"images/{stem}/{png_name}",
            })

            # frame-level rows
            if export_frames:
                in_t = (t_ac >= token["t_start"]) & (t_ac <= token["t_end"])
                for tt, tn, pk, stg, flag in zip(t_ac, t_norm, peak, strength_i, in_t):
                    frame_rows.append({
                        "file_name": stem,
                        "token_index": tok_i,
                        "time_abs_s": float(tt),
                        "time_norm_0_to_T": float(tn),
                        "peak_autocorr": float(pk) if not np.isnan(pk) else np.nan,
                        "praat_strength": float(stg) if not np.isnan(stg) else np.nan,
                        "in_t_interval": bool(flag),
                        "pattern": token["pattern"],
                    })

    if not token_rows:
        st.error("No tokens detected in the uploaded data.")
        st.stop()

    df_tokens = pd.DataFrame(token_rows)
    st.markdown("## Token-level results")
    st.dataframe(df_tokens, use_container_width=True)

    # CSV bytes
    tokens_csv = df_tokens.to_csv(index=False).encode("utf-8")

    # optional CSVs
    frames_csv = None
    if export_frames and frame_rows:
        df_frames = pd.DataFrame(frame_rows)
        frames_csv = df_frames.to_csv(index=False).encode("utf-8")
        st.markdown("## Frame-level results (preview)")
        st.dataframe(df_frames.head(200), use_container_width=True)

    ort_csv = None
    if export_ort_intervals and ort_rows:
        df_ort = pd.DataFrame(ort_rows)
        ort_csv = df_ort.to_csv(index=False).encode("utf-8")

    # Download buttons
    st.download_button(
        "Download tokens.csv",
        data=tokens_csv,
        file_name="tokens.csv",
        mime="text/csv",
    )

    if frames_csv is not None:
        st.download_button(
            "Download frames_long.csv",
            data=frames_csv,
            file_name="frames_long.csv",
            mime="text/csv",
        )

    if ort_csv is not None:
        st.download_button(
            "Download ort_mau_intervals.csv",
            data=ort_csv,
            file_name="ort_mau_intervals.csv",
            mime="text/csv",
        )

    # Zip everything (PNGs + CSVs)
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # PNGs
        for zpath, data in png_items:
            z.writestr(zpath, data)

        # CSVs
        z.writestr("tokens.csv", tokens_csv)
        if frames_csv is not None:
            z.writestr("frames_long.csv", frames_csv)
        if ort_csv is not None:
            z.writestr("ort_mau_intervals.csv", ort_csv)

        # Manifest
        manifest = {
            "created_iso": datetime.now().isoformat(timespec="seconds"),
            "pairs_found": len(pairs),
            "tokens_found": len(df_tokens),
            "export_frames": export_frames,
            "export_ort_intervals": export_ort_intervals,
        }
        z.writestr("manifest.json", pd.Series(manifest).to_json())

    zip_buf.seek(0)
    st.download_button(
        "Download ALL outputs as ZIP",
        data=zip_buf,
        file_name="autocorr_export.zip",
        mime="application/zip",
    )
