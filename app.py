import streamlit as st
import numpy as np
import pandas as pd
import parselmouth
from textgrid import TextGrid
from pathlib import Path
from io import BytesIO
import tempfile
import matplotlib.pyplot as plt

# =========================
# Configuration matplotlib
# =========================

plt.rcParams["figure.dpi"] = 120

# =========================
# Constantes d'analyse
# =========================

TIER_NAME = "MAU"          # tier segmental
ORT_TIER_NAME = "ORT-MAU"  # tier orthographique

T_LABELS = ("t", "t?", "t_?", "tʔ", "?", "Q?")

SAMPA_VOWELS = {
    "i", "i:", "I",
    "e", "e:", "E",
    "{", "V", "A",
    "Q", "O", "o", "o:",
    "U", "u", "u:",
    "@", "3", "3:",
    "2", "9",
    "Y", "y"
}

SONORANT_SAMPA = {
    "m", "n", "N",
    "l", "l_", "L",
    "r", "r\\", "4",
    "j", "w"
}

FRAME_STEP = 0.005
WINDOW_LENGTH = 0.030
MAX_LAG = 0.020


# =========================
# Helpers TextGrid / SAMPA
# =========================

def get_tier(tg: TextGrid, tier_name: str):
    """Je récupère le tier demandé, en ignorant casse / espaces."""
    target = tier_name.strip().lower()
    for tier in tg.tiers:
        if tier.name and tier.name.strip().lower() == target:
            return tier
    raise ValueError(f"Tier {tier_name!r} introuvable.")


def normalize_sampa(lab: str) -> str:
    """Je nettoie un label SAMPA pour le comparer à mon inventaire."""
    lab = lab.strip()
    for ch in ('"', "%", ":", "'", " "):
        lab = lab.replace(ch, "")
    return lab


def is_vowel(label: str) -> bool:
    """Test rapide : est-ce que c'est une voyelle ?"""
    return normalize_sampa(label) in SAMPA_VOWELS


def is_sonorant(label: str) -> bool:
    """Test rapide : est-ce que c'est une consonne sonante ?"""
    return normalize_sampa(label) in SONORANT_SAMPA


def get_overlapping_labels(tg: TextGrid, tier_name: str, start: float, end: float):
    """Je récupère les labels ORT-MAU qui recouvrent [start, end]."""
    tier = get_tier(tg, tier_name)
    labs = []
    for intv in tier.intervals:
        if intv.maxTime > start and intv.minTime < end:
            if intv.mark.strip():
                labs.append(intv.mark.strip())
    return " ".join(labs)


# =========================
# Détection V t V / V t Cson
# =========================

def find_vt_like_tokens(tg: TextGrid,
                        tier_name: str = TIER_NAME,
                        t_labels=T_LABELS):
    """
    Je cherche uniquement :
      - V t V
      - V t C[+son]

    Je renvoie une liste de dicts (un dict par token).
    """
    tier = get_tier(tg, tier_name)
    ints = tier.intervals
    tokens = []

    for i, seg in enumerate(ints):
        if seg.mark not in t_labels:
            continue

        # /t/ doit avoir un voisin à gauche et à droite
        if i == 0 or i == len(ints) - 1:
            continue

        left = ints[i - 1]
        right = ints[i + 1]

        # Contrainte : voyelle à gauche
        if not is_vowel(left.mark):
            continue

        # À droite : voyelle ou C sonante
        if is_vowel(right.mark):
            pattern = "V_t_V"
        elif is_sonorant(right.mark):
            pattern = "V_t_Cson"
        else:
            continue

        tokens.append({
            "prev_label": left.mark,
            "t_label": seg.mark,
            "next_label": right.mark,
            "start": left.minTime,
            "t_start": seg.minTime,
            "t_end": seg.maxTime,
            "end": right.maxTime,
            "pattern": pattern,
        })

    return tokens


# =========================
# Autocorr courte
# =========================

def short_time_autocorr_track(sound: parselmouth.Sound,
                              start: float,
                              end: float,
                              frame_step: float = FRAME_STEP,
                              window_length: float = WINDOW_LENGTH,
                              max_lag: float = MAX_LAG):
    """
    Je calcule le peak d'autocorrélation sur un segment [start, end].
    Retourne :
      - times (temps absolus)
      - peaks (peak d'autocorr par frame)
    """
    sr = sound.sampling_frequency
    snd = sound.extract_part(from_time=start, to_time=end, preserve_times=True)
    x = snd.values[0]
    t0 = snd.xmin

    half = int(window_length * sr / 2)
    maxlag = int(max_lag * sr)

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

        win = np.hanning(len(seg))
        segw = seg * win

        acf = np.correlate(segw, segw, mode="full")
        acf = acf[len(acf) // 2:]

        if acf[0] == 0:
            peaks.append(np.nan)
            continue

        acf = acf / acf[0]

        m = min(maxlag, len(acf) - 1)
        if m <= 1:
            peaks.append(np.nan)
            continue

        pos = acf[1:m]
        peaks.append(np.max(pos) if len(pos) else np.nan)

    return np.array(times), np.array(peaks)


# =========================
# Plot d'un token -> figure
# =========================
def plot_token(sound: parselmouth.Sound,
               token: dict,
               times: np.ndarray,
               peaks: np.ndarray,
               title_prefix: str,
               info_line: str):
    """
    Version app web : texte d’info EN BAS pour éviter toute superposition
    avec le titre ou le graphe. Compatible avec Streamlit.
    """

    # Extraction du segment sonore correspondant au token
    snd = sound.extract_part(from_time=token["start"],
                             to_time=token["end"],
                             preserve_times=True)

    x = snd.values[0]  # signal
    t_axis = np.linspace(snd.xmin, snd.xmax, len(x)) - token["start"]
    times_rel = times - token["start"]

    t_t_start = token["t_start"] - token["start"]
    t_t_end = token["t_end"] - token["start"]

    # Figure
    fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)

    # ---------- Waveform ----------
    ax[0].plot(t_axis, x)
    ax[0].axvspan(t_t_start, t_t_end, alpha=0.2)
    ax[0].set_ylabel("Amplitude")

    # ---------- Peak autocorr ----------
    ax[1].plot(times_rel, peaks)
    ax[1].axvspan(t_t_start, t_t_end, alpha=0.2)
    ax[1].set_ylim(0.5, 1.0)
    ax[1].set_xlabel("Temps relatif (s)")
    ax[1].set_ylabel("Peak autocorr")

    # ---------- Titre principal ----------
    fig.suptitle(f"{title_prefix} {token['pattern']}", fontsize=12)

    # ---------- Texte d’info EN BAS ----------
    fig.text(
        0.5, 0.02,              # position centrale en bas
        info_line,
        ha="center",
        va="bottom",
        fontsize=7
    )

    # ---------- Ajustement du layout ----------
    # On réserve 6% d’espace en bas pour éviter que le texte touche l’axe X
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    return fig




# =========================
# Traitement d'une paire wav + TextGrid
# =========================

def process_pair(word: str,
                 wav_path: Path,
                 tg_path: Path,
                 rows: list):
    """
    Traitement d'une paire wav + TextGrid :
    - extraction des tokens VtV / VtCson
    - calcul autocorr
    - génération figure
    - ajout ligne au tableau pour CSV
    """

    st.markdown(f"### Mot **{word}** – fichier `{wav_path.name}`")

    # Charger audio + TextGrid
    sound = parselmouth.Sound(str(wav_path))
    tg = TextGrid.fromFile(str(tg_path))

    # Tokens VtV / VtCson
    tokens = find_vt_like_tokens(tg)
    st.write(f"Nombre de tokens VtV / VtCson : {len(tokens)}")

    if not tokens:
        return rows

    for idx, token in enumerate(tokens, start=1):
        # Calcul de l'autocorrélation
        times, peaks = short_time_autocorr_track(
            sound, token["start"], token["end"]
        )

        mean_peak = float(np.nanmean(peaks))
        min_peak = float(np.nanmin(peaks))
        max_peak = float(np.nanmax(peaks))

        # Contexte orthographique
        ort_label = get_overlapping_labels(
            tg, ORT_TIER_NAME, token["start"], token["end"]
        )

        # Construire info_line AVANT l’appel à plot_token
        info_line = (
            f"Token {idx:02d} {token['pattern']}: "
            f"{token['prev_label']} - {token['t_label']} - {token['next_label']} | "
            f"ORT: '{ort_label}' | "
            f"peak_auto mean={mean_peak:.3f}, min={min_peak:.3f}, max={max_peak:.3f}"
        )

        # Log texte
        st.text(info_line)

        # Figure corrigée (texte en bas)
        fig = plot_token(
            sound,
            token,
            times,
            peaks,
            title_prefix=f"Token {idx}",
            info_line=info_line,
        )
        st.pyplot(fig)

        # Ajouter une ligne pour le CSV
        rows.append({
            "word": word,
            "filename": wav_path.name,
            "pattern": token["pattern"],
            "prev_label": token["prev_label"],
            "t_label": token["t_label"],
            "next_label": token["next_label"],
            "start": token["start"],
            "t_start": token["t_start"],
            "t_end": token["t_end"],
            "end": token["end"],
            "duration": token["end"] - token["start"],
            "mean_peak": mean_peak,
            "min_peak": min_peak,
            "max_peak": max_peak,
            "ort_label": ort_label,
        })

    return rows



# =========================
# Interface Streamlit
# =========================

st.title("Analyse V–t–(V / Cson) & glottalisation")

uploaded_files = st.file_uploader(
    "Uploader les fichiers .wav et .TextGrid (plusieurs à la fois)",
    type=["wav", "WAV", "TextGrid", "textgrid"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("En attente de fichiers…")
else:
    # Je stocke tout temporairement dans un dossier
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        file_map = {}  # stem -> {"wav": path, "tg": path}

        for up in uploaded_files:
            name = up.name
            suffix = Path(name).suffix
            stem = Path(name).stem

            dest = tmpdir / name
            dest.write_bytes(up.read())

            entry = file_map.setdefault(stem, {})
            if suffix.lower() == ".wav":
                entry["wav"] = dest
            elif suffix.lower() in (".textgrid", ".textgrid"):
                entry["tg"] = dest

        rows = []

        # Pour chaque stem qui a bien un wav + un TextGrid
        for stem, paths in sorted(file_map.items()):
            if "wav" in paths and "tg" in paths:
                # Ici j'interprète stem comme "word",
                # tu adapteras si tu veux autre chose.
                word = stem
                process_pair(word, paths["wav"], paths["tg"], rows)
            else:
                st.warning(f"Stem `{stem}` incomplet (wav ou TextGrid manquant).")

    # Construction du DataFrame à partir de rows
    if rows:
        df_tokens = pd.DataFrame(rows)
        st.markdown("## Résumé sous forme de DataFrame")
        st.dataframe(df_tokens)

        csv_bytes = df_tokens.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Télécharger le CSV des tokens",
            data=csv_bytes,
            file_name="tokens_vt_glottalisation.csv",
            mime="text/csv",
        )
    else:
        st.warning("Aucun token détecté dans les fichiers fournis.")

# =========================
# Bloc "À propos" en bas de page
# =========================

st.markdown("""
---
### À propos de cette application

Cette application propose un outil léger pour le calcul rapide de l’autocorrélation dans des séquences de type **V–t–V** ou **V–t–C[+son]**, à partir de paires annotées `.wav` + `.TextGrid`.  
Elle permet notamment :

- l’upload sécurisé de fichiers audio et TextGrid,
- l’extraction automatique des contextes VtV et VtC[+sonorant],
- le calcul du **peak autocorrelation** pour chaque token,
- l’affichage des graphiques (waveform + autocorrélation),
- et l’export des résultats sous forme de **fichier CSV**.

Le protocole d’analyse et les mesures sont inspirés de :

> **Garellek, M. (2023).**  
> *Measuring incompleteness: Acoustic correlates of glottal articulations.*  
> *Journal of the International Phonetic Association*, 53(3), 449–474.  
> https://doi.org/10.1017/S002510032200006X

Cette application a été développée à partir d’un notebook Python,  
avec l’aide de l’assistant IA ChatGPT pour l’adaptation du code,  
la structuration du pipeline d’analyse et la création de l’interface Streamlit.
""")
