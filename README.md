# glottalization-streamlit

We developed a Python/Parselmouth workflow to quantify glottalization in intervocalic /t/ contexts using short-time autocorrelation. The initial implementation tracked “peak autocorrelation” per 5-ms frame but (incorrectly) allowed extremely small lags (e.g., 1 sample) when selecting the maximum peak; because adjacent samples are highly similar in most speech signals, this produced peak values near 1.0 even in potentially irregular phonation, masking the expected dips. We corrected the method by restricting the peak search to lags corresponding to plausible pitch periods (defined by an F0 floor and ceiling, e.g. 50–500 Hz). As a methodological cross-check, we also extracted Praat’s autocorrelation-based pitch “voicing strength” (0–1), which is closely related to the height of the selected autocorrelation peak within the pitch-lag range. The notebook plots both tracks aligned to each V–t–(V/sonorant) token and supports time normalization over the entire segment (0–10 scale) and over an automatically detected “decline→recovery” window (autocorrelation dip), enabling aggregation across tokens.

## Methodological note: autocorrelation, lags, and replication target

This notebook was developed with the explicit goal of reproducing, as closely as possible, the logic of the short-time autocorrelation analysis used by Ashby & Przedlacka (2014) to study glottal stops and glottalized alveolar plosives. In their work, glottalization is characterized acoustically by **temporary dips in autocorrelation**, reflecting a loss of regular periodic voicing rather than simple silence.

### Autocorrelation and lags

Autocorrelation measures the similarity between a signal and a time-shifted version of itself. Formally, for a signal \(x(t)\), the autocorrelation at a given *lag* \(\tau\) is defined as the correlation between \(x(t)\) and \(x(t + \tau)\).

A **lag** therefore corresponds to the amount of temporal shift applied to the signal:
- In digital signals, lags are implemented as shifts in samples.
- In speech analysis, lags are typically interpreted in time units (milliseconds) and are directly related to the **pitch period** of voiced speech.
- For example, an F0 of 100 Hz corresponds to a pitch period of 10 ms, i.e. a lag of 10 ms.

In normalized autocorrelation, the value at lag 0 is 1 by definition, and all other autocorrelation values fall within the range \([-1, 1]\). The height of the highest positive autocorrelation peak at non-zero lag provides a measure of how strongly periodic the signal is.

### Peak autocorrelation and methodological alignment

Following Ashby & Przedlacka (2014), this notebook computes a **short-time peak autocorrelation track** by:
- analysing overlapping short windows (e.g. 30 ms) at a fixed frame step (e.g. 5 ms),
- computing the normalized autocorrelation function for each window,
- and extracting the **maximum positive autocorrelation peak within a restricted lag range** corresponding to plausible pitch periods (defined by an F0 floor and ceiling).

Restricting the lag search to pitch-relevant values is crucial. Allowing arbitrarily small lags (e.g. one-sample shifts) leads to artificially high autocorrelation peaks even in irregular or glottalized phonation, thereby obscuring the dips that are central to Ashby & Przedlacka’s analysis.

As a methodological cross-check, the notebook also extracts Praat’s autocorrelation-based pitch *voicing strength* (0–1). This measure corresponds closely to the height of the selected autocorrelation peak at the estimated pitch period and provides an independent validation of the custom peak autocorrelation tracking.

### Time normalization and glottalization windows

In line with the interpretive framework of Ashby & Przedlacka (2014), the notebook supports time normalization across entire V–t–(V/sonorant) intervals, as well as over automatically detected “decline→recovery” windows corresponding to autocorrelation dips. This allows aggregation and comparison across tokens of differing absolute durations while preserving the temporal structure of glottalization.

---

[^1]: Ashby, M., & Przedlacka, J. (2014). *Measuring incompleteness: Acoustic correlates of glottal articulations*. In M. J. Jones & R. J. Knight (Eds.), **The Bloomsbury Companion to Phonetics** (pp. 285–302). London: Bloomsbury.
