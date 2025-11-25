# xddsp_improved_buzz

```text
MODULE NAME:
xddsp_improved_buzz

DESCRIPTION:
Bandlimited analytic Buzz oscillator using the classic sin(Nx)/(N sin x) formulation, with:
- phase-correct integration (wrap after increment, sample before wrap),
- stable zero-crossing handling via Taylor-series approximation,
- optional skew (phase warping),
- optional FM and PM.

INPUTS:
- x / fm_hz : per-sample frequency modulation in Hz (added to base frequency).
- base_freq_hz (in state) : base oscillator frequency in Hz.
- sr_hz (in params)       : sample rate.
- nharm (in params)       : nominal number of harmonics (timbre).
- skew (in params)        : harmonic skew (-1..1), implemented as phase warp.
- pm_rad (in params)      : static phase modulation in radians.

OUTPUTS:
- y : output Buzz oscillator sample.
- new_state : updated oscillator state tuple.

STATE VARIABLES:
(phase, base_freq_hz)
where:
- phase : current unwrapped phase in radians.
- base_freq_hz : base oscillator frequency in Hz.

EQUATIONS / MATH:

Let:
- radius = π  (phase spans (-π, π])
- span   = 2 * radius
- wrap(φ) = maps φ to (-radius, radius] using modular arithmetic.

Phase integration with FM:
    f_eff[n] = max(base_freq_hz + fm_hz[n], 0)
    inc[n]   = (2 * radius * f_eff[n]) / sr_hz

We treat `phase[n]` as an unwrapped accumulator. For audio output:
    φw[n] = wrap(phase[n])          # wrapped phase for shaping
    φ[n]  = φw[n] + pm_rad          # phase modulation

Buzz core:
Let x = φ[n] / 2.

The ideal analytic Buzz core is:
    buzz(x, N) = sin(N * x) / (N * sin(x))

We implement a numerically-stable version:

For small |x|:
    buzz(x, N) ≈ 1 - ((N^2 - 1) / 6) * x^2       (2nd-order Taylor)

Otherwise:
    buzz(x, N) = sin(N * x) / (N * sin(x))

Phase skew (simple phase warp, not spectrum-analytic):
    skew_clamped = clamp(skew, -1, 1)
    φ_skew = φ * (1 + 0.5 * skew_clamped)

Core sample:
    x      = φ_skew / 2
    y_core = buzz(x, nharm)

Output:
    y[n] = y_core

State update:
    phase[n+1] = phase[n] + inc[n]
    base_freq_hz is unchanged at audio rate (updated via control path).

through-zero rules:
- f_eff[n] = max(base_freq_hz + fm_hz[n], 0)  # no negative freq; clamps at 0.

phase wrapping rules:
- wrap(φ) = ((φ + radius) mod (2 * radius)) - radius

nonlinearities:
- Buzz formula is a bounded nonlinear mapping from phase to amplitude.
- Optional skew introduces non-harmonic asymmetry via phase scaling.

interpolation rules:
- No explicit interpolation; FM is treated as per-sample modulation.

any time-varying coefficient rules:
- FM is time-varying (fm_hz[n]).
- Base frequency, nharm, skew, pm_rad can be changed between blocks via update_state().

NOTES:
- nharm should be positive; values >= 1 recommended.
- For anti-aliasing in practice, choose nharm <= sr_hz / (2 * base_freq_hz).
- skew in [-1, 1] is recommended; outside this range is clamped internally.
- Phase is kept unwrapped in state for numerical robustness; wrapping is applied only where needed for the waveform.
```

---

## FULL PYTHON MODULE: `xddsp_improved_buzz.py`

```python
"""
xddsp_improved_buzz
-------------------

Bandlimited analytic Buzz oscillator in XDDSP style.

This module implements a functional, Numba-JIT-friendly Buzz oscillator:

    y(φ) = sin(N * φ/2) / (N * sin(φ/2))

with:

- Phase-correct integration (unwrapped phase accumulator, wrapped only for shaping).
- Stable zero-crossing handling using a 2nd-order Taylor approximation near φ/2 = 0.
- Optional phase skew via a simple phase warp.
- Optional FM (frequency modulation) in Hz and PM (phase modulation) in radians.
- Pure functional state: state and params are tuples, no classes/dicts.

Public API (XDDSP style):

    xddsp_improved_buzz_init(...)
    xddsp_improved_buzz_update_state(...)
    xddsp_improved_buzz_tick(x, state, params)
    xddsp_improved_buzz_process(x, state, params)

Where:
- x is per-sample FM in Hz for tick/process.
- state = (phase, base_freq_hz)
- params = (sr_hz, radius, nharm, skew, pm_rad)

All DSP math is implemented in @njit(cache=True, fastmath=True) functions,
with no dynamic allocation inside jitted code (buffers are preallocated outside).
"""

import numpy as np
from numba import njit
import math


# ==========================
# Low-level helpers (JIT)
# ==========================

@njit(cache=True, fastmath=True)
def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


@njit(cache=True, fastmath=True)
def _wrap_phase(phase: float, radius: float) -> float:
    """
    Wrap phase into (-radius, radius].

    Equivalent to:
        span = 2 * radius
        return ((phase + radius) % span) - radius
    """
    span = 2.0 * radius
    p = phase + radius
    # float modulo is fine in Numba; this avoids Python-side branching on arrays.
    p = p % span
    return p - radius


@njit(cache=True, fastmath=True)
def _buzz_shape(phi: float, nharm: float, skew: float) -> float:
    """
    Analytic Buzz core with improved numerical stability near zero.

    phi   : phase in radians (wrapped or not; we only care about modulo 2π)
    nharm : number of harmonics (float > 0)
    skew  : phase skew in [-1, 1] (clamped), simple warp

    Returns:
        y = sin(nharm * phi/2) / (nharm * sin(phi/2))
    with a Taylor-series approximation near phi/2 ≈ 0 to avoid singularities.
    """
    # Clamp skew for safety; this is scalar so branching is cheap.
    s = _clamp(skew, -1.0, 1.0)
    phi = phi * (1.0 + 0.5 * s)

    x = 0.5 * phi  # x = phi/2

    # If nharm is <= 0, return silence.
    if nharm <= 0.0:
        return 0.0

    # Small-angle threshold
    eps = 1e-6
    ax = abs(x)

    if ax < eps:
        # Taylor-series approximation:
        # sin(Nx) ≈ N x - (N^3 x^3)/6
        # sin(x) ≈ x - x^3/6
        # => sin(Nx)/(N sin x) ≈ 1 - ((N^2 - 1)/6) * x^2
        n2 = nharm * nharm
        c = (n2 - 1.0) / 6.0
        return 1.0 - c * x * x
    else:
        num = math.sin(nharm * x)
        den = math.sin(x)
        # Guard against den very close to zero (should be rare due to branch).
        if abs(den) < 1e-12:
            return 0.0
        return num / (nharm * den)


@njit(cache=True, fastmath=True)
def _xddsp_improved_buzz_tick_core(
    fm_hz: float,
    state,
    params,
):
    """
    Core single-sample Buzz tick.

    Inputs:
        fm_hz : frequency modulation in Hz (added to base frequency).

    state  = (phase, base_freq_hz)
    params = (sr_hz, radius, nharm, skew, pm_rad)

    Returns:
        y, new_state
    """
    phase = state[0]
    base_freq_hz = state[1]

    sr_hz = params[0]
    radius = params[1]
    nharm = params[2]
    skew = params[3]
    pm_rad = params[4]

    # Effective frequency with through-zero handling (clamped at 0).
    freq = base_freq_hz + fm_hz
    if freq < 0.0:
        freq = 0.0

    # Phase increment
    inc = (2.0 * radius * freq) / sr_hz

    # Sample the waveform at the current wrapped phase
    phase_wrapped = _wrap_phase(phase, radius)
    phi = phase_wrapped + pm_rad

    y = _buzz_shape(phi, nharm, skew)

    # Advance unwrapped phase
    new_phase = phase + inc

    new_state = (new_phase, base_freq_hz)
    return y, new_state


@njit(cache=True, fastmath=True)
def _xddsp_improved_buzz_process_core(
    fm_hz_buf,
    state,
    params,
    y_out,
):
    """
    Core block-processing Buzz loop.

    All arrays are preallocated outside this function.
    """
    phase = state[0]
    base_freq_hz = state[1]

    sr_hz = params[0]
    radius = params[1]
    nharm = params[2]
    skew = params[3]
    pm_rad = params[4]

    n = fm_hz_buf.shape[0]

    for i in range(n):
        fm_hz = fm_hz_buf[i]

        freq = base_freq_hz + fm_hz
        if freq < 0.0:
            freq = 0.0

        inc = (2.0 * radius * freq) / sr_hz

        phase_wrapped = _wrap_phase(phase, radius)
        phi = phase_wrapped + pm_rad

        y = _buzz_shape(phi, nharm, skew)
        y_out[i] = y

        phase = phase + inc

    new_state = (phase, base_freq_hz)
    return new_state


# ==========================
# Public XDDSP-style API
# ==========================

def xddsp_improved_buzz_init(
    freq_hz: float,
    sr_hz: float,
    phase_unit: float = 0.0,
    nharm: float = 10.0,
    skew: float = 0.0,
    pm_rad: float = 0.0,
    radius: float = math.pi,
):
    """
    Initialize Buzz oscillator state and params.

    Parameters
    ----------
    freq_hz : float
        Base oscillator frequency in Hz.
    sr_hz : float
        Sample rate in Hz.
    phase_unit : float, optional
        Initial phase in [0..1), mapped to (-radius, radius].
    nharm : float, optional
        Number of harmonics (timbre control). Should be > 0.
    skew : float, optional
        Phase skew in [-1, 1] (clamped internally).
    pm_rad : float, optional
        Static phase modulation in radians.
    radius : float, optional
        Phase radius (default π, giving phase in (-π, π]).

    Returns
    -------
    state : tuple
        (phase, base_freq_hz)
    params : tuple
        (sr_hz, radius, nharm, skew, pm_rad)
    """
    r = float(radius)
    # Map 0..1 to (-radius, radius)
    p = (2.0 * r) * float(phase_unit) - r

    state = (p, float(freq_hz))
    params = (float(sr_hz), r, float(nharm), float(skew), float(pm_rad))
    return state, params


def xddsp_improved_buzz_update_state(
    freq_hz: float,
    nharm: float,
    skew: float,
    pm_rad: float,
    state,
    params,
):
    """
    Update state/params in a functional way.

    freq_hz updates base frequency.
    nharm, skew, pm_rad update timbre parameters.

    Returns
    -------
    new_state : tuple
    new_params : tuple
    """
    phase = state[0]

    sr_hz = params[0]
    radius = params[1]

    new_state = (phase, float(freq_hz))
    new_params = (sr_hz, radius, float(nharm), float(skew), float(pm_rad))
    return new_state, new_params


def xddsp_improved_buzz_tick(
    x,
    state,
    params,
):
    """
    Public single-sample tick, XDDSP style.

    x      : FM in Hz (scalar float)
    state  : (phase, base_freq_hz)
    params : (sr_hz, radius, nharm, skew, pm_rad)

    Returns
    -------
    y : float
    new_state : tuple
    """
    fm_hz = float(x)
    y, new_state = _xddsp_improved_buzz_tick_core(fm_hz, state, params)
    return y, new_state


def xddsp_improved_buzz_process(
    x,
    state,
    params,
):
    """
    Block-processing wrapper, XDDSP style.

    x      : 1D array-like of FM in Hz.
    state  : (phase, base_freq_hz)
    params : (sr_hz, radius, nharm, skew, pm_rad)

    Returns
    -------
    y : np.ndarray
        Output Buzz signal, same shape as x.
    new_state : tuple
        Updated state after processing the block.
    """
    fm_hz_buf = np.asarray(x, dtype=np.float64)
    y_out = np.empty_like(fm_hz_buf)
    new_state = _xddsp_improved_buzz_process_core(fm_hz_buf, state, params, y_out)
    return y_out, new_state


# ==========================
# Smoke test / demo
# ==========================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Optional: simple listening demo.
    try:
        import sounddevice as sd
        HAVE_SD = True
    except Exception:
        HAVE_SD = False

    # --- Parameters ---
    sr = 48000.0
    freq = 220.0
    duration = 1.0
    nharm = 20.0
    skew = 0.0
    pm_rad = 0.0

    n_samples = int(sr * duration)

    # --- Init oscillator ---
    state, params = xddsp_improved_buzz_init(
        freq_hz=freq,
        sr_hz=sr,
        phase_unit=0.0,
        nharm=nharm,
        skew=skew,
        pm_rad=pm_rad,
    )

    # --- Build FM buffer (no FM for smoke test) ---
    fm_buf = np.zeros(n_samples, dtype=np.float64)

    # --- Render audio ---
    y, state = xddsp_improved_buzz_process(fm_buf, state, params)

    # --- Plot waveform (first few cycles) ---
    t = np.arange(n_samples) / sr
    cycles_to_show = 5
    samples_to_show = int(cycles_to_show * sr / freq)

    plt.figure(figsize=(10, 4))
    plt.plot(t[:samples_to_show], y[:samples_to_show])
    plt.title("xddsp_improved_buzz: waveform (first %d cycles)" % cycles_to_show)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # --- Plot spectrum ---
    window = np.hanning(n_samples)
    yf = np.fft.rfft(y * window)
    freqs = np.fft.rfftfreq(n_samples, 1.0 / sr)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(yf), 1e-12))

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, mag_db)
    plt.title("xddsp_improved_buzz: magnitude spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xlim(0, sr * 0.5)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # --- Optional listening demo ---
    if HAVE_SD:
        print("Playing 1 second of xddsp_improved_buzz at %.1f Hz..." % freq)
        # Attenuate a bit to avoid clipping.
        sd.play(0.2 * y.astype(np.float32), int(sr))
        sd.wait()
    else:
        print("sounddevice not available; skipping audio playback.")
```

---

## Next things you could do on top of this module

1. **Add adaptive anti-aliasing**

   * Compute `nharm_eff = min(nharm, sr / (2 * freq))` inside the core based on current `freq`, to reduce aliasing at high base frequencies.

2. **Time-varying nharm / skew**

   * Extend `xddsp_improved_buzz_process` to accept optional `nharm_buf` and `skew_buf` arrays, with a second jitted core that handles those as per-sample scalars.

3. **Stereo / multi-osc bank helper**

   * Build a small wrapper that runs multiple Buzz instances in parallel for additive synthesis / unison.

4. **Envelope + Buzz macro-module**

   * Glue this with an XDDSP-style envelope generator and a filter to form a complete bandlimited “buzz pad” voice.

If you want, I can generate a companion `xddsp_improved_buzz_aa` module with adaptive anti-aliasing and time-varying nharm/skew next.
