# omni-ffi

**Zero-cost C++/CUDA FFI bridge for the OmniPulse Wavelet Scattering Transform engine**

[![Crates.io](https://img.shields.io/crates/v/omni-ffi.svg)](https://crates.io/crates/omni-ffi)
[![docs.rs](https://img.shields.io/docsrs/omni-ffi)](https://docs.rs/omni-ffi)
[![License](https://img.shields.io/crates/l/omni-ffi.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-76B900)](#cuda-gpu-build)

`omni-ffi` provides a type-safe, zero-overhead Rust FFI bridge into the [OmniPulse WST math engine](https://github.com/samvardhan03/Module-I-omni-wst-core) — a production-grade C++/CUDA library for computing the **Wavelet Scattering Transform (WST)** and **Joint Time-Frequency Scattering (JTFS)** on high-frequency time-series data.

Built on top of [`cxx`](https://cxx.rs), this crate compiles the real C++ math engine (Radix-2 Cooley-Tukey FFT + analytic Morlet filter bank + depth-*m* scattering cascade) directly into your Rust binary — **no mocks, no stubs, no runtime binding resolution**.

---

## Why This Exists

The WST/JTFS pipeline is a mathematically rigorous alternative to deep-learning feature extractors. It produces **deformation-stable**, **Lipschitz-continuous** fingerprints that are formally bounded against adversarial perturbations — properties that neural networks cannot guarantee.

`omni-ffi` lets Rust applications call this pipeline with:

- **Zero-copy memory transfer** — raw `f32` buffers pass through the FFI without serialization or heap duplication.
- **Compile-time dispatch** — the `cuda` feature flag selects the GPU code path at build time; no runtime branching.
- **Deterministic ownership** — `WSTResult` makes memory ownership explicit: the C++ side allocates, the Rust side frees via `free_wst_result` / `free_fingerprint`.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Build Modes](#build-modes)
  - [CPU Build (Default)](#cpu-build-default)
  - [CUDA GPU Build](#cuda-gpu-build)
- [API Overview](#api-overview)
  - [`run_wst_pipeline`](#run_wst_pipeline)
  - [`execute_fingerprint_pass`](#execute_fingerprint_pass)
  - [`free_fingerprint` / `free_wst_result`](#free_fingerprint--free_wst_result)
  - [`WSTResult`](#wstresult)
- [Architecture](#architecture)
- [Mathematical Background](#mathematical-background)
- [Use Cases](#use-cases)
- [Safety Contract](#safety-contract)
- [Environment Variables](#environment-variables)
- [License](#license)

---

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
omni-ffi = "0.1"
```

Then in your Rust code:

```rust
use omni_ffi::{WSTResult, execute_fingerprint_pass, free_fingerprint};

fn process_signal(signal: &[f32]) -> Vec<f32> {
    let ptr = signal.as_ptr() as u64;
    let len = signal.len() as i32;

    // J=8 (max wavelet scale 2^8), Q=16 (wavelets per octave)
    let result: WSTResult = unsafe {
        execute_fingerprint_pass(ptr, len, 8, 16)
            .expect("WST pipeline failed")
    };

    // Copy coefficients into a safe Vec before releasing the C++ allocation
    let coeffs = unsafe {
        std::slice::from_raw_parts(
            result.fingerprint_ptr as *const f32,
            result.coeff_count as usize,
        )
        .to_vec()
    };

    // Release the C++ heap allocation
    unsafe { free_fingerprint(result) };

    coeffs
}
```

---

## Build Modes

### CPU Build (Default)

```bash
cargo build
```

The default build compiles `wst_bridge_cpu.cpp`, which links the full C++ math engine:

- **Radix-2 Cooley-Tukey FFT** for spectral decomposition
- **Analytic Morlet filter bank** construction via `build_cpu_morlet_bank()`
- **Depth-*m* scattering cascade** with modulus nonlinearity

No CUDA toolkit required. Runs anywhere Rust + a C++17 compiler are available.

> **Note:** The CPU build silently ignores the `use_jtfs` flag — it always computes a plain WST pass. For JTFS phase recovery, use the CUDA build.

### CUDA GPU Build

```bash
cargo build --features cuda
```

When the `cuda` feature is enabled, the build system:

1. Compiles `wst_bridge_cuda.cpp` (with `-DOMNI_FFI_HAS_CUDA`)
2. Links `cudart` and `cufft`
3. Dispatches to the templated `WSTEngine<HopperTag, J, Q>` kernel

**Requirements:**

| Dependency            | Version          |
|-----------------------|------------------|
| NVIDIA CUDA Toolkit   | 11.x or 12.x    |
| C++17 Compiler        | GCC / Clang      |
| GPU Architecture      | Ampere+ (sm_80)  |

Set `CUDA_LIB_DIR` if your CUDA libraries are in a non-standard location:

```bash
CUDA_LIB_DIR=/usr/local/cuda/lib64 cargo build --features cuda
```

---

## API Overview

### `run_wst_pipeline`

The low-level FFI function. Full control over all parameters.

```rust
unsafe fn run_wst_pipeline(
    input_plasma_ptr: u64,  // pointer to contiguous f32 buffer
    signal_len: i32,        // samples per signal (> 0)
    batch_size: i32,        // number of signals in the batch (> 0)
    J: i32,                 // maximum wavelet scale 2^J (> 0)
    Q: i32,                 // wavelets per octave (> 0)
    depth: i32,             // scattering cascade depth (> 0)
    use_jtfs: bool,         // enable JTFS phase-recovery (CUDA only)
) -> Result<WSTResult>
```

### `execute_fingerprint_pass`

Convenience wrapper: single-batch, depth-2, plain-WST.

```rust
pub unsafe fn execute_fingerprint_pass(
    plasma_id: u64,   // pointer to f32[signal_len]
    signal_len: i32,
    j: i32,
    q: i32,
) -> Result<WSTResult, cxx::Exception>
```

### `free_fingerprint` / `free_wst_result`

Release the C++ tensor allocation. **Must be called exactly once** per successful pipeline invocation.

```rust
pub unsafe fn free_fingerprint(result: WSTResult);
```

### `WSTResult`

```rust
#[derive(Clone, Copy, Debug)]
pub struct WSTResult {
    pub fingerprint_ptr: u64,  // opaque pointer to output tensor
    pub coeff_count: u64,      // number of f32 coefficients
    pub exec_time_us: u64,     // wall-clock execution time (µs)
}
```

- **CPU build:** `fingerprint_ptr` is a `float*` from `new float[]` — released via `delete[]`.
- **CUDA build:** `fingerprint_ptr` is a `CUdeviceptr` from `cudaMalloc` — released via `cudaFree`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Rust Application                         │
│                                                                 │
│   execute_fingerprint_pass()  ──►  ffi::run_wst_pipeline()     │
│   free_fingerprint()          ──►  ffi::free_wst_result()      │
└────────────────────────┬────────────────────────────────────────┘
                         │  cxx bridge (zero-cost)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     wst_bridge.h (C++ FFI)                      │
├─────────────────────────┬───────────────────────────────────────┤
│  CPU (default)          │  CUDA (--features cuda)               │
│  wst_bridge_cpu.cpp     │  wst_bridge_cuda.cpp                  │
│  ┌───────────────────┐  │  ┌──────────────────────────────────┐ │
│  │ cpu_wst_engine.h  │  │  │ WSTEngine<HopperTag, J, Q>      │ │
│  │ • Radix-2 FFT     │  │  │ • cuFFT spectral decomposition  │ │
│  │ • Morlet bank     │  │  │ • Template-specialized tiles     │ │
│  │ • Scattering      │  │  │ • Double-buffered pinned memory  │ │
│  │   cascade         │  │  │ • Dual-stream JTFS dispatch      │ │
│  └───────────────────┘  │  └──────────────────────────────────┘ │
└─────────────────────────┴───────────────────────────────────────┘
```

### Memory Ownership Model

1. **Allocation:** C++ allocates the output tensor (`new float[]` on CPU, `cudaMalloc` on GPU).
2. **Transfer:** The pointer is returned to Rust as `u64` inside `WSTResult`.
3. **Deallocation:** Rust calls `free_wst_result` / `free_fingerprint`, which invokes `delete[]` (CPU) or `cudaFree` (GPU).
4. **Sentinel:** A `fingerprint_ptr` of `0` is a no-op free — safe to call unconditionally.

---

## Mathematical Background

The Wavelet Scattering Transform constructs deformation-stable signal representations through a cascade of wavelet convolutions and modulus nonlinearities.

### Scattering Cascade

Given input signal *x(u)*, a Morlet wavelet filter bank *ψ_λ*, and low-pass scaling function *φ_J*:

- **Zero-order:** `S[0]x = x ∗ φ_J`
- **First-order:** `S[1]x(u, λ₁) = |x ∗ ψ_λ₁| ∗ φ_J`
- **Depth-*m*:** `S[m]x = || ... |x ∗ ψ_λ₁| ∗ ... | ∗ ψ_λₘ| ∗ φ_J`

### Key Properties

| Property | Guarantee |
|----------|-----------|
| **Parseval energy conservation** | `Σ ‖S[p]x‖² = ‖x‖²` — no information loss across depth |
| **Lipschitz continuity** | `‖S[p]x − S[p]y‖ ≤ (‖ψ‖₁)^m · ‖x − y‖` — bounded adversarial sensitivity |
| **Translation invariance** | Controlled by scale `2^J` of the low-pass filter |

### Joint Time-Frequency Scattering (JTFS)

JTFS recovers phase-coupling information discarded by the modulus operator, applying separable 2D convolution across both time and log-frequency:

```
Ψ_μ,l,s(t, λ) = ψ_μ(t) · ψ_l,s(λ)
```

On the CUDA build, this is computed via dual parallel streams for maximum GPU utilization.

---

## Use Cases

| Domain | Application |
|--------|-------------|
| **Audio forensics** | Perceptual fingerprinting robust to adversarial phase-shifting attacks |
| **Astrophysics** | Gravitational wave chirp detection in broadband noise |
| **Neuroscience** | Deformation-stable EEG/ECG feature extraction |
| **Genomics** | Translation-invariant ChIP-seq peak calling |
| **MLOps** | Deterministic feature extraction for reproducible ML pipelines |

---

## Safety Contract

All FFI functions are `unsafe`. The caller **must** guarantee:

1. **Valid pointer:** `input_plasma_ptr` points to a contiguous, host-readable `f32` array of exactly `signal_len × batch_size` elements.
2. **Liveness:** The backing memory must remain live for the entire duration of the call.
3. **Positive parameters:** `signal_len`, `batch_size`, `J`, `Q`, and `depth` must all be `> 0`.
4. **Single free:** Each `WSTResult` must be freed exactly once via `free_wst_result` / `free_fingerprint`. Double-free is undefined behavior.
5. **CUDA registration (GPU build):** The input pointer must be registered via `cudaHostRegister` for UVA access.

Violations of invariants 1–3 will cause `cxx::Exception` (C++ throws `std::runtime_error`). Violations of 4–5 are undefined behavior.

---

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OMNI_WST_CORE_CPP` | Path to Module-I-omni-wst-core C++ sources | `../Module-I-omni-wst-core/cpp` |
| `CUDA_LIB_DIR` | Path to CUDA shared libraries | System default |
| `CARGO_FEATURE_CUDA` | Set automatically by `--features cuda` | — |

---

## Building from Source

```bash
git clone https://github.com/samvardhan03/Module-1.1-omni-ffi
cd Module-1.1-omni-ffi

# Ensure the WST engine sources are available
git clone https://github.com/samvardhan03/Module-I-omni-wst-core ../Module-I-omni-wst-core

# CPU build
cargo build --release

# GPU build (Linux with CUDA toolkit)
cargo build --release --features cuda

# Run tests
cargo test

# Generate documentation
cargo doc --open
```

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

```
Copyright 2025 Samvardhan Singh
```

---

*Part of the [OmniPulse](https://github.com/samvardhan03) signal intelligence platform.*
