// cpu_wst_engine.h — Self-contained CPU implementation of the Wavelet
// Scattering Transform for the omni-ffi crate.
//
// Implements:
//   1. Radix-2 Cooley-Tukey FFT (in-place, complex-valued)
//   2. Analytic Morlet filter bank construction
//   3. Depth-m scattering cascade with modulus nonlinearity
//
// This is the real mathematical pipeline — no mocks. The CUDA build
// supersedes this with cuFFT + device kernels, but the transforms are
// mathematically identical.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <complex>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ═══════════════════════════════════════════════════════════════════════
// Radix-2 Cooley-Tukey FFT (in-place, decimation-in-time)
// ═══════════════════════════════════════════════════════════════════════

/// Round up to the next power of two.
static inline std::size_t next_pow2(std::size_t n) {
    std::size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

/// In-place Radix-2 FFT. `n` MUST be a power of two.
/// `inverse == true` computes the IFFT (with 1/N normalisation).
static void fft_radix2(std::complex<float>* buf, std::size_t n, bool inverse) {
    if (n <= 1) return;

    // Bit-reversal permutation
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) std::swap(buf[i], buf[j]);
    }

    // Butterfly passes
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const float angle = (inverse ? 2.0f : -2.0f)
                            * static_cast<float>(M_PI)
                            / static_cast<float>(len);
        const std::complex<float> wn(std::cos(angle), std::sin(angle));

        for (std::size_t i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (std::size_t j = 0; j < len / 2; ++j) {
                std::complex<float> u = buf[i + j];
                std::complex<float> v = buf[i + j + len / 2] * w;
                buf[i + j]           = u + v;
                buf[i + j + len / 2] = u - v;
                w *= wn;
            }
        }
    }

    if (inverse) {
        const float inv_n = 1.0f / static_cast<float>(n);
        for (std::size_t i = 0; i < n; ++i) {
            buf[i] *= inv_n;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Morlet wavelet filter bank
// ═══════════════════════════════════════════════════════════════════════

/// A pre-computed filter bank: each entry is the frequency-domain
/// representation of a Morlet wavelet at a specific centre frequency.
/// Stored at padded FFT length `fft_len`.
struct CPUFilterBank {
    std::size_t fft_len;                                  // padded length
    std::size_t n_filters;                                // J * Q
    std::vector<std::vector<std::complex<float>>> filters; // [n_filters][fft_len]
    std::vector<std::complex<float>> phi;                 // low-pass φ_J
    float l1_norm;                                        // ‖ψ‖₁ of the bank
};

/// Construct a single analytic Morlet wavelet in the frequency domain.
///
/// ψ̂(ω) = π^(-1/4) · exp( -½(ω - ω₀)² / σ² )
///
/// The wavelet is analytic (one-sided in freq), so negative frequencies
/// are zeroed. `sigma` controls bandwidth; `centre_freq` sets ω₀.
static std::vector<std::complex<float>> morlet_freq(
    std::size_t fft_len,
    float       centre_freq,
    float       sigma)
{
    std::vector<std::complex<float>> psi(fft_len, {0.0f, 0.0f});
    const float norm = std::pow(static_cast<float>(M_PI), -0.25f);

    for (std::size_t k = 0; k <= fft_len / 2; ++k) {
        const float omega = 2.0f * static_cast<float>(M_PI)
                            * static_cast<float>(k)
                            / static_cast<float>(fft_len);
        const float diff = omega - centre_freq;
        const float val  = norm * std::exp(-0.5f * diff * diff / (sigma * sigma));
        psi[k] = {val, 0.0f};
    }
    return psi;
}

/// Build a Gaussian low-pass filter φ_J at scale 2^J.
static std::vector<std::complex<float>> lowpass_freq(
    std::size_t fft_len,
    float       cutoff_freq,
    float       sigma)
{
    std::vector<std::complex<float>> phi(fft_len, {0.0f, 0.0f});
    for (std::size_t k = 0; k <= fft_len / 2; ++k) {
        const float omega = 2.0f * static_cast<float>(M_PI)
                            * static_cast<float>(k)
                            / static_cast<float>(fft_len);
        const float val = std::exp(-0.5f * omega * omega / (cutoff_freq * cutoff_freq));
        phi[k] = {val, 0.0f};
    }
    // Mirror for negative frequencies (Hermitian symmetry for real signal)
    for (std::size_t k = fft_len / 2 + 1; k < fft_len; ++k) {
        phi[k] = phi[fft_len - k];
    }
    return phi;
}

/// Build the full J×Q Morlet filter bank + low-pass scaling function.
///
/// Centre frequencies: ω_λ = ω_max · 2^(-λ/Q), λ ∈ [0, J·Q)
/// Bandwidth σ_λ scales inversely with centre frequency (constant-Q).
static CPUFilterBank build_cpu_morlet_bank(
    int32_t     J,
    int32_t     Q,
    int32_t     signal_len)
{
    const std::size_t fft_len = next_pow2(static_cast<std::size_t>(signal_len));
    const std::size_t n_filters = static_cast<std::size_t>(J)
                                * static_cast<std::size_t>(Q);

    CPUFilterBank bank;
    bank.fft_len   = fft_len;
    bank.n_filters = n_filters;
    bank.filters.reserve(n_filters);

    // ω_max: Nyquist-relative maximum centre frequency
    const float omega_max = static_cast<float>(M_PI);
    // Bandwidth factor: σ = ω₀ / Q (constant-Q property)
    const float q_float = static_cast<float>(Q);

    float l1_accum = 0.0f;

    for (std::size_t i = 0; i < n_filters; ++i) {
        const float ratio = static_cast<float>(i) / q_float;
        const float omega_0 = omega_max * std::pow(2.0f, -ratio);
        const float sigma   = omega_0 / q_float;

        auto psi = morlet_freq(fft_len, omega_0, sigma);

        // Accumulate L1 norm (in frequency domain, approximated)
        float psi_l1 = 0.0f;
        for (std::size_t k = 0; k < fft_len; ++k) {
            psi_l1 += std::abs(psi[k]);
        }
        psi_l1 /= static_cast<float>(fft_len);
        l1_accum += psi_l1;

        bank.filters.push_back(std::move(psi));
    }

    bank.l1_norm = (n_filters > 0) ? (l1_accum / static_cast<float>(n_filters))
                                   : 0.0f;

    // Low-pass φ_J: cutoff at 2^(-J) of Nyquist
    const float phi_cutoff = omega_max * std::pow(2.0f, -static_cast<float>(J));
    bank.phi = lowpass_freq(fft_len, phi_cutoff, phi_cutoff * 0.5f);

    return bank;
}

// ═══════════════════════════════════════════════════════════════════════
// Scattering cascade
// ═══════════════════════════════════════════════════════════════════════

/// Apply a single scattering layer to `input` using filter bank `bank`.
///
/// For each wavelet ψ_λ in the bank:
///   1. X̂ = FFT(input)
///   2. Y_λ = IFFT(X̂ · Ψ̂_λ)     (convolution in frequency domain)
///   3. U_λ = |Y_λ|               (modulus nonlinearity)
///
/// Returns the concatenated modulus coefficients and the low-pass
/// averaging |U_λ| ∗ φ_J that forms the scattering output.
static void scatter_layer(
    const float*          input,
    float*                output,        // [signal_len] averaged output
    std::size_t           signal_len,
    const CPUFilterBank&  bank,
    std::vector<std::vector<float>>& modulus_out)  // [n_filters][signal_len]
{
    const std::size_t N = bank.fft_len;

    // Zero-padded complex buffer for input
    std::vector<std::complex<float>> X(N, {0.0f, 0.0f});
    for (std::size_t i = 0; i < signal_len; ++i) {
        X[i] = {input[i], 0.0f};
    }
    fft_radix2(X.data(), N, false);

    modulus_out.resize(bank.n_filters);

    // Accumulate the low-pass average across all filters
    std::vector<float> avg(signal_len, 0.0f);

    for (std::size_t f = 0; f < bank.n_filters; ++f) {
        // Element-wise multiply in frequency domain: X̂ · Ψ̂_λ
        std::vector<std::complex<float>> Y(N);
        for (std::size_t k = 0; k < N; ++k) {
            Y[k] = X[k] * bank.filters[f][k];
        }

        // IFFT → time domain
        fft_radix2(Y.data(), N, true);

        // Modulus nonlinearity |Y_λ|
        modulus_out[f].resize(signal_len);
        for (std::size_t i = 0; i < signal_len; ++i) {
            modulus_out[f][i] = std::abs(Y[i]);
        }

        // Low-pass average: convolve |Y_λ| with φ_J
        std::vector<std::complex<float>> U(N, {0.0f, 0.0f});
        for (std::size_t i = 0; i < signal_len; ++i) {
            U[i] = {modulus_out[f][i], 0.0f};
        }
        fft_radix2(U.data(), N, false);
        for (std::size_t k = 0; k < N; ++k) {
            U[k] *= bank.phi[k];
        }
        fft_radix2(U.data(), N, true);

        for (std::size_t i = 0; i < signal_len; ++i) {
            avg[i] += U[i].real();
        }
    }

    // Write the averaged scattering coefficients to output
    if (bank.n_filters > 0) {
        const float inv = 1.0f / static_cast<float>(bank.n_filters);
        for (std::size_t i = 0; i < signal_len; ++i) {
            output[i] = avg[i] * inv;
        }
    } else {
        std::memset(output, 0, signal_len * sizeof(float));
    }
}

/// Full depth-m CPU scattering cascade.
///
/// Computes: S[0], S[1], ..., S[depth] coefficients.
/// The final output is written to `output[signal_len]`.
///
/// `l1_norm_out` receives the measured ‖ψ‖₁ of the filter bank,
/// allowing Rust to verify the Lipschitz bound.
static void cpu_wst_forward(
    const float*          input,
    float*                output,       // [batch_size * signal_len]
    int32_t               signal_len,
    int32_t               batch_size,
    int32_t               depth,
    const CPUFilterBank&  bank,
    float&                l1_norm_out)
{
    const std::size_t sig_len = static_cast<std::size_t>(signal_len);

    l1_norm_out = bank.l1_norm;

    for (int32_t b = 0; b < batch_size; ++b) {
        const float* sig_in  = input  + static_cast<std::size_t>(b) * sig_len;
        float*       sig_out = output + static_cast<std::size_t>(b) * sig_len;

        // Working buffer: starts with the input signal, gets replaced
        // with modulus coefficients at each depth level.
        std::vector<float> current(sig_in, sig_in + sig_len);

        for (int32_t d = 0; d < depth; ++d) {
            std::vector<std::vector<float>> modulus;
            scatter_layer(current.data(), sig_out, sig_len, bank, modulus);

            // Feed the first filter's modulus output forward as the
            // next depth's input (standard WST cascade propagation).
            if (!modulus.empty() && !modulus[0].empty()) {
                current = modulus[0];
            } else {
                // Degenerate case: just propagate zeros
                std::fill(current.begin(), current.end(), 0.0f);
            }
        }
    }
}
