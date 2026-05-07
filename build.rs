use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let cuda_enabled = env::var_os("CARGO_FEATURE_CUDA").is_some();

    let mut build = cxx_build::bridge("src/lib.rs");
    build
        .include(crate_dir.join("cpp"))
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-Wno-unused-function")
        .flag_if_supported("-Wno-unused-parameter");

    if cuda_enabled {
        // GPU build path. Compiles the CUDA-aware bridge translation unit
        // and links the CUDA runtime + cuFFT. The actual nvcc-compiled
        // kernel objects (wst_kernel.cuh / jtfs_kernel.cuh / wst_bridge.cu)
        // are expected to be pre-built into a static lib that the consuming
        // binary links via `cargo:rustc-link-search`.
        //
        // The upstream Module-I-omni-wst-core C++ engine sources are needed
        // for CUDA kernel headers. Override the path at build time with:
        //   OMNI_WST_CORE_CPP=/abs/path cargo build --features cuda
        let engine_cpp = env::var("OMNI_WST_CORE_CPP")
            .map(PathBuf::from)
            .unwrap_or_else(|_| crate_dir.join("../Module-I-omni-wst-core/cpp"));

        if !engine_cpp.exists() {
            panic!(
                "\n\nomni-ffi CUDA build error: upstream engine sources not found at {:?}.\n\
                 Set OMNI_WST_CORE_CPP=/path/to/Module-I-omni-wst-core/cpp or clone the repo \
                 next to this crate.\n\n",
                engine_cpp
            );
        }

        build
            .include(&engine_cpp)
            .file("cpp/wst_bridge_cuda.cpp")
            .define("OMNI_FFI_HAS_CUDA", None);

        if let Ok(cuda_lib_dir) = env::var("CUDA_LIB_DIR") {
            println!("cargo:rustc-link-search=native={}", cuda_lib_dir);
        }
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cufft");
        println!("cargo:rerun-if-changed=cpp/wst_bridge_cuda.cpp");
        println!("cargo:rerun-if-env-changed=CUDA_LIB_DIR");
        println!("cargo:rerun-if-env-changed=OMNI_WST_CORE_CPP");
    } else {
        // CPU-only path. The Radix-2 FFT + Morlet cascade lives entirely
        // in cpp/cpu_wst_engine.h — no external dependencies required.
        build.file("cpp/wst_bridge_cpu.cpp");
        println!("cargo:rerun-if-changed=cpp/wst_bridge_cpu.cpp");
        println!("cargo:rerun-if-changed=cpp/cpu_wst_engine.h");
    }

    build.compile("omni_wst_bridge");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cpp/wst_bridge.h");
}
