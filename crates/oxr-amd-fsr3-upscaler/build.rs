use hassle_rs::*;
use std::path::Path;

fn main() {
    let shader_dir = Path::new("src/upscalers/shaders");
    // AA shaders come from the imba-test submodule
    let aa_shader_dir = Path::new("../../vendor/imba-test/shaders/imba/aa");
    let out_dir = std::env::var("OUT_DIR").unwrap();

    // (path relative to shader_dir, entry, profile)
    let shaders: &[(&str, &str, &str)] = &[
        ("blit_vs.hlsl", "VS", "vs_6_2"),
        ("blit_ps.hlsl", "PS", "ps_6_2"),
        ("lanczos_ps.hlsl", "PS", "ps_6_2"),
        ("debug_ps.hlsl", "PS", "ps_6_2"),
        ("SGSRv1/sgsr_ps.hlsl", "PS", "ps_6_2"),
        ("rcas_ps.hlsl", "PS", "ps_6_2"),
        ("SGSRv2/2PassFS/sgsr2_vs.hlsl", "VS", "vs_6_2"),
        ("SGSRv2/2PassFS/sgsr2_convert_ps.hlsl", "PS", "ps_6_2"),
        ("SGSRv2/2PassFS/sgsr2_upscale_ps.hlsl", "PS", "ps_6_2"),
        ("SGSRv2/3Pass/sgsr2_3p_convert_ps.hlsl", "PS", "ps_6_2"),
        ("SGSRv2/3Pass/sgsr2_3p_activate_ps.hlsl", "PS", "ps_6_2"),
        ("SGSRv2/3Pass/sgsr2_3p_upscale_ps.hlsl", "PS", "ps_6_2"),
        ("imgui_vs.hlsl", "VS", "vs_6_2"),
        ("imgui_ps.hlsl", "PS", "ps_6_2"),
    ];

    // AA compute shaders (filename, entry, profile) — read from aa_shader_dir
    let aa_shaders: &[(&str, &str, &str)] = &[
        ("0_PixelUnshuffleCS.hlsl", "main", "cs_6_2"),
        ("1_ConvCS.hlsl", "main", "cs_6_2"),
        ("1a_Conv3x3_16x16CS.hlsl", "main", "cs_6_2"),
        ("1b_Conv3x3_32x32CS.hlsl", "main", "cs_6_2"),
        ("1c_Conv3x3_32x16CS.hlsl", "main", "cs_6_2"),
        ("1d_Conv3x3_16x12CS.hlsl", "main", "cs_6_2"),
        ("1e_Conv3x3_12x12CS.hlsl", "main", "cs_6_2"),
        ("1f_Conv3x3S2_16x32CS.hlsl", "main", "cs_6_2"),
        ("2_GNStatsCS.hlsl", "main", "cs_6_2"),
        ("2b_GNStatsReduceCS.hlsl", "main", "cs_6_2"),
        ("3_GNApplyCS.hlsl", "main", "cs_6_2"),
        ("4_BackwardWarpCS.hlsl", "main", "cs_6_2"),
        ("5_AttentionCS.hlsl", "main", "cs_6_2"),
        ("6_NearestUpsampleCS.hlsl", "main", "cs_6_2"),
        ("7_SkipConcatConvCS.hlsl", "main", "cs_6_2"),
        ("8_PixelShuffleOutCS.hlsl", "main", "cs_6_2"),
        ("9_ScaleMVCS.hlsl", "main", "cs_6_2"),
    ];

    // Track AACommon.hlsli include
    let aa_common_path = aa_shader_dir.join("AACommon.hlsli");
    println!("cargo:rerun-if-changed={}", aa_common_path.display());

    let aa_common_src = std::fs::read_to_string(&aa_common_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", aa_common_path.display(), e));

    let dxc = Dxc::new(None).expect("Failed to load DXC library");
    let compiler = dxc
        .create_compiler()
        .expect("Failed to create DXC compiler");
    let library = dxc
        .create_library()
        .expect("Failed to create DXC library interface");

    // Compile regular shaders
    for &(src_path, entry, profile) in shaders {
        let full_path = shader_dir.join(src_path);
        compile_shader(
            &compiler, &library, &full_path, src_path, entry, profile, &out_dir, None,
        );
    }

    // Compile AA shaders (with AACommon.hlsli inlining)
    for &(filename, entry, profile) in aa_shaders {
        let full_path = aa_shader_dir.join(filename);
        let out_name = format!("aa_{}", filename);
        compile_shader(
            &compiler,
            &library,
            &full_path,
            &out_name,
            entry,
            profile,
            &out_dir,
            Some(&aa_common_src),
        );
    }
}

fn compile_shader(
    compiler: &DxcCompiler,
    library: &DxcLibrary,
    full_path: &Path,
    src_name: &str,
    entry: &str,
    profile: &str,
    out_dir: &str,
    inline_include: Option<&str>,
) {
    println!("cargo:rerun-if-changed={}", full_path.display());

    let mut source = std::fs::read_to_string(full_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", full_path.display(), e));

    // Inline #include "AACommon.hlsli" if provided
    if let Some(include_src) = inline_include {
        if source.contains("#include \"AACommon.hlsli\"") {
            source = source.replace("#include \"AACommon.hlsli\"", include_src);
        }
    }

    let blob = library
        .create_blob_with_encoding_from_str(&source)
        .unwrap_or_else(|e| panic!("Failed to create blob for {}: {}", src_name, e));

    let args = ["-O3", "-enable-16bit-types"];

    let result = compiler.compile(&blob, src_name, entry, profile, &args, None, &[]);

    let dxc_result = match result {
        Ok(r) => r,
        Err((dxc_result, _hr)) => {
            let err_str = dxc_result
                .get_error_buffer()
                .ok()
                .and_then(|eb| library.get_blob_as_string(&eb.into()).ok())
                .unwrap_or_else(|| "unknown error".to_string());
            panic!("DXC compilation failed for {}:\n{}", src_name, err_str);
        }
    };

    let compiled = dxc_result
        .get_result()
        .unwrap_or_else(|e| panic!("Failed to get compiled result for {}: {}", src_name, e));
    let bytecode: Vec<u8> = compiled.to_vec();

    let out_name = src_name.replace('/', "_").replace(".hlsl", ".dxil");
    let out_path = Path::new(out_dir).join(&out_name);
    std::fs::write(&out_path, &bytecode)
        .unwrap_or_else(|e| panic!("Failed to write {}: {}", out_path.display(), e));
}
