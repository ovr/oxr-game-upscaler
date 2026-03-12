use hassle_rs::*;
use std::path::Path;

fn main() {
    let shader_dir = Path::new("src/upscalers/shaders");
    let out_dir = std::env::var("OUT_DIR").unwrap();

    let shaders: &[(&str, &str, &str)] = &[
        ("blit_vs.hlsl", "VS", "vs_6_0"),
        ("blit_ps.hlsl", "PS", "ps_6_0"),
        ("lanczos_ps.hlsl", "PS", "ps_6_0"),
        ("debug_ps.hlsl", "PS", "ps_6_0"),
        ("SGSRv1/sgsr_ps.hlsl", "PS", "ps_6_0"),
        ("rcas_ps.hlsl", "PS", "ps_6_0"),
        ("SGSRv2/2PassFS/sgsr2_vs.hlsl", "VS", "vs_6_0"),
        ("SGSRv2/2PassFS/sgsr2_convert_ps.hlsl", "PS", "ps_6_0"),
        ("SGSRv2/2PassFS/sgsr2_upscale_ps.hlsl", "PS", "ps_6_0"),
        ("SGSRv2/3Pass/sgsr2_3p_convert_ps.hlsl", "PS", "ps_6_0"),
        ("SGSRv2/3Pass/sgsr2_3p_activate_ps.hlsl", "PS", "ps_6_0"),
        ("SGSRv2/3Pass/sgsr2_3p_upscale_ps.hlsl", "PS", "ps_6_0"),
        ("imgui_vs.hlsl", "VS", "vs_6_0"),
        ("imgui_ps.hlsl", "PS", "ps_6_0"),
    ];

    let dxc = Dxc::new(None).expect("Failed to load DXC library");
    let compiler = dxc
        .create_compiler()
        .expect("Failed to create DXC compiler");
    let library = dxc
        .create_library()
        .expect("Failed to create DXC library interface");

    for &(src_path, entry, profile) in shaders {
        let full_path = shader_dir.join(src_path);
        println!("cargo:rerun-if-changed={}", full_path.display());

        let source = std::fs::read_to_string(&full_path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", full_path.display(), e));

        let blob = library
            .create_blob_with_encoding_from_str(&source)
            .unwrap_or_else(|e| panic!("Failed to create blob for {}: {}", src_path, e));

        let result = compiler.compile(&blob, src_path, entry, profile, &["-O3"], None, &[]);

        let dxc_result = match result {
            Ok(r) => r,
            Err((dxc_result, _hr)) => {
                let err_str = dxc_result
                    .get_error_buffer()
                    .ok()
                    .and_then(|eb| library.get_blob_as_string(&eb.into()).ok())
                    .unwrap_or_else(|| "unknown error".to_string());
                panic!("DXC compilation failed for {}:\n{}", src_path, err_str);
            }
        };

        let compiled = dxc_result
            .get_result()
            .unwrap_or_else(|e| panic!("Failed to get compiled result for {}: {}", src_path, e));
        let bytecode: Vec<u8> = compiled.to_vec();

        let out_name = src_path.replace('/', "_").replace(".hlsl", ".dxil");
        let out_path = Path::new(&out_dir).join(&out_name);
        std::fs::write(&out_path, &bytecode)
            .unwrap_or_else(|e| panic!("Failed to write {}: {}", out_path.display(), e));
    }
}
