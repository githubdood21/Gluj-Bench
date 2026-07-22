pub const FP32: &str = r#"
struct Params { loop_count: u32, seed: u32, invocation_count: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= params.invocation_count) { return; }
    let base = f32((index ^ params.seed) & 1023u) * 0.0001 + 0.25;
    var a0 = vec4<f32>(base + 0.01, base + 0.02, base + 0.03, base + 0.04);
    var a1 = a0 + vec4<f32>(0.05);
    var a2 = a0 + vec4<f32>(0.10);
    var a3 = a0 + vec4<f32>(0.15);
    var a4 = a0 + vec4<f32>(0.20);
    var a5 = a0 + vec4<f32>(0.25);
    var a6 = a0 + vec4<f32>(0.30);
    var a7 = a0 + vec4<f32>(0.35);
    for (var i = 0u; i < params.loop_count; i++) {
        a0 = fma(a0, vec4<f32>(1.000001), vec4<f32>(0.000001));
        a1 = fma(a1, vec4<f32>(0.999999), vec4<f32>(0.000002));
        a2 = fma(a2, vec4<f32>(1.000002), vec4<f32>(0.000003));
        a3 = fma(a3, vec4<f32>(0.999998), vec4<f32>(0.000004));
        a4 = fma(a4, vec4<f32>(1.000003), vec4<f32>(0.000005));
        a5 = fma(a5, vec4<f32>(0.999997), vec4<f32>(0.000006));
        a6 = fma(a6, vec4<f32>(1.000004), vec4<f32>(0.000007));
        a7 = fma(a7, vec4<f32>(0.999996), vec4<f32>(0.000008));
    }
    output[index] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
"#;

pub const FP16: &str = r#"
enable f16;
struct Params { loop_count: u32, seed: u32, invocation_count: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read_write> output: array<vec4<f16>>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= params.invocation_count) { return; }
    let base = f16((index ^ params.seed) & 255u) * 0.001h + 0.25h;
    var a0 = vec4<f16>(base + 0.01h, base + 0.02h, base + 0.03h, base + 0.04h);
    var a1 = a0 + vec4<f16>(0.05h);
    var a2 = a0 + vec4<f16>(0.10h);
    var a3 = a0 + vec4<f16>(0.15h);
    var a4 = a0 + vec4<f16>(0.20h);
    var a5 = a0 + vec4<f16>(0.25h);
    var a6 = a0 + vec4<f16>(0.30h);
    var a7 = a0 + vec4<f16>(0.35h);
    for (var i = 0u; i < params.loop_count; i++) {
        a0 = fma(a0, vec4<f16>(1.0009765625h), vec4<f16>(0.0009765625h));
        a1 = fma(a1, vec4<f16>(0.99951171875h), vec4<f16>(0.001953125h));
        a2 = fma(a2, vec4<f16>(1.0009765625h), vec4<f16>(0.0029296875h));
        a3 = fma(a3, vec4<f16>(0.99951171875h), vec4<f16>(0.00390625h));
        a4 = fma(a4, vec4<f16>(1.0009765625h), vec4<f16>(0.0048828125h));
        a5 = fma(a5, vec4<f16>(0.99951171875h), vec4<f16>(0.005859375h));
        a6 = fma(a6, vec4<f16>(1.0009765625h), vec4<f16>(0.0068359375h));
        a7 = fma(a7, vec4<f16>(0.99951171875h), vec4<f16>(0.0078125h));
    }
    output[index] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
"#;

pub const FP64: &str = r#"
struct Params { loop_count: u32, seed: u32, invocation_count: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read_write> output: array<vec4<f64>>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= params.invocation_count) { return; }
    let base = f64((index ^ params.seed) & 1023u) * f64(0.0001) + f64(0.25);
    var a0 = vec4<f64>(base + f64(0.01), base + f64(0.02), base + f64(0.03), base + f64(0.04));
    var a1 = a0 + vec4<f64>(f64(0.05));
    var a2 = a0 + vec4<f64>(f64(0.10));
    var a3 = a0 + vec4<f64>(f64(0.15));
    var a4 = a0 + vec4<f64>(f64(0.20));
    var a5 = a0 + vec4<f64>(f64(0.25));
    var a6 = a0 + vec4<f64>(f64(0.30));
    var a7 = a0 + vec4<f64>(f64(0.35));
    for (var i = 0u; i < params.loop_count; i++) {
        a0 = fma(a0, vec4<f64>(f64(1.000000001)), vec4<f64>(f64(0.000000001)));
        a1 = fma(a1, vec4<f64>(f64(0.999999999)), vec4<f64>(f64(0.000000002)));
        a2 = fma(a2, vec4<f64>(f64(1.000000002)), vec4<f64>(f64(0.000000003)));
        a3 = fma(a3, vec4<f64>(f64(0.999999998)), vec4<f64>(f64(0.000000004)));
        a4 = fma(a4, vec4<f64>(f64(1.000000003)), vec4<f64>(f64(0.000000005)));
        a5 = fma(a5, vec4<f64>(f64(0.999999997)), vec4<f64>(f64(0.000000006)));
        a6 = fma(a6, vec4<f64>(f64(1.000000004)), vec4<f64>(f64(0.000000007)));
        a7 = fma(a7, vec4<f64>(f64(0.999999996)), vec4<f64>(f64(0.000000008)));
    }
    output[index] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
"#;

pub const INT32: &str = r#"
struct Params { loop_count: u32, seed: u32, invocation_count: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read_write> output: array<vec4<u32>>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= params.invocation_count) { return; }
    let base = (index ^ params.seed) | 1u;
    var a0 = vec4<u32>(base, base + 1u, base + 2u, base + 3u);
    var a1 = a0 + vec4<u32>(5u); var a2 = a0 + vec4<u32>(10u);
    var a3 = a0 + vec4<u32>(15u); var a4 = a0 + vec4<u32>(20u);
    var a5 = a0 + vec4<u32>(25u); var a6 = a0 + vec4<u32>(30u);
    var a7 = a0 + vec4<u32>(35u);
    for (var i = 0u; i < params.loop_count; i++) {
        a0 = a0 * vec4<u32>(1664525u) + vec4<u32>(1013904223u);
        a1 = a1 * vec4<u32>(22695477u) + vec4<u32>(1u);
        a2 = a2 * vec4<u32>(1103515245u) + vec4<u32>(12345u);
        a3 = a3 * vec4<u32>(214013u) + vec4<u32>(2531011u);
        a4 = a4 * vec4<u32>(134775813u) + vec4<u32>(1u);
        a5 = a5 * vec4<u32>(69069u) + vec4<u32>(362437u);
        a6 = a6 * vec4<u32>(747796405u) + vec4<u32>(2891336453u);
        a7 = a7 * vec4<u32>(277803737u) + vec4<u32>(18782u);
    }
    output[index] = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
}
"#;

pub const INT8_PACKED: &str = r#"
struct Params { loop_count: u32, seed: u32, invocation_count: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read_write> output: array<vec4<i32>>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= params.invocation_count) { return; }
    var a0 = i32(index ^ params.seed); var a1 = a0 + 1; var a2 = a0 + 2; var a3 = a0 + 3;
    var a4 = a0 + 4; var a5 = a0 + 5; var a6 = a0 + 6; var a7 = a0 + 7;
    for (var i = 0u; i < params.loop_count; i++) {
        let salt = i * 0x9e3779b9u + params.seed;
        a0 += dot4I8Packed(bitcast<u32>(a0) ^ salt, 0x01020304u);
        a1 += dot4I8Packed(bitcast<u32>(a1) ^ salt, 0x05060708u);
        a2 += dot4I8Packed(bitcast<u32>(a2) ^ salt, 0x090a0b0cu);
        a3 += dot4I8Packed(bitcast<u32>(a3) ^ salt, 0x0d0e0f10u);
        a4 += dot4I8Packed(bitcast<u32>(a4) ^ salt, 0x11121314u);
        a5 += dot4I8Packed(bitcast<u32>(a5) ^ salt, 0x15161718u);
        a6 += dot4I8Packed(bitcast<u32>(a6) ^ salt, 0x191a1b1cu);
        a7 += dot4I8Packed(bitcast<u32>(a7) ^ salt, 0x1d1e1f20u);
    }
    output[index] = vec4<i32>(a0 ^ a4, a1 ^ a5, a2 ^ a6, a3 ^ a7);
}
"#;
