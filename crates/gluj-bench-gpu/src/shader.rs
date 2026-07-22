pub const BANDWIDTH_SHADER: &str = r#"
struct Params {
    element_count: u32,
    seed: u32,
    row_width: u32,
    access_count: u32,
}

@group(0) @binding(0) var<storage, read> source: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> destination: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> checksums: array<u32>;

fn invocation_index(gid: vec3<u32>) -> u32 {
    return gid.x + gid.y * params.row_width;
}

fn mapped_index(index: u32) -> u32 {
    if (params.access_count != params.element_count) {
        return index & (params.element_count - 1u);
    }
    return index;
}

fn fold(value: vec4<u32>) -> u32 {
    return value.x ^ value.y ^ value.z ^ value.w;
}

fn write_vector(index: u32) {
    let value = params.seed ^ index;
    destination[index] = vec4<u32>(value, value + 1u, value + 2u, value + 3u);
}

fn copy_vector(index: u32) {
    destination[index] = source[index];
}

// Four coalesced vectors per invocation keeps small cache working sets highly parallel.
@compute @workgroup_size(256)
fn read_cache(@builtin(global_invocation_id) gid: vec3<u32>) {
    let invocation = invocation_index(gid);
    let invocation_count = params.access_count / 4u;
    if (invocation >= invocation_count) { return; }
    var a = source[mapped_index(invocation)];
    var b = source[mapped_index(invocation + invocation_count)];
    a = a ^ source[mapped_index(invocation + invocation_count * 2u)];
    b = b ^ source[mapped_index(invocation + invocation_count * 3u)];
    checksums[invocation] = fold(a ^ b) ^ params.seed;
}

@compute @workgroup_size(256)
fn write_cache(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = invocation_index(gid);
    if (index >= params.element_count) { return; }
    let value = params.seed ^ index;
    destination[index] = vec4<u32>(value, value + 1u, value + 2u, value + 3u);
}

@compute @workgroup_size(256)
fn copy_cache(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = invocation_index(gid);
    if (index >= params.element_count) { return; }
    destination[index] = source[index];
}

// Sixty-four coalesced vectors with eight dependency chains amortize address and checksum traffic.
@compute @workgroup_size(256)
fn read_stream(@builtin(global_invocation_id) gid: vec3<u32>) {
    let invocation = invocation_index(gid);
    let stride = params.access_count / 64u;
    if (invocation >= stride) { return; }
    var a = source[invocation];
    var b = source[invocation + stride];
    var c = source[invocation + stride * 2u];
    var d = source[invocation + stride * 3u];
    var e = source[invocation + stride * 4u];
    var f = source[invocation + stride * 5u];
    var g = source[invocation + stride * 6u];
    var h = source[invocation + stride * 7u];
    a = a ^ source[invocation + stride * 8u];
    b = b ^ source[invocation + stride * 9u];
    c = c ^ source[invocation + stride * 10u];
    d = d ^ source[invocation + stride * 11u];
    e = e ^ source[invocation + stride * 12u];
    f = f ^ source[invocation + stride * 13u];
    g = g ^ source[invocation + stride * 14u];
    h = h ^ source[invocation + stride * 15u];
    a = a ^ source[invocation + stride * 16u];
    b = b ^ source[invocation + stride * 17u];
    c = c ^ source[invocation + stride * 18u];
    d = d ^ source[invocation + stride * 19u];
    e = e ^ source[invocation + stride * 20u];
    f = f ^ source[invocation + stride * 21u];
    g = g ^ source[invocation + stride * 22u];
    h = h ^ source[invocation + stride * 23u];
    a = a ^ source[invocation + stride * 24u];
    b = b ^ source[invocation + stride * 25u];
    c = c ^ source[invocation + stride * 26u];
    d = d ^ source[invocation + stride * 27u];
    e = e ^ source[invocation + stride * 28u];
    f = f ^ source[invocation + stride * 29u];
    g = g ^ source[invocation + stride * 30u];
    h = h ^ source[invocation + stride * 31u];
    a = a ^ source[invocation + stride * 32u];
    b = b ^ source[invocation + stride * 33u];
    c = c ^ source[invocation + stride * 34u];
    d = d ^ source[invocation + stride * 35u];
    e = e ^ source[invocation + stride * 36u];
    f = f ^ source[invocation + stride * 37u];
    g = g ^ source[invocation + stride * 38u];
    h = h ^ source[invocation + stride * 39u];
    a = a ^ source[invocation + stride * 40u];
    b = b ^ source[invocation + stride * 41u];
    c = c ^ source[invocation + stride * 42u];
    d = d ^ source[invocation + stride * 43u];
    e = e ^ source[invocation + stride * 44u];
    f = f ^ source[invocation + stride * 45u];
    g = g ^ source[invocation + stride * 46u];
    h = h ^ source[invocation + stride * 47u];
    a = a ^ source[invocation + stride * 48u];
    b = b ^ source[invocation + stride * 49u];
    c = c ^ source[invocation + stride * 50u];
    d = d ^ source[invocation + stride * 51u];
    e = e ^ source[invocation + stride * 52u];
    f = f ^ source[invocation + stride * 53u];
    g = g ^ source[invocation + stride * 54u];
    h = h ^ source[invocation + stride * 55u];
    a = a ^ source[invocation + stride * 56u];
    b = b ^ source[invocation + stride * 57u];
    c = c ^ source[invocation + stride * 58u];
    d = d ^ source[invocation + stride * 59u];
    e = e ^ source[invocation + stride * 60u];
    f = f ^ source[invocation + stride * 61u];
    g = g ^ source[invocation + stride * 62u];
    h = h ^ source[invocation + stride * 63u];
    checksums[invocation] = fold(a ^ b ^ c ^ d ^ e ^ f ^ g ^ h) ^ params.seed;
}

@compute @workgroup_size(256)
fn write_stream(@builtin(global_invocation_id) gid: vec3<u32>) {
    let invocation = invocation_index(gid);
    let stride = params.element_count / 16u;
    if (invocation >= stride) { return; }
    write_vector(invocation);
    write_vector(invocation + stride);
    write_vector(invocation + stride * 2u);
    write_vector(invocation + stride * 3u);
    write_vector(invocation + stride * 4u);
    write_vector(invocation + stride * 5u);
    write_vector(invocation + stride * 6u);
    write_vector(invocation + stride * 7u);
    write_vector(invocation + stride * 8u);
    write_vector(invocation + stride * 9u);
    write_vector(invocation + stride * 10u);
    write_vector(invocation + stride * 11u);
    write_vector(invocation + stride * 12u);
    write_vector(invocation + stride * 13u);
    write_vector(invocation + stride * 14u);
    write_vector(invocation + stride * 15u);
}

@compute @workgroup_size(256)
fn copy_stream(@builtin(global_invocation_id) gid: vec3<u32>) {
    let invocation = invocation_index(gid);
    let stride = params.element_count / 16u;
    if (invocation >= stride) { return; }
    copy_vector(invocation);
    copy_vector(invocation + stride);
    copy_vector(invocation + stride * 2u);
    copy_vector(invocation + stride * 3u);
    copy_vector(invocation + stride * 4u);
    copy_vector(invocation + stride * 5u);
    copy_vector(invocation + stride * 6u);
    copy_vector(invocation + stride * 7u);
    copy_vector(invocation + stride * 8u);
    copy_vector(invocation + stride * 9u);
    copy_vector(invocation + stride * 10u);
    copy_vector(invocation + stride * 11u);
    copy_vector(invocation + stride * 12u);
    copy_vector(invocation + stride * 13u);
    copy_vector(invocation + stride * 14u);
    copy_vector(invocation + stride * 15u);
}
"#;
