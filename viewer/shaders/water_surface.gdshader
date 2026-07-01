// Water-surface shader (M2, HANDOFF §7.3 viewer / §2 water surface).
//
// Draws the solver's water as a lifted, depth-coloured surface over the M0
// terrain. The mesh is a flat subdivided plane spanning the domain in XZ; this
// shader lifts every vertex to the water-surface elevation eta = bed + depth
// (Godot Y is up) and colours it by depth, discarding dry ground so terrain
// shows through.
//
// Two textures carry the fields, both raw metres in FORMAT_RF (R32F), sampled by
// the plane's UV (u -> +X -> column, v -> +Z -> row) which matches the row-major
// byte layout the solver writes (viewer_export.py) and the M0 terrain loader:
//   * depth_tex : per-frame water depth h (streamed as the timeline scrubs)
//   * bed_tex   : static bed elevation z (same bytes Terrain3D renders)
// Reconstructing eta in-shader from the shared bed texture means the export only
// ships depth, and water registers with terrain by construction.
//
// depth_max is the manifest's global p99 (a clamped colormap range) so a thin
// floodplain sheet stays visible while a rare deep channel saturates rather than
// washing everything else to the low end.

shader_type spatial;
render_mode cull_disabled, depth_prepass_alpha;

uniform sampler2D depth_tex : filter_linear;      // water depth h, metres
uniform sampler2D bed_tex : filter_linear;        // bed elevation z, metres
uniform sampler2D colormap_tex : filter_linear;   // 1-D depth colormap (viridis)
uniform float h_dry = 0.001;                      // dry threshold (matches solver H_DRY)
uniform float depth_max = 1.0;                     // colormap clamp (global p99), metres
uniform bool flip_v = false;                       // set if terrain/water are N-S mirrored

varying vec2 v_uv;

void vertex() {
	vec2 uv = UV;
	if (flip_v) {
		uv.y = 1.0 - uv.y;
	}
	v_uv = uv;
	float depth = texture(depth_tex, uv).r;
	float bed = texture(bed_tex, uv).r;
	// Lift to the water surface; drop dry vertices onto the bed so the mesh does
	// not tent up over nearly-dry cells (fragment discard removes them anyway).
	VERTEX.y = bed + max(depth, 0.0);
}

void fragment() {
	// Per-fragment depth (full texture resolution) -> crisp wet/dry edges even
	// though the mesh is coarser than the grid.
	float depth = texture(depth_tex, v_uv).r;
	if (depth < h_dry) {
		discard;  // dry ground: show the terrain underneath
	}
	float t = clamp(depth / max(depth_max, 1e-6), 0.0, 1.0);
	vec3 col = texture(colormap_tex, vec2(t, 0.5)).rgb;

	ALBEDO = col;
	// Fade the thin runoff sheet toward transparent so deep channels read clearly
	// (shallow water is barely there physically, too).
	ALPHA = mix(0.35, 0.9, sqrt(t));
	ROUGHNESS = 0.12;
	METALLIC = 0.0;
	SPECULAR = 0.5;
}
