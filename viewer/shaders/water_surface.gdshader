// Water-surface shader (M2, HANDOFF §7.3 viewer / §2 water surface).
//
// Draws the solver's water as a lifted, depth-coloured surface over the run's own
// terrain. The mesh is a flat subdivided plane spanning the domain in XZ; this
// shader lifts every vertex to the water-surface elevation (Godot Y is up) and
// colours it by depth, discarding dry ground so terrain shows through.
//
// Textures carry the fields, all raw metres in FORMAT_RF (R32F), sampled by the
// plane's UV (u -> +X -> column, v -> +Z -> row) which matches the row-major byte
// layout the solver writes (viewer_export.py) and the M0 terrain loader:
//   * depth_tex   : per-frame cell-mean water depth h (streamed as the timeline scrubs)
//   * bed_tex     : static floodplain bed elevation z (the surface Terrain3D renders)
//   * chan_w_tex  : static sub-grid channel width w, 0 where there is no channel (M6)
//   * chan_d_tex  : static sub-grid channel bank-full depth d below the bed (M6)
// Reconstructing the surface in-shader from the shared static fields means the export
// only ships depth per frame, and water registers with terrain by construction.
//
// THE LIFT (viewer_export.render_eta is the reference; keep the two identical):
//
//   no channel :  eta = z + h                    the M1 relation, unchanged
//   h >  h_bf  :  eta = z + (h - h_bf)           the exact M6 storage curve, overbank
//   h <= h_bf  :  eta = z                        the bank, flat
//
//   with h_bf = w*d/dx, the cell-mean depth at bank full. Both channel branches then
//   get `bank_bias` added -- a constant, so the two stay continuous at bank full, and
//   `render_eta` does not model it because it is a z-buffer offset, not a surface.
//
// The third line is the one place this departs from solver/core/channels.py, and it
// departs deliberately: below bank full the true surface z - d + h*dx/w is *under* the
// floodplain bed (up to 2.46 m under it on the M6 demo), and the terrain has no trench
// to put it in because the channel is sub-grid. Drawing the true elevation there hides
// the river inside the ground; drawing z + h -- what this shader did through M7 -- floats
// it up to d above the bank. Drawing the bank sits between them: never above the terrain,
// never buried under it, one-sided and measured (manifest static.channel.in_bank_offset_m).
// `bank_bias` is a depth-buffer offset, not physics: an in-bank surface is coplanar with
// the terrain it is drawn on, and coplanar surfaces z-fight.
//
// The mesh is coarser than the grid (WATER_SEGMENTS vs the results width), so a river one
// cell wide is not resolved by the *geometry* either way -- what this fixes is the sheet
// over and beside it, which is what a reach-scale view actually shows.
//
// Channel width/depth are sampled with **nearest**: channel presence is a genuine
// discontinuity in the parameterization, and interpolating a 20 m width against a 0 m
// neighbour invents a river between them. Depth and bed stay linear (both are smooth
// fields, and the bed must match how Terrain3D interpolates the same bytes).
//
// COLOUR is still the cell-mean depth h, which is the quantity the manifest's colormap
// range (global p99) is computed over. Over a channel cell that is the volume spread
// across the whole cell, not the depth of the river -- the water column there is deeper
// by up to dx/w. Recolouring by the column depth needs its own measured range or every
// river cell saturates, so it is deliberately a separate pass.
//
// depth_max is the manifest's global p99 (a clamped colormap range) so a thin
// floodplain sheet stays visible while a rare deep channel saturates rather than
// washing everything else to the low end.

shader_type spatial;
render_mode cull_disabled, depth_prepass_alpha;

uniform sampler2D depth_tex : filter_linear;      // water depth h, metres
uniform sampler2D bed_tex : filter_linear;        // bed elevation z, metres
uniform sampler2D chan_w_tex : filter_nearest;    // sub-grid channel width, metres
uniform sampler2D chan_d_tex : filter_nearest;    // sub-grid bank-full depth, metres
uniform sampler2D colormap_tex : filter_linear;   // 1-D depth colormap (viridis)
uniform bool has_channels = false;                 // set when the run shipped w/d
uniform float cell_size = 1.0;                     // dx, metres (h_bf = w*d/dx)
uniform float bank_bias = 0.05;                    // depth-buffer offset for in-bank water, m
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
	float depth = max(texture(depth_tex, uv).r, 0.0);
	float bed = texture(bed_tex, uv).r;
	// Lift to the water surface; dry vertices land on the bed so the mesh does not
	// tent up over nearly-dry cells (fragment discard removes them anyway).
	float eta = bed + depth;
	if (has_channels) {
		float w = texture(chan_w_tex, uv).r;
		float d = texture(chan_d_tex, uv).r;
		if (w > 0.0 && d > 0.0) {
			float h_bf = w * d / max(cell_size, 1e-6);
			// Continuous at bank full: both branches give the bank (plus the bias).
			eta = bed + max(depth - h_bf, 0.0) + bank_bias;
		}
	}
	VERTEX.y = eta;
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
