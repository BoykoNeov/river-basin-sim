extends Node3D
## Results player (M2, HANDOFF §7 -- the explore end of configure -> run -> explore).
##
## The read-only half of the decoupling contract (§4): it launches the solver as a
## subprocess (via run_controller) and otherwise only *reads* files the solver
## wrote -- the M0 terrain tile, and the §7.3 per-frame viewer stream
## (`data/results/frames/`). It renders the terrain (Terrain3D, as in M0), lifts a
## depth-coloured water surface over it, and lets you scrub a timeline of frames.
##
## Byte conventions match M0/viewer_export.py exactly: tiles are raw little-endian
## float32, row-major, metres -> a Godot FORMAT_RF image, no rescale, no transpose.
##
## M6 (reach scale): the results geometry comes from the frame manifest, not from
## the terrain tile -- a run may cover a tile *mosaic* and/or be coarsened, so its
## depth field need not share the terrain tile's shape or cell size -- and a frame
## may arrive as a row-major `tile_grid` of `.raw` tiles, blitted into one image.
## A small domain still exports one file per frame and takes the same path as M2.
##
## The **terrain** follows the same rule. It used to be the M0 tile set's *first*
## tile, which through M5 was the run domain but at reach scale is one patch of a
## mosaic under a much wider water sheet -- water and terrain each correct, the
## composite broken. So when the manifest carries `static.bed` (the solver's own bed,
## already mosaicked / windowed / coarsened / un-shifted), that is the terrain, and
## extent, origin and cell size agree by construction. The M0 tile is the fallback,
## for the pre-run scene and for stores written before the bed shipped with frames.
##
## Terrain is therefore built *from results* and rebuilt whenever they change (a run
## finishing can change the grid under us), not once in `_ready`.
##
## The **sub-grid channel geometry** travels the same way and for the same reason. The
## shader used to reconstruct every surface as `bed + depth`, the M1 relation, which over
## a river cell draws the sheet up to the bank-full depth `d` too high (2.74 m on the M6
## demo). So `manifest["static"]` now ships `channel_width`/`channel_depth` too and the
## shader takes the M6 storage curve where the water is overbank. Below bank full it
## draws the **bank**, flat, rather than the true surface: that surface is *under* the
## floodplain bed -- up to 2.46 m under it on the same demo -- and the rendered terrain
## has no sub-grid trench to hold it, so drawing it there would hide the river inside the
## ground. That residual is one-sided, measured by the export per run, and printed
## (`_report_channel_surface`) for the same reason gap fill and a stale morphed bed are:
## the picture has to be able to say what it is not. See
## `docs/plans/viewer-channel-surface.md`.

const TILES_SUBPATH := "data/tiles/demo"
const RESULTS_SUBPATH := "data/results"
const CONFIG_REL := "scenarios/demo_basin_rain.toml"
const OUT_REL := "data/results/demo.zarr"
const DATA_DIR := "user://terrain_results_data"
const RunController := preload("res://scripts/run_controller.gd")

# Water mesh resolution cap (segments per side). The mesh only carries the lifted
# surface; the fragment shader samples the full-res depth texture for crisp edges,
# so a coarser mesh is fine and keeps vertex count sane on a 1024-cell tile.
const WATER_SEGMENTS := 512
const PLAY_FPS := 4.0

var _repo_root := ""
# The bed currently rendered/lifted over: the run's own bed when the results
# manifest ships one, else the M0 terrain tile.
var _bed_img: Image = null
var _bed_from_results := false
# Exported bed min/max from `manifest["static"]`, to check the *imported* surface
# against. Zero when the results ship no bed.
var _bed_range := Vector2.ZERO
# Sub-grid channel geometry from `manifest["static"]` (M6 runs only): what lets the
# shader take the storage curve instead of lifting every cell as `bed + depth`. False
# for a run without channels, and then the water surface is the pre-M6 one exactly.
var _has_channels := false
var _chan_w_img: Image = null
var _chan_d_img: Image = null
# Depth-buffer offset for water drawn *at* the bank, metres. Not physics: an in-bank
# surface is coplanar with the terrain, and coplanar surfaces z-fight.
const BANK_BIAS := 0.05
# M0 terrain-tile geometry (the fallback surface, and nothing else).
var _grid_w := 0
var _grid_h := 0
var _dx := 1.0

var _terrain: Node = null
var _camera: Camera3D = null
# Geometry the terrain was actually built at. Kept so registration with the results
# grid is an assertable fact (`--rbverify`) rather than an assumption.
var _terrain_w := 0
var _terrain_h := 0
var _terrain_dx := 0.0
var _water: MeshInstance3D = null
var _water_mat: ShaderMaterial = null
var _depth_tex: ImageTexture = null

var _manifest: Dictionary = {}
var _frames: Array = []
# Results-grid geometry, from the frame manifest rather than the terrain tile: at
# reach scale (M6) a run may cover a tile *mosaic* and/or be coarsened, so the
# depth field's shape and cell size need not match the terrain tile the viewer
# renders. Defaults to the terrain grid until a manifest says otherwise.
var _res_w := 0
var _res_h := 0
var _res_dx := 1.0
# Per-frame tile geometry (M6, §7.3 tile_grid). Empty => one file per frame.
var _res_tiles: Array = []
var _frame := 0
var _playing := false
var _play_accum := 0.0

var _controller: Node = null

# UI
var _slider: HSlider = null
var _time_label: Label = null
var _status_label: Label = null
var _play_btn: Button = null
var _run_btn: Button = null


func _ready() -> void:
	_repo_root = _repo_root_path()

	if not _load_fallback_bed():
		return
	_build_environment()
	_build_water()
	_build_ui()

	_controller = RunController.new()
	add_child(_controller)
	_controller.progress.connect(_on_run_progress)
	_controller.finished.connect(_on_run_finished)

	# Load any results already on disk so the timeline is live immediately.
	var manifest_path := _repo_root.path_join(RESULTS_SUBPATH).path_join("frames/manifest.json")
	if FileAccess.file_exists(manifest_path):
		_load_results(manifest_path)
	else:
		# Nothing has been run: show the M0 tile so the scene is not empty. A run
		# replaces this outright -- the terrain is a property of the results.
		_apply_geometry(_bed_img, _grid_w, _grid_h, _dx)
		_set_status("no results yet -- press Run solver")

	_handle_cmdline()


# --- terrain -----------------------------------------------------------------

func _load_fallback_bed() -> bool:
	## Read the M0 terrain tile: the surface shown before anything has been run, and
	## the fallback for a results store written before the bed shipped with frames.
	## It is tile 0 on purpose -- an M0 tile set has no run to define a window, so
	## there is no "right" mosaic to assemble here, and guessing one would render a
	## domain no run ever used.
	var tiles_dir := _repo_root.path_join(TILES_SUBPATH)
	var manifest := _load_json(tiles_dir.path_join("tiles.json"))
	if manifest.is_empty() or not manifest.has("tiles") or manifest["tiles"].is_empty():
		push_error("River Basin: no tile manifest at " + tiles_dir + "/tiles.json")
		return false

	_dx = float(manifest.get("dx_m", 1.0))
	var tile: Dictionary = manifest["tiles"][0]
	_grid_w = int(tile["width"])
	_grid_h = int(tile["height"])
	# Until a results manifest says otherwise, the results grid is the terrain grid.
	_res_w = _grid_w
	_res_h = _grid_h
	_res_dx = _dx
	_bed_img = _load_r32_as_rf(tiles_dir.path_join(String(tile["file"])), _grid_w, _grid_h)
	return _bed_img != null


func _build_terrain(bed: Image, w: int, h: int, dx: float) -> void:
	## (Re)build the Terrain3D node on a given bed. Re-entrant: results can arrive
	## after `_ready` and can change the grid, so the old node and its region data go
	## first -- Terrain3D keeps regions keyed by world position, and importing a
	## smaller domain over a larger one leaves the difference standing. It rebuilds
	## unconditionally (this runs per *load*, not per frame): matching dimensions do
	## not mean the same bed, and a stale surface under new water is the bug this
	## whole path exists to remove.
	if _terrain != null:
		remove_child(_terrain)
		_terrain.queue_free()
		_terrain = null
	var data_path := ProjectSettings.globalize_path(DATA_DIR)
	_clear_dir(data_path)
	DirAccess.make_dir_recursive_absolute(data_path)

	_terrain = ClassDB.instantiate("Terrain3D")
	_terrain.name = "Terrain"
	_terrain.vertex_spacing = dx
	_terrain.data_directory = DATA_DIR
	add_child(_terrain)
	# One assembled image, one import: the mosaic is stitched before Terrain3D sees
	# it (as frames are), so region placement never has to be reasoned about.
	_terrain.data.import_images([bed, null, null], Vector3.ZERO, 0.0, 1.0)
	_terrain.data.calc_height_range(true)
	_terrain_w = w
	_terrain_h = h
	_terrain_dx = dx
	var hr: Vector2 = _terrain.data.get_height_range()
	print("River Basin viewer: terrain %dx%d dx=%.2fm (%.1f x %.1f km) height~%.0f..%.0f m %s"
		% [w, h, dx, w * dx / 1000.0, h * dx / 1000.0, hr.x, hr.y,
			"[run bed]" if _bed_from_results else "[M0 tile]"])


func _clear_dir(abs_path: String) -> void:
	var d := DirAccess.open(abs_path)
	if d == null:
		return
	d.list_dir_begin()
	var entry := d.get_next()
	while entry != "":
		if not d.current_is_dir():
			d.remove(entry)
		entry = d.get_next()
	d.list_dir_end()


func _apply_geometry(bed: Image, w: int, h: int, dx: float) -> void:
	## Put terrain, water and camera on one grid. Everything that depends on the
	## domain goes through here, so they cannot be fitted to different extents.
	if bed == null or w <= 0 or h <= 0:
		return
	_build_terrain(bed, w, h, dx)
	_fit_water(w, h, dx)
	if _water_mat:
		# The shader lifts water by sampling the bed: it must be the *run's* bed, or
		# the surface is reconstructed over elevations the solver never used.
		_water_mat.set_shader_parameter("bed_tex", ImageTexture.create_from_image(bed))
		# Bank full is `w*d/dx`, so the storage curve needs the cell size this grid was
		# fitted at -- it belongs here with everything else the domain sets.
		_water_mat.set_shader_parameter("cell_size", dx)
	_fit_camera(w, h, dx)


# --- water surface -----------------------------------------------------------

func _build_water() -> void:
	var plane := PlaneMesh.new()
	plane.size = Vector2(_grid_w * _dx, _grid_h * _dx)
	var seg := mini(WATER_SEGMENTS, maxi(_grid_w, _grid_h))
	plane.subdivide_width = seg
	plane.subdivide_depth = seg

	_water_mat = ShaderMaterial.new()
	_water_mat.shader = load("res://shaders/water_surface.gdshader")
	_water_mat.set_shader_parameter("bed_tex", ImageTexture.create_from_image(_bed_img))
	_water_mat.set_shader_parameter("colormap_tex", _viridis_texture())
	_water_mat.set_shader_parameter("h_dry", 0.001)
	_water_mat.set_shader_parameter("depth_max", 1.0)
	_water_mat.set_shader_parameter("flip_v", false)
	# No channels until a results manifest ships them: the samplers are bound to 1x1
	# zeros regardless, because an unbound sampler is undefined, not "off".
	_set_channel_geometry(null, null)
	_water_mat.set_shader_parameter("cell_size", _dx)
	_water_mat.set_shader_parameter("bank_bias", BANK_BIAS)

	_water = MeshInstance3D.new()
	_water.name = "Water"
	_water.mesh = plane
	_water.material_override = _water_mat
	# Plane is centred on its origin; shift so it spans [0, w*dx] x [0, h*dx] to
	# register with the terrain (imported at world origin in M0).
	_water.position = Vector3(_grid_w * _dx * 0.5, 0.0, _grid_h * _dx * 0.5)
	# The vertex shader displaces Y arbitrarily; give it a generous AABB so Godot
	# never frustum-culls it when the camera is close to the lifted surface.
	_water.custom_aabb = AABB(
		Vector3(-_grid_w * _dx, -10000.0, -_grid_h * _dx),
		Vector3(2.0 * _grid_w * _dx, 20000.0, 2.0 * _grid_h * _dx)
	)
	_water.visible = false
	add_child(_water)


func _set_channel_geometry(w_img: Image, d_img: Image) -> void:
	## Bind (or unbind) the sub-grid channel fields the shader takes the storage curve
	## from. Pass nulls for a run without channels: the samplers are still bound, to 1x1
	## zeros, because a zero width is what switches the shader back to `bed + depth` and
	## an *unbound* sampler is undefined rather than off. Called on every results load,
	## so a channel-free run loaded after a channelled one cannot inherit its river.
	if _water_mat == null:
		return
	_has_channels = w_img != null and d_img != null
	_chan_w_img = w_img
	_chan_d_img = d_img
	var blank := Image.create(1, 1, false, Image.FORMAT_RF)
	blank.set_pixel(0, 0, Color(0.0, 0.0, 0.0, 1.0))
	var wt := ImageTexture.create_from_image(w_img if _has_channels else blank)
	var dt := ImageTexture.create_from_image(d_img if _has_channels else blank)
	_water_mat.set_shader_parameter("chan_w_tex", wt)
	_water_mat.set_shader_parameter("chan_d_tex", dt)
	_water_mat.set_shader_parameter("has_channels", _has_channels)


func _viridis_texture() -> ImageTexture:
	# Compact viridis approximation (perceptually uniform, colour-blind friendly).
	var stops := [
		Color(0.267, 0.005, 0.329),
		Color(0.229, 0.322, 0.545),
		Color(0.128, 0.567, 0.551),
		Color(0.369, 0.788, 0.383),
		Color(0.993, 0.906, 0.144),
	]
	var n := 256
	var img := Image.create(n, 1, false, Image.FORMAT_RGB8)
	for i in range(n):
		var t := float(i) / float(n - 1) * (stops.size() - 1)
		var lo := int(floor(t))
		var hi := mini(lo + 1, stops.size() - 1)
		img.set_pixel(i, 0, stops[lo].lerp(stops[hi], t - lo))
	return ImageTexture.create_from_image(img)


# --- results (the §7.3 per-frame stream) -------------------------------------

func _load_results(manifest_path: String) -> void:
	_manifest = _load_json(manifest_path)
	if _manifest.is_empty() or not _manifest.has("frames"):
		_set_status("results manifest missing/invalid")
		return
	_frames = _manifest["frames"]
	if _frames.is_empty():
		_set_status("results manifest has no frames")
		return

	# Take the results geometry from the manifest (M6): mosaic domains and coarsened
	# runs both produce a depth field that is not the terrain tile's shape.
	var mgrid: Dictionary = _manifest.get("grid", {})
	_res_w = int(mgrid.get("width", _grid_w))
	_res_h = int(mgrid.get("height", _grid_h))
	_res_dx = float(_manifest.get("dx", _dx))
	_res_tiles = _manifest.get("tile_grid", {}).get("tiles", [])
	if _res_tiles.size() <= 1:
		_res_tiles = []          # one file per frame -- the M2 path

	# Sub-grid channel geometry, before the geometry is applied: `_apply_geometry` fits
	# the water plane and sets `cell_size`, and the shader needs all three together or
	# it evaluates bank full against the wrong cell size for one frame.
	var chan := _read_results_channels()
	_set_channel_geometry(chan[0], chan[1])

	# The terrain is the run's own bed when the manifest ships one, which is the only
	# surface that registers with the depth field on a mosaic/coarsened domain.
	var run_bed := _read_results_bed()
	if run_bed != null:
		_bed_img = run_bed
		_bed_from_results = true
		var brange: Dictionary = _manifest.get("static", {}).get("bed", {})
		_bed_range = Vector2(float(brange.get("min", 0.0)), float(brange.get("max", 0.0)))
		_apply_geometry(_bed_img, _res_w, _res_h, _res_dx)
	else:
		# Older export, or a store without a bed: fall back to the M0 tile and say so
		# if it does not cover the run -- water over an unrelated DEM looks like a
		# rendering bug and is really a provenance one.
		_bed_from_results = false
		_bed_range = Vector2.ZERO
		_apply_geometry(_bed_img, _grid_w, _grid_h, _dx)
		_fit_water(_res_w, _res_h, _res_dx)
		if _res_w != _grid_w or _res_h != _grid_h or not is_equal_approx(_res_dx, _dx):
			push_warning("River Basin viewer: results %dx%d @ %.2fm but terrain tile is "
				% [_res_w, _res_h, _res_dx]
				+ "%dx%d @ %.2fm -- re-export frames to ship the run's bed"
				% [_grid_w, _grid_h, _dx])
	_report_domain()
	_report_channel_surface()
	_report_morphology()

	var gdepth: Dictionary = _manifest.get("global", {}).get("depth", {})
	var p99 := float(gdepth.get("p99", 1.0))
	_water_mat.set_shader_parameter("depth_max", maxf(p99, 1e-3))
	_water_mat.set_shader_parameter("h_dry", float(_manifest.get("h_dry", 0.001)))

	_depth_tex = ImageTexture.create_from_image(_read_frame_image(0))
	_water_mat.set_shader_parameter("depth_tex", _depth_tex)
	_water.visible = true

	if _slider:
		_slider.max_value = _frames.size() - 1
		_slider.value = 0
	_frame = 0
	_apply_frame(0)
	_set_status("loaded %d frames (colormap 0..%.2f m)" % [_frames.size(), p99])
	print("River Basin viewer: loaded %d result frames" % _frames.size())


func _fit_water(w: int, h: int, dx: float) -> void:
	## Resize/reposition the water plane onto the results extent (M6).
	##
	## Through M5 the results grid was always the terrain tile, so the plane built in
	## `_build_water` already fitted. A mosaic or coarsened run breaks that, and a
	## water sheet stretched over the wrong extent is a silently wrong picture.
	if _water == null or w <= 0 or h <= 0:
		return
	var span_x := w * dx
	var span_z := h * dx
	var plane := _water.mesh as PlaneMesh
	if plane:
		plane.size = Vector2(span_x, span_z)
		var seg := mini(WATER_SEGMENTS, maxi(w, h))
		plane.subdivide_width = seg
		plane.subdivide_depth = seg
	_water.position = Vector3(span_x * 0.5, 0.0, span_z * 0.5)
	_water.custom_aabb = AABB(
		Vector3(-span_x, -10000.0, -span_z),
		Vector3(2.0 * span_x, 20000.0, 2.0 * span_z)
	)


func _frames_dir() -> String:
	return _repo_root.path_join(RESULTS_SUBPATH).path_join("frames")


func _read_raw(path: String, expect_floats: int) -> PackedByteArray:
	## Read a .raw payload, or an empty buffer if it is missing or the wrong size.
	##
	## The size check is the guard against a partial write (the solver may still be
	## writing) or a manifest/grid mismatch: `create_from_data` would read past the
	## buffer, and `blit_rect` would place garbage.
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		push_error("River Basin viewer: frame not found: " + path)
		return PackedByteArray()
	var bytes := f.get_buffer(f.get_length())
	f.close()
	if bytes.size() != expect_floats * 4:
		push_error("River Basin viewer: %s byte size %d != %d"
			% [path.get_file(), bytes.size(), expect_floats * 4])
		return PackedByteArray()
	return bytes


func _read_field_image(entry: Dictionary, field: String, what: String) -> Image:
	## Decode one §7.3 payload -- a whole-frame `.raw`, or the manifest's row-major
	## tile grid blitted into one image. Frames and the static bed are written in the
	## same shape by `viewer_export`, so they decode through the same path; `what` is
	## only for error messages.
	var blank := Image.create(maxi(_res_w, 1), maxi(_res_h, 1), false, Image.FORMAT_RF)
	if _res_tiles.is_empty():
		# One file per field (small domains; the M2 shape, unchanged).
		var rel := String(entry.get("files", {}).get(field, ""))
		if rel.is_empty():
			push_error("River Basin viewer: %s has no %s file" % [what, field])
			return blank
		var bytes := _read_raw(_frames_dir().path_join(rel), _res_w * _res_h)
		if bytes.is_empty():
			return blank
		return Image.create_from_data(_res_w, _res_h, false, Image.FORMAT_RF, bytes)

	# Reach scale (M6): a row-major tile grid; blit each tile into place.
	var names: Array = entry.get("tiles", {}).get(field, [])
	if names.size() != _res_tiles.size():
		push_error("River Basin viewer: %s lists %d %s tiles, manifest geometry has %d"
			% [what, names.size(), field, _res_tiles.size()])
		return blank
	var full := blank
	for t in range(_res_tiles.size()):
		var geom: Dictionary = _res_tiles[t]
		var tw := int(geom["width"])
		var th := int(geom["height"])
		var tbytes := _read_raw(_frames_dir().path_join(String(names[t])), tw * th)
		if tbytes.is_empty():
			continue                      # a bad tile leaves a hole, not a crash
		var tile_img := Image.create_from_data(tw, th, false, Image.FORMAT_RF, tbytes)
		full.blit_rect(tile_img, Rect2i(0, 0, tw, th), Vector2i(int(geom["x"]), int(geom["y"])))
	return full


func _read_frame_image(i: int) -> Image:
	return _read_field_image(_frames[i], "depth", "frame %d" % i)


func _read_results_bed() -> Image:
	## The run's own bed from `manifest["static"]`, or null when the export predates
	## it. Same tile layout as the frames, so the same decoder reads it.
	var static_entry: Dictionary = _manifest.get("static", {})
	if static_entry.is_empty():
		return null
	if static_entry.get("files", {}).get("bed", "") == "" \
			and static_entry.get("tiles", {}).get("bed", []).is_empty():
		return null
	return _read_field_image(static_entry, "bed", "static bed")


func _read_results_channels() -> Array:
	## The run's sub-grid channel width/depth from `manifest["static"]`, as
	## `[width, depth]`, or `[null, null]` when the run had no channels (or the export
	## predates them). Same tile layout as the frames, so the same decoder reads them.
	var static_entry: Dictionary = _manifest.get("static", {})
	var fields: Array = static_entry.get("fields", [])
	if not (fields.has("channel_width") and fields.has("channel_depth")):
		return [null, null]
	return [
		_read_field_image(static_entry, "channel_width", "channel width"),
		_read_field_image(static_entry, "channel_depth", "channel depth"),
	]


func _report_channel_surface() -> void:
	## Say how the river is drawn, and by how much that is an approximation (M6/§7.3).
	##
	## Overbank water takes the exact storage curve; water still inside the channel is
	## drawn at the **bank**, because its true surface is below the floodplain bed and
	## the terrain carries no sub-grid trench. The export measures the resulting offset
	## on the run's own frames, so the number is a property of this picture rather than
	## a bound from the geometry. Same idiom as `_report_domain` / `_report_morphology`.
	var chan: Dictionary = _manifest.get("static", {}).get("channel", {})
	if chan.is_empty():
		return
	if not _has_channels:
		# The manifest declares channels but the fields did not decode: the shader is
		# lifting `bed + depth` over a river, which is exactly the old defect.
		push_warning("River Basin viewer: manifest declares %d channel cells but the "
			% int(chan.get("cells", 0))
			+ "geometry did not load -- water over the river is drawn too high")
		return
	print("River Basin viewer: %d sub-grid channel cells (width <= %.1f m, bank-full "
		% [int(chan.get("cells", 0)), float(chan.get("width_max_m", 0.0))]
		+ "depth <= %.2f m); overbank water takes the storage curve, in-bank water is "
		% float(chan.get("depth_max_m", 0.0))
		+ "drawn at the bank -- up to %.2f m above its true surface on %d cells (frame %d)"
		% [float(chan.get("in_bank_offset_m", 0.0)), int(chan.get("in_bank_cells", 0)),
			int(chan.get("frame", -1))])


func _report_domain() -> void:
	## Say when part of the rendered surface is fill, not terrain. `assemble_mosaic`
	## fills cells no tile covered at the minimum covered elevation; rendered, that is
	## a flat plateau indistinguishable from a bug unless the picture declares it.
	var domain: Dictionary = _manifest.get("domain", {})
	var gaps := int(domain.get("gap_cells", 0))
	if gaps > 0:
		# Counted on the mosaic's own (pre-coarsen) grid, so use its shape as the
		# denominator rather than the results grid.
		var shape: Array = domain.get("shape", [])
		var total := float(shape[0]) * float(shape[1]) if shape.size() == 2 else 0.0
		var pct := 100.0 * gaps / total if total > 0.0 else 0.0
		print("River Basin viewer: %d mosaic cells (%.2f%%) are fill @ %.1f m -- flat by "
			% [gaps, pct, float(domain.get("fill_value", 0.0))]
			+ "construction, not by the terrain")


func _report_morphology() -> void:
	## Say when the terrain shown is the bed the run *started* on (M7).
	##
	## A morphological run moves `z`, but the exported bed is the initial one and M7
	## deliberately does not animate terrain: the shader still lifts water as
	## `bed + depth` rather than through the sub-grid storage curve, so a moving bed
	## would move that mis-lift with it. Fix the lift first, then animate. Until then
	## the picture has to declare what it is not -- with the number, because "the
	## terrain is a little out of date" and "it scoured a metre" are different
	## pictures. Same idiom as `_report_domain`.
	var morph: Dictionary = _manifest.get("morphology", {})
	if morph.is_empty():
		return
	var dz: Dictionary = morph.get("bed_change", {})
	print("River Basin viewer: terrain is the bed at t=0 -- this run moved it by "
		+ "%+.3f..%+.3f m by t = %.0f s (not animated in M7)"
		% [float(dz.get("min", 0.0)), float(dz.get("max", 0.0)), float(dz.get("time", 0.0))])


func _apply_frame(i: int) -> void:
	i = clampi(i, 0, _frames.size() - 1)
	_frame = i
	if _depth_tex:
		_depth_tex.update(_read_frame_image(i))
	var t := float(_frames[i].get("time", 0.0))
	var mx := float(_frames[i].get("depth", {}).get("max", 0.0))
	if _time_label:
		_time_label.text = "frame %d/%d   t = %.0f s (%.1f min)   max %.2f m" % [
			i, _frames.size() - 1, t, t / 60.0, mx]


# --- UI ----------------------------------------------------------------------

func _build_ui() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)

	var panel := PanelContainer.new()
	panel.anchor_left = 0.0
	panel.anchor_top = 1.0
	panel.anchor_right = 1.0
	panel.anchor_bottom = 1.0
	panel.offset_top = -96.0
	layer.add_child(panel)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 4)
	panel.add_child(vbox)

	_status_label = Label.new()
	_status_label.text = "starting..."
	vbox.add_child(_status_label)

	_time_label = Label.new()
	_time_label.text = "no frame"
	vbox.add_child(_time_label)

	var row := HBoxContainer.new()
	vbox.add_child(row)

	_play_btn = Button.new()
	_play_btn.text = "Play"
	_play_btn.pressed.connect(_toggle_play)
	row.add_child(_play_btn)

	_run_btn = Button.new()
	_run_btn.text = "Run solver"
	_run_btn.pressed.connect(_start_run)
	row.add_child(_run_btn)

	_slider = HSlider.new()
	_slider.min_value = 0
	_slider.max_value = 0
	_slider.step = 1
	_slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_slider.custom_minimum_size = Vector2(600, 0)
	_slider.value_changed.connect(_on_slider)
	row.add_child(_slider)


func _toggle_play() -> void:
	if _frames.is_empty():
		return
	_playing = not _playing
	_play_btn.text = "Pause" if _playing else "Play"


func _on_slider(v: float) -> void:
	_playing = false
	if _play_btn:
		_play_btn.text = "Play"
	_apply_frame(int(round(v)))


func _process(delta: float) -> void:
	if not _playing or _frames.is_empty():
		return
	_play_accum += delta
	if _play_accum >= 1.0 / PLAY_FPS:
		_play_accum = 0.0
		var nxt := _frame + 1
		if nxt >= _frames.size():
			nxt = 0
		if _slider:
			_slider.set_value_no_signal(nxt)
		_apply_frame(nxt)


func _set_status(msg: String) -> void:
	if _status_label:
		_status_label.text = msg


# --- solver run (subprocess) -------------------------------------------------

func _start_run() -> void:
	if _controller.is_running():
		return
	_set_status("launching solver...")
	if _run_btn:
		_run_btn.disabled = true
	_controller.launch(_repo_root, CONFIG_REL, OUT_REL)


func _on_run_progress(state: String, fraction: float, sim_time: float, message: String) -> void:
	_set_status("solver: %s  %d%%  (t=%.0fs)  %s" % [state, int(fraction * 100.0), sim_time, message])


func _on_run_finished(success: bool, message: String) -> void:
	if _run_btn:
		_run_btn.disabled = false
	if success:
		_set_status("solver done -- loading results")
		var manifest_path := _repo_root.path_join(RESULTS_SUBPATH).path_join("frames/manifest.json")
		_load_results(manifest_path)
	else:
		_set_status("solver error: " + message)


# --- helpers (shared with the M0 loader) -------------------------------------

func _repo_root_path() -> String:
	return ProjectSettings.globalize_path("res://").path_join("..").simplify_path()


func _load_json(path: String) -> Dictionary:
	if not FileAccess.file_exists(path):
		return {}
	var parsed = JSON.parse_string(FileAccess.get_file_as_string(path))
	return parsed if parsed is Dictionary else {}


func _load_r32_as_rf(path: String, w: int, h: int) -> Image:
	if not FileAccess.file_exists(path):
		push_error("River Basin viewer: tile not found: " + path)
		return null
	var f := FileAccess.open(path, FileAccess.READ)
	var bytes := f.get_buffer(f.get_length())
	f.close()
	if bytes.size() != w * h * 4:
		push_error("River Basin viewer: tile byte size %d != %d" % [bytes.size(), w * h * 4])
		return null
	return Image.create_from_data(w, h, false, Image.FORMAT_RF, bytes)


func _build_environment() -> void:
	## Sun, sky and camera *node*. The camera's placement depends on the domain, which
	## is not known until results load, so it is fitted separately (`_fit_camera`).
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-45.0, -130.0, 0.0)
	sun.light_energy = 1.3
	sun.shadow_enabled = true
	add_child(sun)

	var env := WorldEnvironment.new()
	var e := Environment.new()
	e.background_mode = Environment.BG_COLOR
	e.background_color = Color(0.55, 0.62, 0.72)
	e.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	e.ambient_light_color = Color(0.45, 0.47, 0.52)
	e.ambient_light_energy = 0.6
	env.environment = e
	add_child(env)

	_camera = Camera3D.new()
	add_child(_camera)
	_camera.current = true


func _fit_camera(w: int, h: int, dx: float) -> void:
	## Frame the current domain. Re-run on every geometry change: a camera left on the
	## previous extent is how a correct mosaic ends up looking like a broken one.
	if _camera == null or _terrain == null:
		return
	var hr: Vector2 = _terrain.data.get_height_range()
	var span_x := w * dx
	var span_z := h * dx
	var center := Vector3(span_x * 0.5, (hr.x + hr.y) * 0.5, span_z * 0.5)
	var relief := maxf(hr.y - hr.x, 100.0)
	# Clip planes scale with the domain: a fixed far plane is either too near to see a
	# 76.8 km mosaic or so far that the depth range degenerates (and Godot's light
	# culler starts refusing frustums).
	_camera.near = maxf(dx, 1.0)
	_camera.far = maxf(span_x, span_z) * 4.0
	# Low, oblique vantage from a corner: fills the frame and lets the directional
	# light rake the relief (which is a small fraction of a reach-scale span).
	_camera.position = center + Vector3(-span_x * 0.42, span_x * 0.20, -span_z * 0.42)
	_camera.look_at(center + Vector3(span_x * 0.05, -relief * 0.3, span_z * 0.05), Vector3.UP)
	print("River Basin viewer: camera at (%.0f, %.0f, %.0f) looking at (%.0f, %.0f, %.0f), "
		% [_camera.position.x, _camera.position.y, _camera.position.z,
			center.x, center.y, center.z]
		+ "near=%.1f far=%.0f" % [_camera.near, _camera.far])


# --- CLI hooks (headless verify / screenshot / launch-and-quit) --------------

func _handle_cmdline() -> void:
	for arg in OS.get_cmdline_user_args():
		if arg == "--rbverify":
			_verify_then_quit()
			return
		if arg.begins_with("--rbshot"):
			var out := arg.split("=")[1] if "=" in arg else "results_screenshot.png"
			_screenshot_then_quit(_repo_root.path_join(RESULTS_SUBPATH).path_join(out))
			return
		if arg == "--rblaunch":
			_launch_then_quit()
			return


func _verify_then_quit() -> void:
	# Headless proof the read/render path works: results loaded, a frame decoded to
	# real depths, water made visible. No subprocess, no GPU render needed.
	var ok := not _frames.is_empty() and _water != null and _water.visible
	var wet := 0
	if not _frames.is_empty():
		var img := _read_frame_image(_frames.size() - 1)
		for y in range(0, _res_h, 8):
			for x in range(0, _res_w, 8):
				if img.get_pixel(x, y).r >= 0.001:
					wet += 1
	# And that the terrain is the *run's* domain. Frames-loaded plus water-visible was
	# always true of the tile-0 terrain the water floated over, so registration needs
	# its own assertion: same cell count, same cell size, and -- the claim this whole
	# path rests on -- an imported surface that is the *exported bed*, checked by
	# sampling Terrain3D and bracketing it against the manifest's own min/max.
	# (`get_height_range().x` cannot do this job: it reads 0 off an uninitialised
	# padding texel, so its "relief" is really the maximum elevation.)
	var registered := (_terrain != null and _terrain_w == _res_w and _terrain_h == _res_h
		and is_equal_approx(_terrain_dx, _res_dx))
	var sampled := _sampled_height_range()
	var relief := sampled.y - sampled.x
	var surface_ok := relief > 0.0
	var against := "no exported bed"
	if _bed_range != Vector2.ZERO:
		# Bilinear sampling cannot leave the data range; a stale or partial import
		# shows up as a range that does not sit inside the exported one, or collapses.
		var tol := maxf(1e-3 * (_bed_range.y - _bed_range.x), 0.01)
		surface_ok = (relief >= 0.5 * (_bed_range.y - _bed_range.x)
			and sampled.x >= _bed_range.x - tol and sampled.y <= _bed_range.y + tol)
		against = "bed %.1f..%.1f m" % [_bed_range.x, _bed_range.y]
	# And that a channelled run's river geometry actually reached the shader. The blank
	# a failed tile read returns is the right *size*, so shape proves nothing here --
	# count the channel cells and match the export's own count. Without this the water
	# is silently lifted `bed + depth` over the river again, and no other check looks.
	var chan: Dictionary = _manifest.get("static", {}).get("channel", {})
	var chan_declared := int(chan.get("cells", 0))
	var chan_decoded := _count_channel_cells()
	var channels_ok := chan.is_empty() or (_has_channels and chan_decoded == chan_declared)
	if not channels_ok:
		push_error("River Basin viewer: manifest declares %d channel cells, decoded %d "
			% [chan_declared, chan_decoded]
			+ "(has_channels=%s) -- the river's surface is not the storage curve"
			% _has_channels)
	var terrain_ok := registered and surface_ok
	if not terrain_ok:
		push_error("River Basin viewer: terrain %dx%d @ %.2fm sampled %.1f..%.1f m does "
			% [_terrain_w, _terrain_h, _terrain_dx, sampled.x, sampled.y]
			+ "not match results %dx%d @ %.2fm / %s (bed_from_results=%s)"
			% [_res_w, _res_h, _res_dx, against, _bed_from_results])
	var pass_all := ok and wet > 0 and terrain_ok and channels_ok
	print("River Basin viewer: headless verify %s (frames=%d, wet_samples=%d, "
		% ["OK" if pass_all else "FAIL", _frames.size(), wet]
		+ "terrain=%dx%d @ %.2fm sampled %.1f..%.1f m vs %s, run_bed=%s, "
		% [_terrain_w, _terrain_h, _terrain_dx, sampled.x, sampled.y, against,
			_bed_from_results]
		+ "channel_cells=%d/%d)" % [chan_decoded, chan_declared])
	get_tree().quit(0 if pass_all else 1)


func _count_channel_cells() -> int:
	## Cells whose decoded width *and* depth are positive -- the same test the shader
	## and `viewer_export` use for "this cell has a channel". Full scan: this runs once,
	## in `--rbverify`, and an exact count is the point (a strided sample of a river two
	## cells wide is a coin flip).
	if _chan_w_img == null or _chan_d_img == null:
		return 0
	var n := 0
	for y in range(_chan_w_img.get_height()):
		for x in range(_chan_w_img.get_width()):
			if _chan_w_img.get_pixel(x, y).r > 0.0 and _chan_d_img.get_pixel(x, y).r > 0.0:
				n += 1
	return n


func _sampled_height_range() -> Vector2:
	## Min/max of the *imported* surface, sampled on a coarse grid. Trustworthy where
	## `get_height_range()` is not: that one includes an uninitialised region-padding
	## texel, which reads 0 and turns any "relief" derived from it into the maximum.
	if _terrain == null:
		return Vector2.ZERO
	var lo := INF
	var hi := -INF
	var step := maxi(1, mini(_terrain_w, _terrain_h) / 96)
	for iy in range(0, _terrain_h, step):
		for ix in range(0, _terrain_w, step):
			var hv: float = _terrain.data.get_height(
				Vector3(ix * _terrain_dx, 0.0, iy * _terrain_dx))
			if not is_nan(hv):
				lo = minf(lo, hv)
				hi = maxf(hi, hv)
	return Vector2.ZERO if lo > hi else Vector2(lo, hi)


func _screenshot_then_quit(path: String) -> void:
	if not _frames.is_empty():
		_apply_frame(_frames.size() - 1)  # final flooded frame
	for _i in range(12):
		await get_tree().process_frame
	await get_tree().create_timer(0.5).timeout
	var img := get_viewport().get_texture().get_image()
	var err := img.save_png(path)
	print("River Basin viewer: screenshot %s -> %s" % ["OK" if err == OK else "FAIL", path])
	get_tree().quit()


func _launch_then_quit() -> void:
	# Full-loop smoke: actually launch the solver subprocess and wait for status
	# to reach done/error, printing transitions. Proves §7.4 end to end.
	print("River Basin viewer: launching solver subprocess...")
	_controller.finished.connect(func(success, message):
		print("River Basin viewer: run finished success=%s msg=%s" % [success, message])
		get_tree().quit(0 if success else 1))
	_controller.progress.connect(func(state, frac, st, msg):
		print("  status: %s %d%% t=%.0f %s" % [state, int(frac * 100.0), st, msg]))
	_start_run()
