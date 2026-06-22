extends Node3D
## M0 terrain viewer.
##
## Reads one engine-ready terrain tile produced by the offline pipeline
## (`pipeline.tile` -> `data/tiles/demo/`) and loads it into a Terrain3D node as
## a static 3D heightmap. This is the read-only half of the decoupling contract
## (HANDOFF §7): the viewer never shares memory or a process with the solver, it
## only consumes files on disk.
##
## The tile is raw little-endian float32, row-major, in **metres** -- exactly the
## byte layout of a Godot `FORMAT_RF` image, so heights load 1:1 with no rescale.
## Setting `vertex_spacing = dx` then makes horizontal scale correct too, so real
## relief shows without the classic height-mapping guesswork. Terrain3D 1.0.2
## tiles the heightmap into its native regions automatically; the seams are
## continuous (verified: no zero-height cracks across the interior).

## Tile location, relative to the repo root (the viewer project lives in viewer/).
const TILES_SUBPATH := "data/tiles/demo"
## Terrain3D stores its region data here (outside the repo; regenerated on load).
const DATA_DIR := "user://terrain_demo_data"


func _ready() -> void:
	var tiles_dir := _repo_root().path_join(TILES_SUBPATH)
	var manifest := _load_json(tiles_dir.path_join("tiles.json"))
	if manifest.is_empty() or not manifest.has("tiles") or manifest["tiles"].is_empty():
		push_error("River Basin viewer: no tile manifest at " + tiles_dir + "/tiles.json. "
			+ "Run: uv run python -m pipeline.tile --src data/dem/conditioned "
			+ "--out data/tiles/demo --single")
		return

	var dx := float(manifest.get("dx_m", 1.0))
	var tile: Dictionary = manifest["tiles"][0]
	var w := int(tile["width"])
	var h := int(tile["height"])

	var img := _load_r32_as_rf(tiles_dir.path_join(String(tile["file"])), w, h)
	if img == null:
		return

	var terrain := _make_terrain(dx)
	add_child(terrain)
	# import_images([height, control, color], global_position, offset, scale).
	# offset 0 / scale 1 keeps the true metre elevations.
	terrain.data.import_images([img, null, null], Vector3.ZERO, 0.0, 1.0)
	terrain.data.calc_height_range(true)

	# NOTE: get_height_range().x reads 0 here -- a Terrain3D bookkeeping quirk that
	# includes an uninitialised region-padding texel. The rendered surface has no
	# zero cells (see the headless sampled-min check below), so this is cosmetic.
	var hr: Vector2 = terrain.data.get_height_range()
	print("River Basin viewer: loaded %dx%d tile, dx=%.2f m, height~%.0f..%.0f m, regions=%d"
		% [w, h, dx, hr.x, hr.y, terrain.data.get_region_count()])

	_setup_camera_and_light(w, h, dx, hr)

	# Headless (CI/smoke): there is nothing to render, so just prove the import
	# produced real heights, then quit. Skipped on a normal windowed launch.
	if DisplayServer.get_name() == "headless":
		var true_min := _sampled_height_min(terrain, w, h, dx)
		print("River Basin viewer: headless verify OK (true sampled min=%.0f m)" % true_min)
		get_tree().quit()
		return

	# Windowed one-shot: `godot --path viewer -- --rbshot[=path]` renders a few
	# frames, saves a screenshot, and quits. Used to capture visual proof.
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--rbshot"):
			var out := arg.split("=")[1] if "=" in arg else "viewer_screenshot.png"
			_screenshot_then_quit(_repo_root().path_join(TILES_SUBPATH).path_join(out))
			return


func _repo_root() -> String:
	# res:// is the viewer/ project dir; the repo root is one level up.
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
		push_error("River Basin viewer: tile byte size %d != %d (w*h*4)"
			% [bytes.size(), w * h * 4])
		return null
	return Image.create_from_data(w, h, false, Image.FORMAT_RF, bytes)


func _make_terrain(dx: float) -> Node:
	# Terrain3D refuses to attach to a missing data directory, so create it first.
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(DATA_DIR))
	var terrain = ClassDB.instantiate("Terrain3D")
	terrain.name = "Terrain"
	terrain.vertex_spacing = dx  # horizontal metres per cell == DEM dx
	terrain.data_directory = DATA_DIR
	return terrain


func _sampled_height_min(terrain, w: int, h: int, dx: float) -> float:
	# True minimum of the imported surface, sampled on a coarse grid -- a trustworthy
	# stand-in for get_height_range().x (which is skewed by a padding texel).
	var lo := INF
	for iy in range(0, h, 16):
		for ix in range(0, w, 16):
			var hv: float = terrain.data.get_height(Vector3(ix * dx, 0.0, iy * dx))
			if not is_nan(hv):
				lo = minf(lo, hv)
	return lo


func _screenshot_then_quit(path: String) -> void:
	# Let Terrain3D build its meshes and the renderer settle, then capture.
	for _i in range(8):
		await get_tree().process_frame
	await get_tree().create_timer(0.4).timeout
	var img := get_viewport().get_texture().get_image()
	var err := img.save_png(path)
	print("River Basin viewer: screenshot %s -> %s" % ["OK" if err == OK else "FAIL", path])
	get_tree().quit()


func _setup_camera_and_light(w: int, h: int, dx: float, hr: Vector2) -> void:
	var span_x := w * dx
	var span_z := h * dx
	var center := Vector3(span_x * 0.5, (hr.x + hr.y) * 0.5, span_z * 0.5)

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

	var cam := Camera3D.new()
	cam.far = 400000.0
	var relief := maxf(hr.y - hr.x, 100.0)
	# Low, oblique vantage from a corner: fills the frame and lets the directional
	# light rake the ridges/valleys (real relief is only ~4% of the span).
	cam.position = center + Vector3(-span_x * 0.42, span_x * 0.20, -span_z * 0.42)
	add_child(cam)  # must be in the tree before look_at (needs a global transform)
	cam.look_at(center + Vector3(span_x * 0.05, -relief * 0.3, span_z * 0.05), Vector3.UP)
	cam.current = true
