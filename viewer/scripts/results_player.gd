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
var _bed_img: Image = null
var _grid_w := 0
var _grid_h := 0
var _dx := 1.0

var _terrain: Node = null
var _water: MeshInstance3D = null
var _water_mat: ShaderMaterial = null
var _depth_tex: ImageTexture = null

var _manifest: Dictionary = {}
var _frames: Array = []
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

	if not _load_terrain():
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
		_set_status("no results yet -- press Run solver")

	_handle_cmdline()


# --- terrain (M0 load path) --------------------------------------------------

func _load_terrain() -> bool:
	var tiles_dir := _repo_root.path_join(TILES_SUBPATH)
	var manifest := _load_json(tiles_dir.path_join("tiles.json"))
	if manifest.is_empty() or not manifest.has("tiles") or manifest["tiles"].is_empty():
		push_error("River Basin: no tile manifest at " + tiles_dir + "/tiles.json")
		return false

	_dx = float(manifest.get("dx_m", 1.0))
	var tile: Dictionary = manifest["tiles"][0]
	_grid_w = int(tile["width"])
	_grid_h = int(tile["height"])
	_bed_img = _load_r32_as_rf(tiles_dir.path_join(String(tile["file"])), _grid_w, _grid_h)
	if _bed_img == null:
		return false

	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(DATA_DIR))
	_terrain = ClassDB.instantiate("Terrain3D")
	_terrain.name = "Terrain"
	_terrain.vertex_spacing = _dx
	_terrain.data_directory = DATA_DIR
	add_child(_terrain)
	_terrain.data.import_images([_bed_img, null, null], Vector3.ZERO, 0.0, 1.0)
	_terrain.data.calc_height_range(true)
	var hr: Vector2 = _terrain.data.get_height_range()
	print("River Basin viewer: terrain %dx%d dx=%.2fm height~%.0f..%.0f m"
		% [_grid_w, _grid_h, _dx, hr.x, hr.y])
	return true


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


func _read_frame_image(i: int) -> Image:
	var fr: Dictionary = _frames[i]
	var rel := String(fr["files"]["depth"])
	var path := _repo_root.path_join(RESULTS_SUBPATH).path_join("frames").path_join(rel)
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		push_error("River Basin viewer: frame not found: " + path)
		return Image.create(_grid_w, _grid_h, false, Image.FORMAT_RF)
	var bytes := f.get_buffer(f.get_length())
	f.close()
	return Image.create_from_data(_grid_w, _grid_h, false, Image.FORMAT_RF, bytes)


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
	var hr: Vector2 = _terrain.data.get_height_range()
	var span_x := _grid_w * _dx
	var span_z := _grid_h * _dx
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
	cam.position = center + Vector3(-span_x * 0.42, span_x * 0.20, -span_z * 0.42)
	add_child(cam)
	cam.look_at(center + Vector3(span_x * 0.05, -relief * 0.3, span_z * 0.05), Vector3.UP)
	cam.current = true


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
		for y in range(0, _grid_h, 8):
			for x in range(0, _grid_w, 8):
				if img.get_pixel(x, y).r >= 0.001:
					wet += 1
	print("River Basin viewer: headless verify %s (frames=%d, wet_samples=%d)"
		% ["OK" if ok and wet > 0 else "FAIL", _frames.size(), wet])
	get_tree().quit(0 if ok and wet > 0 else 1)


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
