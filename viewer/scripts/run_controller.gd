extends Node
## Solver run controller (M2, HANDOFF §7.4 -- the subprocess half of the loop).
##
## Launches the solver as a **non-blocking** subprocess and polls the `status.json`
## it writes. This is the viewer's only coupling to the solver: it launches a
## process and reads files (§4). It never imports solver code or shares memory.
##
## Windows launch: Godot's `OS.create_process` cannot set a working directory, and
## `python -m solver.run` needs the repo root on `sys.path`, so we write a tiny
## batch file that `cd`s into the repo and runs the solver via `uv run` (which
## resolves the project venv). Launching the .bat by absolute path sidesteps all
## of cmd.exe's quoting pitfalls -- the repo path contains a space ("River Basin").

signal progress(state: String, fraction: float, sim_time: float, message: String)
signal finished(success: bool, message: String)

const POLL_HZ := 4.0

var _repo_root := ""
var _status_path := ""
var _pid := -1
var _timer: Timer = null
var _running := false


func _ready() -> void:
	_timer = Timer.new()
	_timer.wait_time = 1.0 / POLL_HZ
	_timer.one_shot = false
	_timer.timeout.connect(_poll)
	add_child(_timer)


## Launch `solver.run --config <config_rel> --out <out_rel>` from `repo_root`.
## Paths are repo-relative POSIX-ish strings; the solver resolves them from cwd.
func launch(repo_root: String, config_rel: String, out_rel: String) -> bool:
	if _running:
		push_warning("run_controller: a run is already in progress")
		return false
	_repo_root = repo_root
	# The solver writes status.json next to the Zarr store (run.py: <out-dir>/status.json),
	# so derive the poll path from out_rel rather than hardcoding data/results -- a caller
	# passing a different --out would otherwise leave us polling the wrong (never-updated) file.
	_status_path = repo_root.path_join(out_rel).get_base_dir().path_join("status.json")

	# Clear any stale status so the poller can't read a previous run's "done".
	if FileAccess.file_exists(_status_path):
		DirAccess.remove_absolute(_status_path)

	var cmd := _build_command(repo_root, config_rel, out_rel)
	if cmd.is_empty():
		emit_signal("finished", false, "unsupported platform for solver launch")
		return false

	_pid = OS.create_process(cmd["exe"], cmd["args"])
	if _pid <= 0:
		emit_signal("finished", false, "failed to launch solver process")
		return false

	_running = true
	emit_signal("progress", "starting", 0.0, 0.0, "launching solver...")
	_timer.start()
	return true


func is_running() -> bool:
	return _running


func _build_command(repo_root: String, config_rel: String, out_rel: String) -> Dictionary:
	var win_root := repo_root.replace("/", "\\")
	if OS.get_name() == "Windows":
		var bat := repo_root.path_join("data/results/_launch_solver.bat")
		# Capture the solver's stdout/stderr to a log next to the results: the
		# subprocess has no console we can see, so this is what the "see console/logs"
		# failure message points at (it's how the status.json write-race was found).
		var lines := "@echo off\r\n"
		lines += "cd /d \"%s\"\r\n" % win_root
		lines += "uv run python -m solver.run --config \"%s\" --out \"%s\" > \"data\\results\\solver_stdout.log\" 2>&1\r\n" % [config_rel, out_rel]
		var f := FileAccess.open(bat, FileAccess.WRITE)
		if f == null:
			return {}
		f.store_string(lines)
		f.close()
		return {"exe": "cmd.exe", "args": ["/c", ProjectSettings.globalize_path(bat)]}

	# POSIX fallback (not the primary target, but keep the loop launchable).
	var sh := repo_root.path_join("data/results/_launch_solver.sh")
	var body := "#!/usr/bin/env bash\ncd \"%s\" || exit 1\n" % repo_root
	body += "uv run python -m solver.run --config \"%s\" --out \"%s\"\n" % [config_rel, out_rel]
	var fp := FileAccess.open(sh, FileAccess.WRITE)
	if fp == null:
		return {}
	fp.store_string(body)
	fp.close()
	return {"exe": "bash", "args": [ProjectSettings.globalize_path(sh)]}


func _poll() -> void:
	if not _running:
		return

	var state := ""
	if FileAccess.file_exists(_status_path):
		var text := FileAccess.get_file_as_string(_status_path)
		if not text.is_empty():  # atomic writes make partial reads rare; be defensive
			var rec = JSON.parse_string(text)
			if rec is Dictionary and rec.has("state"):
				state = String(rec["state"])
				emit_signal(
					"progress",
					state,
					float(rec.get("progress", 0.0)),
					float(rec.get("sim_time", 0.0)),
					String(rec.get("message", "")),
				)
				if state == "done":
					_finish(true, String(rec.get("message", "")))
					return
				if state == "error":
					_finish(false, String(rec.get("message", "")))
					return

	# No terminal status yet. If the process has exited, it died without reporting
	# done/error (crash, config error before status, missing interpreter). Surface
	# it instead of polling forever -- the honest failure path the milestone owes.
	if _pid > 0 and not OS.is_process_running(_pid):
		_finish(false, "solver exited without a terminal status -- see console/logs")


func _finish(success: bool, message: String) -> void:
	_running = false
	_timer.stop()
	emit_signal("finished", success, message)
