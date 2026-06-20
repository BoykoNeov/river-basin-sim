# Viewer (Godot 4.x)

The interactive 3D viewer. It **reads files the solver writes** — it never shares
memory or a process with the solver (HANDOFF §4, §7.4). Responsibilities:

- Scenario setup UI → writes the scenario TOML.
- Launches the solver as a subprocess (`python -m solver.run --config <path>`).
- Polls `status.json` for progress.
- On completion, streams the per-frame tile manifest and renders the timeline +
  3D camera, with depth/velocity colormaps and a water surface lifted off the bed.

## Status

Placeholder. The Godot 4.x project (`project.godot`, `scenes/`, `scripts/`,
`shaders/`) is created in **M0** (a static terrain heightmap scene) and fleshed out
in **M2** (timeline scrubber, depth colormap, water surface, subprocess control).

Terrain LOD at reach scale is the viewer's one demanding job — either the
`Terrain3D` plugin or a custom clipmap/quadtree heightmap (HANDOFF §5, §12).

Godot caches (`.godot/`, `.import/`) are git-ignored.
