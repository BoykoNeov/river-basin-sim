# Data sources & licensing

Notes on the raw datasets the pipeline conditions into engine-ready tiles. **Verify
the current license for each before redistributing any derived data.** Large raw
files live under `data/` and are git-ignored.

## Digital elevation models (DEMs)

| Source | Resolution | Coverage | License (verify) |
|---|---|---|---|
| **3DEP** (USGS) | 1 m / 10 m | USA | Public domain |
| **SRTM** (NASA) | 30 m / 90 m | ~global (±60° lat) | Public domain |
| **Copernicus DEM** (GLO-30 / GLO-90) | 30 m / 90 m | global | Free, attribution; check ESA terms |
| **MERIT DEM / MERIT Hydro** | 90 m | global | Free for research/education; attribution required |

## River networks / hydrography

| Source | Coverage | License (verify) |
|---|---|---|
| **HydroSHEDS** (incl. HydroRIVERS) | global | Free; attribution; non-commercial terms on some products |
| **MERIT Hydro** | global | Free for research/education; attribution |
| **NHD / NHDPlus** (USGS) | USA | Public domain |

## Conditioning steps (M0)

1. Reproject to a metric CRS (e.g. UTM) so `dx` is in metres.
2. Sink-fill / depression handling.
3. D8 flow direction + flow accumulation.
4. Tile into engine-ready rasters (see `tile.py`).

Tooling: **pysheds** (numpy-native) or **WhiteboxTools** (standalone binary). Not
richdem — see `pipeline/__init__.py`.

## Recording provenance

For each dataset actually used, record here: source URL, download date, exact
product/version, the CRS and bounds of the conditioned output, and the license that
applied at download time.
