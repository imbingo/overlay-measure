# Changelog

## V1.5.4

- Moved single and batch overlay calculations to a `QThread` worker with real stage progress and cooperative cancellation.
- Added per-Mark/per-layer ROI source tracking (`recipe`, `manual`, or unset), source display, and explicit recipe-ROI confirmation before manual calculations.
- Made full-image auto recognition explicitly ignore every stored ROI and clear stale automatic candidates when switching workflows.
- Renamed the ROI fitting option to clarify that automatic model selection is not the full-image auto-recognition workflow.
- Added controls to clear the current ROI or all remaining recipe ROIs.
- Combined the web V1.5.3 display/export improvements with the main branch measurement service, non-square-pixel conversion, golden tests, and automatic-search truncation diagnostics.
- Preserved native image intensity ranges when display enhancement is off and retained failed batch runs in repeatability exports.

## V1.5.3

- Added Rx/Ry angle and material thickness compensation for overlay calculations.
- Added repeatability export rows with per-run results, mean, 3 sigma, and PV statistics.
- Added default export filenames based on the input image name plus `Misalignment_Result` and timestamp.
- Preserved Mark image aspect ratio in Excel exports.
- Added a UI display-enhancement switch, defaulting to raw grayscale display.
- Consolidated detection details, overlay results, and repeatability analysis into one result tab area.
- Added progress feedback and cancel support for Mark/batch overlay calculations.
- Restored algorithm-path traceability in the detection detail table and kept the status-bar popup.
- Removed duplicated top recipe/workflow hints and fixed recipe-name fallback after loading JSON.
- Changed final fit outlines and labels to high-contrast green instead of white.

## V1.5

- Added a service layer for manual ROI detection and Rz summary calculation, reducing UI coupling.
- Added golden-sample pytest coverage for manual ROI measurements, non-square pixel conversion, auto-detection reports, and Rz summaries.
- Added pytest project configuration and dependency constraints, plus a locked requirements file for reproducible production installs.
- Added a status-bar algorithm-path button that opens the actual measurement pipeline in a popup without occupying the main workspace.
- Added auto-detection truncation warnings for time, contour, and candidate limits.
- Improved physical unit conversion for non-square pixels, including radial statistics and rotated rectangle sizing.
- Removed redundant table section title labels to give more vertical space to measurement details and results.
- Added `start_overlay_measure.bat` as a clear Windows startup file while keeping `main.py` as the Python entry point.

## V1.4.2

- Replaced the repository root application with the package from `overlay_mark_measure_v1_4_2.zip`.
- Kept the previous repository version under `legacy/v1.0.5/`.
- Removed tracked `__pycache__` and PyCharm `.idea` files from the active project tree.
- Updated `README.md` to point to the current runnable entry point.
- Updated package metadata in `overlay_measure/__init__.py` to `1.4.2`.

## V1.0.5 / V1.1 lineage

- Previous repository root version before the V1.4.2 replacement.
- Retained as `legacy/v1.0.5/` for reference.
