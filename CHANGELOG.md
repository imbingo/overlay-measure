# Changelog

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
