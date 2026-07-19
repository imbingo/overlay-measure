# Changelog

## V1.6.0

- Added production and engineering operation modes. The application starts in production mode; entering engineering mode requires a password, initially `admin123`, and authenticated engineers can change it from the product-information page.
- Locked ROI editing, algorithm parameters, diagnostic controls, product configuration, and recipe saving in production mode.
- Added SHA256 recipe sidecars and blocked production calculations when a recipe is unsealed, modified, unvalidated, or missing required production metadata.
- Added explicit `Pass`, `Fail`, `Invalid`, and `Error` result states so specification failures are no longer conflated with recognition failures or runtime exceptions.
- Added batch image pairing checks for missing layers, count mismatches, and accidental use of the same file as both upper and lower images.
- Added automatic measurement archives containing software version, operation mode, recipe hash, parameter snapshots, input-file hashes, per-run results, and a UI screenshot.
- Added rotating runtime logs, unfinished-task recovery, configurable calculation timeout, and cooperative cancellation.
- Added measurement ID, operation mode, recipe hash, and archive path to exported Excel traceability information.
- Added focused regression coverage for access control, recipe integrity, quality verdicts, batch pairing, traceability archives, and production preflight checks.

## V1.5.7

- Added a searchable quick recipe switcher in the frameless title bar, grouped by favorites, recent use, and all recipes.
- Added a managed local recipe library under the user's local application-data directory with validated, draft, and archived categories.
- Preserved direct JSON file loading with an explicit choice between one-time loading and importing a managed library copy.
- Added a recipe manager for filtering recipes, toggling favorites, opening the local library, and configuring an optional company shared recipe directory.
- Made recipe switching preserve imported single/batch images while replacing old ROIs and clearing stale measurement, auto-detection, and repeatability results after confirmation.
- Kept existing recipe JSON fields compatible and stored favorites, history, and shared-library settings in a separate library-state file.

## V1.5.6

- Refined the frameless window against the approved fused measurement-workstation mockup.
- Reduced the custom title bar to 46 px and limited it to product identity, version, recipe actions, and system window controls.
- Moved image import, display mode, reset, zoom, ROI analysis, overlay calculation, and export into a dedicated command bar.
- Added a direct zoom percentage selector while preserving wheel zoom, panning, and fit-to-window behavior.
- Rebalanced the image workspace and right configuration rail, reduced result-table minimum height, and retained a stable four-column result summary.
- Kept task state, current recipe, persistent progress, current stage, algorithm path, and cancellation visible in one non-jumping bottom status row.
- Removed remaining emoji button labels and aligned controls to a restrained neutral-blue industrial visual system.

## V1.5.5

- Reworked the desktop layout into a frameless, light industrial workspace based on the selected fused UI direction.
- Removed the redundant left workflow rail while keeping the numbered 1–5 configuration tabs on the right.
- Added a stable four-column result strip with responsive typography for compact window sizes.
- Consolidated task state, current recipe, live progress, current calculation stage, algorithm path, and cancellation in the bottom status bar.
- Fixed three-point circle creation so right/middle-button panning takes priority after two points without clearing the selected points.
- Ensured top-level single-image imports replace stale batch previews and switch the run back to single-measurement mode.
- Added a full measurement reset that clears imported images, batch lists, results, detections, automatic candidates, and every ROI while preserving reusable parameters.
- Added append-style batch import from multiple folders, direct folder import, optional recursive subfolder scanning, natural ordering, and duplicate-file skipping.
- Removed the redundant expected-repeat-count input; repeat counts now come directly from imported image lists.

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
