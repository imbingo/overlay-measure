**Design QA**

- Source visual truth: `C:\Users\bingo\.codex\generated_images\019ee4fa-2737-79e3-9919-bd2fffa16076\exec-56929ba4-6da2-4064-bf3b-26284ac3d003.png`
- Implementation screenshot: `D:\CodingTool\overlay_measure\source\artifacts\v1_6_0_production.png`
- Viewport: 1500 x 920.
- State: V1.6.0 frameless workspace in the default production mode before loading images or a recipe.

**Comparison Evidence**

- The V1.5.7 recipe switcher hierarchy remains intact while a compact green production-mode selector is added beside the version badge.
- Production mode disables the ROI and algorithm tabs without hiding their numbered workflow position, so operators can still understand the process while protected settings remain inaccessible.
- Recipe saving is visibly disabled in production mode; image loading, recipe selection, calculation, export, and operator entry remain available.
- The restrained white/neutral-blue visual system and the persistent bottom task/progress/algorithm-path row remain stable at 1500 x 920.
- Pass/fail colors remain reserved for measurement semantics; the production-mode green is limited to the access-state selector.

**Interaction Evidence**

- The application starts in production mode; switching to engineering opens a masked password prompt.
- Wrong credentials keep the application in production mode, while the default initial password unlocks ROI, algorithm, diagnostic, and recipe-edit controls.
- Calculation preflight rejects unvalidated, unsealed, modified, or incomplete production recipes before starting a worker thread.
- Calculation disables access-mode switching until the background task completes.
- Automated regression coverage validates access control, mode locking, recipe integrity, production preflight, and the existing recipe-library behaviors.

**Findings**

- No actionable P0, P1, or P2 visual differences remain.
- The production-mode control uses a native combo arrow so the access transition remains discoverable without adding another title-bar button.

final result: passed
