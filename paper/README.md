IEEE Access preliminary research paper

Compile bare_ieee_access.tex twice with pdfLaTeX from this directory. The archive includes all source, references, images, tables, class/font dependencies and measured result provenance. Author details are blank as requested.

This version reports a frozen early MobileNet global checkpoint on 1,259 development images only (MAE8.285 months), six results plots, five genuine Grad-CAM panels, fourteen results tables and one literature table, plus forty references. Published comparisons use different protocols; NR is explicitly unverified/unreported, never an estimate. Final-holdout SOTA is not established.

A separate larger ResNet50 run (640px global/384px crops, batch4 effective8) is now training in runs/upgraded. Larger model resource profiles are actual optimizer-update measurements, not accuracy results. Saved MobileNet checkpoints are preserved. The same paper source will be updated after final evaluation.

Local MiKTeX compiled this project successfully; the built-in editor compiler reports a platform-directory error. Dataset images, caches and trained weights are excluded from this archive. See results/ for code and provenance.
