# Assets Manifest

This document defines the expected artifacts, their locations, and naming conventions for the research report. All paths are relative to the `docs/` directory.

## Directory Structure

```
docs/
├── report.md              # Main report scaffold
├── research_summary.md    # Existing literature synthesis
├── assets_manifest.md     # This file
├── assets/
│   ├── figures/           # Static figures (PNG, SVG)
│   ├── tables/            # Generated tables (CSV, markdown)
│   ├── videos/            # Generated video samples (MP4, GIF)
│   └── raw/               # Raw data exports for reproducibility
```

## Asset Categories

### Figures

| Asset Path | Description | Format | Status |
|------------|-------------|--------|--------|
| `assets/figures/fig_id_generation_comparison.png` | Side-by-side ID generation: ground truth vs. model outputs | PNG | placeholder |
| `assets/figures/fig_ood_failures.png` | OOD failure cases with annotations | PNG | placeholder |
| `assets/figures/fig_latent_pca.png` | PCA projection of latent representations | PNG | placeholder |
| `assets/figures/fig_loss_curves.png` | Training loss curves for all model variants | PNG | placeholder |
| `assets/figures/fig_probe_accuracy.png` | Linear probe accuracy by layer | PNG | placeholder |
| `assets/figures/fig_architecture.png` | Model architecture diagram | PNG/SVG | placeholder |

### Videos

| Asset Path | Description | Format | Status |
|------------|-------------|--------|--------|
| `assets/videos/gif_id_samples.gif` | Short video samples of ID generation | GIF | placeholder |
| `assets/videos/gif_ood_breakdown.gif` | OOD failure mode demonstrations | GIF | placeholder |
| `assets/videos/gif_context_boundary.mp4` | Short-context boundary artefact examples | MP4 | placeholder |

### Tables

| Asset Path | Description | Format | Status |
|------------|-------------|--------|--------|
| `assets/tables/table_quantitative_results.csv` | Main quantitative metrics | CSV | placeholder |
| `assets/tables/table_hyperparameters.csv` | Training hyperparameters | CSV | placeholder |
| `assets/tables/table_probe_results.csv` | Linear probe accuracy by model/layer | CSV | placeholder |

## Naming Conventions

- Use lowercase with underscores: `fig_loss_curves.png`
- Prefix by figure/table type: `fig_`, `gif_`, `table_`
- Include descriptive suffix: `fig_id_generation_comparison.png`
- Version if needed: `fig_loss_curves_v1.png`

## Generation Commands

These commands (to be run from project root) will generate assets:

```bash
# Generate loss curves
python -m scripts.plot_loss --output docs/assets/figures/fig_loss_curves.png

# Generate PCA visualization
python -m scripts.plot_latent_pca --output docs/assets/figures/fig_latent_pca.png

# Generate comparison videos
python -m scripts.generate_samples --model short --output docs/assets/videos/

# Export quantitative results
python -m scripts.export_results --format csv --output docs/assets/tables/
```

## Placeholder Legend

| Symbol | Meaning |
|--------|---------|
| `TBD` | To be determined (numeric results) |
| `placeholder` | Asset not yet generated; placeholder in report |
| `N/A` | Not applicable |

## Post-Processing Checklist

When populating the report with actual assets:

- [ ] Verify all relative paths resolve correctly from `docs/report.md`
- [ ] Check figure resolutions (minimum 300 DPI for print, 72 DPI for web)
- [ ] Ensure video files are compressed (GIF < 10MB, MP4 < 50MB)
- [ ] Validate table CSV files parse correctly
- [ ] Add alt-text descriptions to figures in report
- [ ] Update this manifest with actual file sizes and checksums for reproducibility
