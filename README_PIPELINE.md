# Automated GitHub Pages Pipeline

This package adds an end-to-end pipeline for:

1. running the simulation,
2. generating the figures,
3. copying them into a Pages-ready bundle,
4. and deploying the site automatically through GitHub Actions.

## Files included

- `.github/workflows/pages.yml`
- `docs/index.html`
- `docs/styles.css`
- `scripts/publish_pages.py`

## Expected simulation outputs

Your `run_simulation.py` should generate these files:

- `outputs/fig_binder_crossing.png`
- `outputs/fig_order_collapse.png`
- `outputs/fig_susceptibility_collapse.png`
- `outputs/fig_observables_grid.png`

Optional JSON files are also copied if present:
- `summary_results.json`
- `raw_results.json`
- `criticality_advanced_2d_results.json`

## Setup

1. Copy these files into your repository.
2. Replace the repository URL in `docs/index.html`.
3. Commit and push.
4. In GitHub, go to **Settings → Pages** and make sure Pages is set to use **GitHub Actions**.
5. Push to `main` or `master`.

## Notes

- If `requirements.txt` exists, the workflow installs from it.
- Otherwise it falls back to `numpy matplotlib`.
- The deploy step uses the official GitHub Pages actions.

## Local test

You can test the publish step locally with:

```bash
python scripts/publish_pages.py
```
