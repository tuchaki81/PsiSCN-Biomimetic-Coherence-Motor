# GitHub Pages Adaptation

This folder contains a static site template for publishing the coherence-criticality figures on GitHub Pages.

## Included files

- `index.html`
- `styles.css`

## Expected image paths

By default, `index.html` looks for:

- `outputs/fig_binder_crossing.png`
- `outputs/fig_observables_grid.png`
- `outputs/fig_order_collapse.png`
- `outputs/fig_susceptibility_collapse.png`

## How to use

1. Create a `docs/` folder in your GitHub repository.
2. Copy `index.html` and `styles.css` into `docs/`.
3. Copy the figures into `docs/outputs/` or update the paths in `index.html`.
4. Go to **Settings → Pages** in GitHub.
5. Set source to your main branch and `/docs`.
6. Save and wait for deployment.

## Customization

- Replace `YOUR-USERNAME/YOUR-REPO` in the repository button.
- Edit the descriptive text if you want the page to reflect a specific paper or preprint.
