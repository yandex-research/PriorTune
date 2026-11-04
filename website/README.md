# PriorTune research website

An interactive website for PriorTune: synthetic tasks select reusable sets of
hyperparameter configurations for downstream tabular learning. The page and its
figure data run in the browser; serving the site does not launch experiments.

Public website: https://yandex-research.github.io/PriorTune/

Website source: the `website/` folder of https://github.com/yandex-research/PriorTune.
Every push to `main` that changes it redeploys the site (`.github/workflows/pages.yml`).

The repository contains one website. The main URL shows the latest wording;
no version selector or URL parameters are needed.

## Run locally

Python 3.9 or newer is sufficient. No Node.js, build dependency, or API key is
required.

```sh
python3 scripts/serve.py
```

Open <http://127.0.0.1:4175/>. The local preview refreshes when public source
files change. Keep the command running; press Ctrl+C to stop it. Use
`--no-live-reload` to disable refresh or `--port 8080` to choose another port.

## Build the deployable website

```sh
python3 scripts/package_site.py
python3 scripts/serve.py --dist --port 4176
```

Open <http://127.0.0.1:4176/>. Only `index.html`, `app.js`, `styles.css`, the
`figures-ba`, `figures-hp`, and `demo` JavaScript/CSS modules, `assets/`, and
`data/` enter `dist/`. A SHA-256 `manifest.json` records the
packaged content. Local preview code is not included in the deployed HTML.

Raw source artifacts, local QA screenshots, environments, and scripts are
excluded from the bundle. Only place intended public content in `assets/`
and `data/`.

To host elsewhere, upload the contents of `dist/` to any static hosting service.
Relative asset paths work at the domain root or in a subdirectory. No backend,
API keys, or version parameters are required. Update the `og:url` and `og:image`
metadata in `index.html` if you want share previews to use the new domain.

## Reproduce the figure data

The committed `data/` directory is sufficient to serve and deploy the website.
To rebuild it from the public research artifacts:

```sh
python3 scripts/fetch_research_sources.py
python3 scripts/build_research_data.py
```

The fetcher restores compact inputs from the GitHub and Hugging Face revisions
pinned in `data/research.json`, verifying their SHA-256 hashes. It saves inputs
under ignored `source-artifacts/`; no credentials, raw datasets, or model fits
are needed. Original numeric CSVs are available in `data/downloads/`, and the
upstream Apache 2.0 notice is retained in `data/SOURCE-LICENSE.txt`.

The website's data includes recorded results and separately identified derived
plots. Random-search, budget and heatmap views show all three model families
together. Heatmaps use each K's measured grid and a labelled per-panel color
scale from 50% to that panel's maximum; a power of 1.7 leaves small gains pale.
Budget figures include raw evaluated cells and the published best-in-bin
envelopes. Hyperparameter curves reproduce the paper's reflected Gaussian KDE
in sampling-CDF coordinates. Their vertical axis is selected density relative
to the original prior, with the prior fixed at 1×. The three slots offer all 36 released
parameters. Categorical choices use probability bars with exact selected counts.
Conditional TabPrep choices use the 37 configurations with TabPrep enabled.
Dropout displays its mass at zero separately from the positive-value density.
The download retains original bins, selected values, and estimator metadata. Protocol distinctions and source hashes
are recorded with every figure export.

The opening illustration is separate from those research results. Both panels
share one deterministic synthetic surface; the random panel gradually adds
candidates while the other displays 25 fixed diagram coordinates. Its linked
win-rate curve is a hand-designed schematic on a 0.3–0.7 axis, not a score
computed from the points, a learned portfolio, or a benchmark result. The illustration makes no
claim of noiseless validation for PriorTune.

## Deploy to GitHub Pages

1. Copy this directory's contents into a GitHub repository root, including
   `.github/` and `.gitignore`.
2. Push to the `main` branch.
3. In **Settings → Pages → Build and deployment**, choose **GitHub Actions**.
4. Run **Deploy PriorTune to GitHub Pages** in Actions, or push another commit.

The workflow builds and uploads only `dist/`. Application resources use
relative URLs, so a repository Pages URL works without a custom domain.
The local workflow does not create a repository or publish a website.

See the [GitHub Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
for hosting configuration.

## Browser checks

With the local server running, optionally install Playwright in an isolated
development environment:

```sh
python3 -m venv .qa/venv
.qa/venv/bin/python -m pip install playwright
.qa/venv/bin/python -m playwright install chromium
.qa/venv/bin/python scripts/check_site.py
.qa/venv/bin/python scripts/check_figures.py
.qa/venv/bin/python scripts/check_experience.py
```

On macOS, the checker uses installed Google Chrome if available; otherwise it
uses Playwright's Chromium. Use `--browser /path/to/chrome` for another local
installation, or `--url https://example.github.io/priortune/` to check a deployed
copy. The checker saves desktop/mobile screenshots and a JSON report in ignored
`.qa/`. These development tools are not required to serve or build the site.
The figure checker verifies plotted measurements against `data/research.json`,
all three models and portfolio sizes, measured heatmap coordinates and legends,
immediate tile details, all 36 parameter choices, exact categorical masses,
continuous density curves, the separate dropout atom, slot shuffling, data
exports, keyboard controls, label collisions, and responsive figure layouts. It saves
separate figure screenshots in `.qa/figures/` for visual review.
The experience checker verifies the illustrative demo's fixed surfaces,
schematic curve and its labeled units, playback, reduced-motion behavior,
and mobile tile and categorical-bar details. The site checker also verifies the
canonical homepage, old-link cleanup, and desktop/mobile layouts. After packaging,
run `.qa/venv/bin/python scripts/check_site.py --dist` to verify the standalone
bundle under a temporary repository-style URL.

## Files

```text
index.html          Page content and accessible controls
app.js              Figure integration and shared interactions
figures-ba.js/.css   BeyondArena plots and responsive layouts
figures-hp.js/.css   Three hyperparameter distribution slots
demo.js/.css         Two illustrative surfaces and schematic curve
styles.css          Responsive styles
assets/             Public visual assets
data/               Public research data and provenance
scripts/            Data preparation, local preview, and packaging
.github/workflows/  GitHub Pages deployment
```
