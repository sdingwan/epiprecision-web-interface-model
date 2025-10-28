# EpiPrecision Web Interface

A React + Material UI single-page application for demonstrating the EpiPrecision imaging workflow (upload ➝ processing ➝ clinician review). The project targets GitHub Pages and a custom domain at `https://demo.epiprecision.tech`.

- Upload MRI/EEG/PET study files, simulate AI processing, and organise results into RSN / Noise / SOZ groupings.
- Leverages React Router for multi-step navigation and a shared context provider for file state.
- Dark-theme UX with clinician-focused touches and export-ready PDF summaries.

---

## Prerequisites
- **Node.js 18.x or 20.x** (Create React App 5 runs best on the current LTS releases).
- **npm 9+** (bundled with Node). Yarn also works, but the scripts below assume npm.
- macOS, Windows, or Linux with a modern browser for local testing.

Verify your toolchain:
```bash
node -v
npm -v
```

---

## Quick Start (Local Development)
1. Install dependencies:
   ```bash
   npm install
   ```
2. Start the development server:
   ```bash
   npm start
   ```
3. Open the app at `http://localhost:3000`. The dev server enables hot reloading, so edits under `src/` reload automatically.

> **Tip**: The landing page checks `localStorage` for login state. Use the mock login form to seed credentials when testing the authenticated flow.

---

## Building for Production
Generate an optimised bundle under `build/`:
```bash
npm run build
```
The build output is what gets published to GitHub Pages. A successful build injects hashed asset names (e.g., `static/js/main.<hash>.js`) and respects the `homepage` setting in `package.json`, so paths remain correct for the custom domain.

---

## Deploying to GitHub Pages (demo.epiprecision.tech)
Deployment is handled by the [`gh-pages`](https://github.com/tschaub/gh-pages) CLI listed in `devDependencies`.

1. Ensure `public/CNAME` contains `demo.epiprecision.tech`. The deploy script copies it into the published branch so GitHub keeps the domain binding.
2. Build and deploy with one command:
   ```bash
   npm run deploy
   ```
   Behind the scenes this runs `npm run build` and pushes the `build/` directory to the `gh-pages` branch.
3. Wait 30–60 seconds for GitHub Pages to refresh, then load `https://demo.epiprecision.tech`. Use a hard refresh (⌘⇧R / Ctrl+F5) if you still see an old bundle.

> **Why the black screen happens**: when the deployed HTML references `/epiprecision-web-interface/static/...`, the assets 404 on the custom domain. Rebuilding after updating `homepage` to `https://demo.epiprecision.tech` fixes the paths. Always deploy through `npm run deploy` so the generated `index.html` stays in sync.

---

## Available npm Scripts
| Script | Purpose |
| --- | --- |
| `npm start` | Start CRA dev server at `http://localhost:3000`. |
| `npm test` | Run Jest/Testing Library suites in watch mode. |
| `npm run build` | Produce an optimised production bundle in `build/`. |
| `npm run deploy` | Build and push `build/` to the `gh-pages` branch. |
| `npm run eject` | Permanently eject CRA configuration (irreversible). |

---

## Python Analysis Pipeline

The React application calls a Python helper pipeline (under `final/`) to drive the
CNN + Knowledge Integrator workflow when a case is uploaded.

### Install pipeline dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> TensorFlow 2.20.0 is required and referenced in `requirements.txt`. The pinned
> wheel works on Apple Silicon and x86 macOS 12+. Adjust the version if you are
> targeting a different platform.

### Running the pipeline manually

```bash
source venv/bin/activate
export PYTHON_PATH="$PWD/venv/bin/python"   # used by src/setupProxy.js during npm start
python final/pipeline.py
```

Inputs & outputs:

| Item | Location |
| --- | --- |
| CNN SavedModel | `final/CNN_modelTrainedPCH/` (ignored in git – store separately) |
| Subject report assets | Uploaded through the UI ➝ staged under `final/uploads/<case>-<timestamp>/` |
| Workspace `.mat` | `Workspace-<CASE>IDV4.mat` in repo root (used automatically if not in the upload) |
| Latest results | `final/pipeline_results.{csv,json}`, `final/predictions_{DL/KL}.csv`, `final/analysis_summary.json` |

All generated artefacts are ignored by Git.

---

## Preparing the repository for GitHub

1. **Clean transient files (optional but recommended)**
   ```bash
   rm -rf venv build final/uploads
   rm -f final/pipeline_results.* final/predictions_*.csv final/analysis_summary.json
   ```
   The `.gitignore` already excludes these paths, so deleting them is only needed
   if you want a pristine working tree.
   Also keep the CNN SavedModel (`final/CNN_modelTrainedPCH/`) and any
   `Workspace-*.mat` files outside of git or host them in object storage / LFS.
2. **Commit your changes**
   ```bash
   git status
   git add .
   git commit -m "Prepare project for GitHub"
   ```
3. **Create the remote repository** on GitHub and connect:
   ```bash
   git remote add origin git@github.com:<your-org>/epiprecision-web-interface.git
   git push -u origin main
   ```
4. Optional: run `npm run build` and, if desired, execute the Python pipeline to
   verify the environment before pushing.

---

## Project Structure
```
src/
├── App.js                  # Router + shared FileProvider context
├── index.js                # React root rendering (StrictMode)
├── index.css               # Global dark-theme styles
├── theme.js                # Custom MUI theme overrides
└── components/
    ├── Navbar.js           # App navigation + mock auth menu
    ├── LandingPage.js      # Data type selection & login gateway
    ├── UploadPage.js       # File ingestion and validation
    ├── ProcessingPage.js   # Simulated AI/EPIK processing steps
    ├── ResultsPage.js      # AI classification review experience
    └── ICReferenceTable.js # Reference data for IC types

public/
├── index.html              # CRA template (injects root div)
├── CNAME                   # Custom domain for GitHub Pages
├── manifest.json           # PWA metadata stub
└── AIHeatmap.png           # Static asset used in results
```

---

## Troubleshooting
- **Black screen on GitHub Pages**: Assets are missing. Run `npm run deploy` so `index.html` ships with `/static/...` asset URLs and confirm the latest bundle exists in the `gh-pages` branch.
- **404s after deploy**: In GitHub Pages settings, ensure the source branch is `gh-pages` and that `public/CNAME` was published. Clear browser cache or open DevTools → Network → “Disable cache” to force reload.
- **Outdated dependencies**: If installs fail, delete `node_modules` and `package-lock.json`, then rerun `npm install`. CRA 5 expects Node 14–20; newer majors may require `--legacy-peer-deps`.
- **Local login loop**: Clear `localStorage` keys `userLoggedIn`, `userEmail`, `userName` from your browser to reset the mock auth state.

---

## Future Enhancements
- Wire upload + processing flows to real backend endpoints.
- Replace mocked AI outputs with live classification results and PDFs.
- Integrate SSO or production-ready authentication.
- Add automated tests covering router flows and context behaviour.

For questions or deployment help, reach out to the EpiPrecision web team.
