# asl-shop

Machine vision project IKT213: Sign language image editor

## Course project (text from Canvas)

Project goals Design and implement a photo editing desktop application or a web application or a mobile application that has the following minimum functionalities:

1. File menu with the following options: New, Open, Save, Save as, Properties, Quit

2. Clipboard menu with the following options: Copy, Paste, Cut

3. Image menu with the following options: Select --> rectangular selection, free-form selection (Lasso), Polygon selection; Crop, Resize, Rotate --> rotate right 90 degrees, Rotate Left 90 degrees, Flip vertical, Flip horizontal;

4. Tools menu with following options: Zoom (Zoom In, Zoom Out), Erase, Color Picker, Paint brushes (with different textures/patterns), Text box, Filters --> Gaussian filter, Sobel filter, Binary filter; Histogram thresholding

5. Shapes menu with the following options: List of Shapes, Outline color, Fill color

6. Colors menu with the following options: Color pallet, Size of brush

Optional feature to design:

1. Layer menu with following option: New Layer, Load layer, Edit layer, Select layer, Delete layer, Rename layer

2. Snapchat/Zoom filters option :)

---

### Additions beyond mvp

- Sign language detection (still images)

### Oppsett

#### Electron-appen

Kjør fra repo-roten på macOS eller Linux:

```sh
./setup.sh
source data/setup-env.sh
npm run dev
```

Grensesnittet bruker React og TypeScript, Vite 8 med Rolldown/Oxc,
Oxlint og Oxfmt. `npm run dev` starter Vite og Electron sammen.
`npm run dev:web` åpner kun nettleserversjonen; lokal hånddeteksjon og
eksport til en mappe krever Electron.

`setup.sh` installerer uv, Python og avhengighetene, Node.js 24 via
[fnm](https://github.com/Schniz/fnm) hvis Node.js 22.12+ eller npm mangler,
frontend-pakkene fra låsefilen, Electron og Chromium for testene. Det henter
modellen og datasettet og bygger Python-adapteren for pakking med PyInstaller.
En eksisterende kompatibel Node/npm-installasjon gjenbrukes. `data/setup-env.sh`
gjør de installerte verktøyene tilgjengelige i terminalen uten å endre
shellprofilen. npm-avhengighetene installeres på nytt fra låsefilen ved hver kjøring.

- **Image editor:** åpne eller dra inn bilder, tegn, legg til former og tekst,
  beskjær, endre størrelse, roter, speilvend, bruk filtre og eksporter PNG.
- **Live camera:** spor opptil åtte hender, vis FPS og ta et bilde som åpnes i editoren.
  Kameraet stoppes når du forlater kameravisningen.
- **Batch studio:** analyser opptil 100 bilder og eksporter PNG-bilder med
  landmarks og en `recognition.json`-rapport til en ny undermappe.

Electron starter en vedvarende Python-prosess via
`scripts/recognition_worker.py`. Den gjenbruker `handTracker/hand_features.py`
og sender landmarks, 63 features og håndkonfidens gjennom en begrenset
preload-API. Det brukes ingen HTTP-server for Python.
Kameraet bruker MediaPipe VIDEO-modus med kapasitet tilpasset antallet synlige
hender. En separat skanning hvert 250 ms finner nye hender, opptil åtte totalt.
Bare ett kamerabilde analyseres om gangen; gamle bilder legges ikke i kø.
FPS-visningen måler fullførte analyser, inkludert bildeoverføring.
Bokstavklassifisering er fortsatt avhengig av modellen i
[issue #7](https://github.com/lindestad/asl-shop/issues/7): `prediction` er
`null` inntil den modellen kobles til. Grensesnittet viser dette eksplisitt.

```sh
npm run build       # TypeScript-sjekk og frontend-bygg
npm run lint        # Oxlint
npm run format      # Oxfmt
npm test            # Editor-, kamera- og Electron-integrasjonstester
npm run package     # Utpakket Electron-app i release/
npm run dist        # AppImage, dmg eller Windows-installasjon
```

Pakkingen bygger først Python-adapteren med PyInstaller og inkluderer
Python, avhengigheter og MediaPipe-modellen i appen. Sluttbrukeren trenger
ikke Python eller uv. Treningsdatasettet pakkes ikke med.
Bygg på operativsystemet og arkitekturen du skal distribuere til.
Linux-bygget er verifisert lokalt; øvrige plattformer må testes på sine maskiner.
Electron-testen bruker `data/asl_alphabet_train/asl_alphabet_train/A/A1000.jpg`
fra datasettet som `setup.sh` henter.

Test samme integrasjon mot en pakket Linux-app:

```sh
ASL_SHOP_TEST_APP=release/linux-unpacked/asl-shop npm test -- --project=desktop
```

#### Python og datasett

Kjør fra repo-roten på macOS eller Linux:

```sh
./setup.sh
```

Skriptet setter opp hele prosjektet som beskrevet over, inkludert
Python 3.12 og prosjektavhengighetene i `.venv`, MediaPipe-modellen og
ASL Alphabet-datasettet i `data/`.
Det krever `curl` eller `wget`, men ikke en eksisterende Python-installasjon eller
Kaggle-konto. Nedlastede data og modellfiler ignoreres av Git.
Kjør samme kommando igjen hvis oppsettet avbrytes; ferdige nedlastinger hoppes over.

For å bare installere avhengighetene manuelt: `uv sync --python 3.12`.

Avhengigheter vedlikeholdes i `pyproject.toml`. Etter endringer, oppdater
låsefilen og generer `requirements.txt` på nytt:

```sh
uv lock
uv export --format requirements-txt --no-hashes --output-file requirements.txt
```

Alternativt kan avhengighetene installeres med `python -m pip install -r requirements.txt`.

Last ned [Hand Landmarker-modellen](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
og lagre den som `data/hand_landmarker.task` i repo-roten. Modellfilen skal ikke sjekkes inn i Git.
Modellbanen er uavhengig av hvilken mappe programmet startes fra.

Start webkamerademoen fra repo-roten med `uv run python handTracker/hand_features.py`.

### ASL Alphabet-datasett

Last ned og pakk ut [ASL Alphabet fra Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet):

```sh
uv run python scripts/download_dataset.py
```

Skriptet bruker bare Pythons standardbibliotek og krever ikke Kaggle-konto.
Det lagrer arkivet (ca. 1 GB) og bildene i `data/`, med mappestrukturen fra
Kaggle: `data/asl_alphabet_train/asl_alphabet_train/` og
`data/asl_alphabet_test/asl_alphabet_test/`. Filene ignoreres av Git.
Senere kjøringer hopper over et ferdig installert datasett. Hvis nedlasting
eller utpakking avbrytes, kjør samme kommando igjen.
