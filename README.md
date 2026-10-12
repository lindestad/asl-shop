# ASL Shop

An optional [neural letter and temporal fingerspelling pipeline](docs/neural-recognition.md)
adds separate current-letter and spelled-text outputs, explicit collection,
signer-disjoint training/evaluation, and OpenFS import support. Trained checkpoints
are local data artifacts; the existing KNN remains a static-letter fallback when
the neural head is unavailable. Optional external OpenFS refinement runs on Finish.

The [measured FSboard results and reproduction commands](docs/training-results-2026-10-12.md)
compare the legacy-data KNN, trained handshape/causal models and released OpenFS
checkpoint on held-out signers. Static scores use model-derived labels; live
held-letter accuracy and idle rejection still need human-labelled evaluation.

The official OpenFS FSboard downloads can be resumed and extracted with:

```powershell
uv run --with gdown python scripts/download_openfs.py --directory 'C:\Users\danie\dev\asl-shop\data\external\openfs' --extract
```

The two archives are `fingerspelling_fsboard_raw.zip` (re-extracted landmarks,
not video) and `fingerspelling_fsboard_proc_clean.zip` (model-derived cleaned frame
annotations). Existing ZIPs are reused and retained. Add `--verify` for a full CRC
scan. On macOS/Linux, `./setup.sh --neural` installs neural dependencies;
`./setup.sh --extract-openfs --openfs-dir /path/to/openfs` also prepares the data.
The optional OpenFS download is approximately 12.7 GB compressed. Dataset and
upstream code terms remain those of [OpenFS](https://github.com/AIRC-KETI/OpenFS).

Native Windows setup is available with uv and Node.js 22.12+ installed:

```powershell
.\setup.ps1 -Neural
.\setup.ps1 -ExtractOpenFS -OpenFSDir 'C:\Users\danie\dev\asl-shop\data\external\openfs'
```

The default `setup.ps1` skips OpenFS and neural dependencies. `-OpenFS` resumes or
reuses the archives without extraction; `-ExtractOpenFS` also extracts them.
Add `-Plan` to inspect commands without modifying anything. Launch trained
checkpoints with `scripts/start-neural.ps1`; see
the [neural guide](docs/neural-recognition.md#run-the-app) for external model paths.

En Electron-app for gjenkjenning av ASL-bokstaver fra live kamera eller ett bilde.
Appen har to visninger: **Live kamera** og **Bilde**. Bilder og kamerarammer analyseres
lokalt av en Python-prosess med MediaPipe Hand Landmarker.

**Bokstavmodell:** En KNN-modell trenes på landemerker fra ASL Alphabet-datasettet.
Den bruker de samme 63 normaliserte håndkoordinatene som kamera- og bildeanalysen.
Den lokale modellen lagres som `data/knn.pkl` og inkluderes ved pakking. Visningen
oppgir hvor mange av de fem nærmeste treningseksemplene som støtter bokstaven;
det er ikke en kalibrert sannsynlighet. Ved færre enn tre stemmer vises ingen bokstav.
J og Z er bevegelsestegn i ASL, så en modell som bare ser ett bilde om gangen kan
ikke gjenkjenne hele bevegelsen.

## Kom i gang

På macOS eller Linux, fra repo-roten:

```sh
./setup.sh
source data/setup-env.sh
npm run dev
```

`npm run dev` starter Vite og Electron. Frontend-kode oppdateres av Vite; endringer i
`electron/` starter Electron-prosessen på nytt. `npm run dev:web` viser bare
grensesnittet i nettleser, uten lokal gjenkjenning.

`setup.sh` installerer prosjektets Node- og Python-avhengigheter og MediaPipe-modellen.
Det ferdige landemerkeuttrekket følger med som `data/extracted_landmarks.csv`, slik at
hver maskin slipper den tidkrevende bildeanalysen. Originalbilder og trente modeller
ligger lokalt i den Git-ignorerte `data/`-mappen. Ved første oppsett trenger du bare
å trene bokstavmodellen før lokal bokstavgjenkjenning og pakking:

```sh
uv run python model/knn.py
```

Last bare ned originalbildene hvis du vil kjøre desktop-testene eller generere et
nytt landemerkeuttrekk fra ASL Alphabet-datasettet:

```sh
uv run python scripts/download_dataset.py
uv run python load_data.py
uv run python model/knn.py
```

`load_data.py` erstatter det delte CSV-uttrekket. Bruk `--output data/my_landmarks.csv`
og `model/knn.py --input data/my_landmarks.csv` for et forsøk som ikke endrer den
versjonerte filen.

`load_data.py` bruker samme normalisering som appen og skriver 63 koordinater per hånd
til `data/extracted_landmarks.csv`. Uten valg behandles alle treningsbildene. For en
raskere prøve, bruk `uv run python load_data.py --samples-per-class 120 --seed 42`.
Uttrekket bruker opptil fire CPU-prosesser som standard. `--workers 2` bruker færre
ressurser på en laptop; `--workers 1` kjører sekvensielt. Hver prosess gjenbruker sin
egen MediaPipe-detektor. Radrekkefølgen og seed er de samme uansett antall prosesser.
Fremdriften viser bilder per sekund og anslått tid igjen. Et eksisterende CSV-uttrekk
erstattes først når hele jobben er ferdig, slik at avbrudd ikke ødelegger det.
Antallet rader kan avvike fra antallet bilder, fordi noen bilder ikke gir hånddeteksjon.

`model/knn.py` bruker et fast 70/30-utvalg med seed 42, skriver antall trenings- og
evalueringsrader og måler nøyaktighet på de holdte radene før `data/knn.pkl` lagres.
Det medfølgende CSV-uttrekket har 63 932 rader, hvorav 44 752 blir brukt
til trening. Nye uttrekk kan gi andre antall. Evalueringen bruker samme bildedatasett
som treningen og måler ikke generalisering til nye personer, bakgrunner og lysforhold.
Begge skriptene har `--input` og `--output` for andre lokale filbaner. Standardbanene
er relative til repo-roten, uavhengig av terminalens arbeidsmappe.

Etter trening trengs bare `data/hand_landmarker.task` og `data/knn.pkl` for å kjøre
appen. Uten bokstavmodellen vises håndlandemerker, men ingen bokstav. Pakkede apper
inkluderer modellene og krever verken treningsdatasettet eller en Python-installasjon.
Last bare pickle-modeller som du selv har trent eller stoler på.

## Struktur

- `frontend/`: React-visningene for kamera og bilde.
- `electron/main.mjs`: vindu, lokale kamera­tillatelser og validerte IPC-kall.
- `electron/preload.cjs`: begrenset API for bildevalg og gjenkjenning.
- `electron/recognition.mjs`: styrer den lokale Python-prosessen.
- `scripts/recognition_worker.py`: hånddeteksjon og bokstavklassifisering.
- `model/knn.py`: trening og enkel evaluering av bokstavmodellen.
- `load_data.py`: uttrekk av landemerker til lokal CSV.
- `handTracker/`: MediaPipe-hjelpere som brukes av worker-prosessen.

Renderer-prosessen er sandboxed, har context isolation og ingen Node-integrasjon.
Den pakkede frontend-en lastes gjennom en begrenset `app://`-protokoll.

## Sjekk og bygg

```sh
npm run build
npm run lint
npm run format:check
npm test
uv run python -m unittest discover -s tests -p "test_*.py"
npm run package
```

`npm run package` bygger frontend og Python-backend, og lager en utpakket app i
`release/`. `npm run dist` lager installasjonsformatet for plattformen som kjøres.
Python, MediaPipe, håndmodellen og bokstavmodellen følger med i den pakkede appen; ASL-datasettet
gjør det ikke. Bygg på operativsystemet og arkitekturen du vil distribuere til.

Pakkingen bruker `electron-builder`. [Electrons pakkeveiledning](https://www.electronjs.org/docs/latest/tutorial/tutorial-packaging)
anbefaler Forge, mens [oversikten over alternativer](https://www.electronjs.org/docs/latest/tutorial/forge-overview)
omtaler `electron-builder` som et tredjepartsvalg uten offisiell støtte fra
Electron-prosjektet. Selve Electron-integrasjonen følger
[sikkerhetsveiledningen](https://www.electronjs.org/docs/latest/tutorial/security/)
for preload, context isolation, sandboxing, IPC-validering, tillatelser og lokal protokoll.

Desktop-integrasjonstesten bruker
`data/asl_alphabet_train/asl_alphabet_train/A/A1000.jpg` fra det separate datasettet.
Last ned datasettet og tren modellen før du kjører desktop-testene. De sjekker også
kamerahastighet, nye hender, stopp, omstart og bytte av visning.
