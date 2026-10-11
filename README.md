# ASL Shop

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
Datasett, landemerker og trente modeller ligger i den Git-ignorerte `data/`-mappen.
Ved første oppsett må bokstavmodellen trenes før lokal bokstavgjenkjenning og pakking:

```sh
uv run python scripts/download_dataset.py
uv run python load_data.py
uv run python model/knn.py
```

`load_data.py` bruker samme normalisering som appen og skriver 63 koordinater per hånd
til `data/extracted_landmarks.csv`. Uten valg behandles alle treningsbildene. For en
raskere prøve, bruk `uv run python load_data.py --samples-per-class 120 --seed 42`.
Antallet rader kan avvike fra antallet bilder, fordi noen bilder ikke gir hånddeteksjon.

`model/knn.py` bruker et fast 70/30-utvalg med seed 42, skriver antall trenings- og
evalueringsrader og måler nøyaktighet på de holdte radene før `data/knn.pkl` lagres.
Det tidligere fullstendige CSV-uttrekket hadde 63 932 rader, hvorav 44 752 ble brukt
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
