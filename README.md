# ASL Shop

En Electron-app for gjenkjenning av ASL-bokstaver fra live kamera eller ett bilde.
Appen har to visninger: **Live kamera** og **Bilde**. Bilder og kamerarammer analyseres
lokalt av en Python-prosess med MediaPipe Hand Landmarker.

**Bokstavmodell:** En KNN-modell er trent på 2 400 bilder der MediaPipe fant en hånd.
Den bruker de samme 63 normaliserte håndkoordinatene som kamera- og bildeanalysen.
Modellen ligger i `model/asl_knn.npz` og er inkludert i den pakkede appen. Visningen
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
ASL-datasettet trengs ikke for å kjøre appen, siden den trente modellen følger med.
Last ned datasettet separat for å trene modellen på nytt eller kjøre integrasjonstesten:

```sh
uv run python scripts/download_dataset.py
uv run python model/knn.py --samples-per-class 120
```

Treningsskriptet henter landemerker med samme kode som appen og lagrer en ny
`model/asl_knn.npz`. Det faste utvalget styres av `--seed 42`. På de separate
testbildene ble 13 av 26 bokstavbilder registrert som hender, og alle de 13 ble
klassifisert riktig. Dette er et lite testsett og sier lite om hvor godt modellen
virker på nye personer, bakgrunner og lysforhold.

## Struktur

- `frontend/`: React-visningene for kamera og bilde.
- `electron/main.mjs`: vindu, lokale kamera­tillatelser og validerte IPC-kall.
- `electron/preload.cjs`: begrenset API for bildevalg og gjenkjenning.
- `electron/recognition.mjs`: styrer den lokale Python-prosessen.
- `scripts/recognition_worker.py`: hånddeteksjon og bokstavklassifisering.
- `model/knn.py`: trening og enkel evaluering av bokstavmodellen.
- `handTracker/`: MediaPipe-hjelpere som brukes av worker-prosessen.

Renderer-prosessen er sandboxed, har context isolation og ingen Node-integrasjon.
Den pakkede frontend-en lastes gjennom en begrenset `app://`-protokoll.

## Sjekk og bygg

```sh
npm run build
npm run lint
npm test
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
Last ned datasettet før du kjører den testen.
