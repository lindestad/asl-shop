# ASL Shop

En Electron-app for gjenkjenning av ASL-bokstaver fra live kamera eller ett bilde.
Appen har to visninger: **Live kamera** og **Bilde**. Bilder og kamerarammer analyseres
lokalt av en Python-prosess med MediaPipe Hand Landmarker.

**Status for bokstavmodellen:** Hånddeteksjon fungerer, men en trent bokstavklassifikator
er ennå ikke koblet til. Backend returnerer `prediction: null`, og grensesnittet viser
ingen bokstav før en faktisk modell gir et resultat. ASL-datasettet er ment for det
videre modellarbeidet.

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
ASL-datasettet trengs ikke for å kjøre appen. Last det ned separat når dere skal
trene eller teste en bokstavmodell:

```sh
uv run python scripts/download_dataset.py
```

## Struktur

- `frontend/`: React-visningene for kamera og bilde.
- `electron/main.mjs`: vindu, lokale kamera­tillatelser og validerte IPC-kall.
- `electron/preload.cjs`: begrenset API for bildevalg og gjenkjenning.
- `electron/recognition.mjs`: styrer den lokale Python-prosessen.
- `scripts/recognition_worker.py`: hånddeteksjon og grensesnitt for fremtidig bokstavmodell.
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
Python, MediaPipe og håndmodellen følger med i den pakkede appen; ASL-datasettet
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
