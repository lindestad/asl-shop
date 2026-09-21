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

Installer Python 3.10 eller nyere og uv. Kjør fra repo-roten:

```sh
uv sync
```

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
