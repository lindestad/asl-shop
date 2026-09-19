# hand_features

Henter ut normaliserte håndlandmarks fra et bilde med MediaPipe Hand Landmarker,
til bruk i klassifisering av ASL-bokstaver.

## Oppsett

Last ned modellfilen og legg den i arbeidsmappen:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

## Bruk

```python
from hand_features import create_detector, extract_hands

detector = create_detector()
frame = cv2.flip(frame, 1)          # speilvend webkamerabildet først
hands = extract_hands(frame, detector)
```

`frame` må være et BGR-bilde fra OpenCV. Bildet må speilvendes før kallet,
ellers blir venstre/høyre-labelen motsatt.

## Returformat

`extract_hands` returnerer en liste med én ordbok per hånd (0–2 elementer):

| Nøkkel      | Type                | Beskrivelse                                  |
|-------------|---------------------|----------------------------------------------|
| `hand`      | `str`               | `"Left"` eller `"Right"`                     |
| `score`     | `float`             | Sikkerhet for sidigheten, 0–1                |
| `features`  | `np.ndarray (63,)`  | Normaliserte koordinater, se under           |
| `landmarks` | liste med 21 punkt  | Rådata fra MediaPipe (0–1), brukes til tegning |

## Featurevektoren

63 tall: 21 ledd × (x, y, z), lagt etter hverandre:
`[x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]`

Bruk `features.reshape(21, 3)` for å få én rad per ledd.

Normalisering:
- Alle punkter er relative til håndleddet, så punkt 0 er alltid `(0, 0, 0)`.
- Skalert slik at avstanden fra håndledd til punkt 9 er 1.
- Venstre hånd er speilet i x, så alle hender ser ut som en høyrehånd.
- Rotasjon er **ikke** normalisert, siden orientering skiller bokstaver
  (f.eks. H/U, K/P, G/Q).
- `y` øker nedover i bildet, så fingre som peker opp har negativ `y`.

## Leddindekser

| Indeks | Del                                   |
|--------|---------------------------------------|
| 0      | Håndledd                              |
| 1–4    | Tommel (rot → tupp)                   |
| 5–8    | Pekefinger (knoke → tupp)             |
| 9–12   | Langfinger                            |
| 13–16  | Ringfinger                            |
| 17–20  | Lillefinger                           |
