#!/bin/sh
cd "$(dirname "$0")/.." && curl -LO https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
