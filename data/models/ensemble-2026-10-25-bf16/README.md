# ens4 fingerspelling ensemble — bf16

Four members of the 2026-10-25 ens4 ensemble, converted from fp32 to bfloat16 so the
checkpoints can travel in plain git (largest file 52 MB, ~137 MB total).

## Provenance

Converted from the fp32 checkpoints of the same run (members unchanged: m1 epoch-62
3-layer, m3 epoch-82 6-layer, m4 6-layer soup e54e59, m5 9-layer soup e41e43e55):

```python
payload["state_dict"] = {k: (v.bfloat16() if v.is_floating_point() else v)
                         for k, v in payload["state_dict"].items()}
payload["dtype"] = "bf16"
```

## Verified on the full OpenFS test split (12,521 clips, beam 8)

| | fp32 reference | bf16 (two runs, bit-identical) |
|---|---|---|
| errors | 8,186 | 8,178 |
| CER | 3.7024% | 3.6988% |
| exact strings | 8,892 | 8,897 |

fp16 was tried and rejected: the m5 member produces NaN logits in half-precision
decode (test CER 1.0). bf16's exponent range avoids this.

## Use

```
ASL_ENSEMBLE_MODEL_DIR=data/models/ensemble-2026-10-25-bf16 npm run dev
```

`model/ensemble_refiner.py` loads `dtype: "bf16"` payloads natively and refuses
fp16 files.

## License

Inherits the OpenFS research-only / non-commercial terms. Personal + research use
only; cannot be shipped commercially.
