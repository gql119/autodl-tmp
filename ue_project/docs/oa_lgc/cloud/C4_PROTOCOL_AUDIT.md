# C4 Protocol Audit

## Result

Status: `blocked`.

Authoritative artifact: `artifacts/oa_lgc/cloud/20260714_151949_C4_0/`.

## Available data

- Local VOC root contains only `VOC2007`.
- VOC2007 train: 2,501 IDs.
- VOC2007 val: 2,510 IDs.
- VOC2007 trainval: 5,011 IDs, 5,011 JPEGs, and 5,011 annotations.
- The 800/200 mini split is fully traceable to VOC2007 trainval and has zero internal train/val ID overlap.

## Missing required data

- `VOC2012/ImageSets/Main/trainval.txt` and its dataset are absent.
- `VOC2007/ImageSets/Main/test.txt` and the corresponding test data are absent.
- Therefore the exact combined training manifest and exact independent test manifest cannot be generated.
- Exact train/test hashes, distributions, and overlap cannot be asserted.

The artifact intentionally does not create files that could be mistaken for valid formal `train_manifest.txt` or `test_manifest.txt`. It stores the available VOC2007 trainval manifest, historical mini manifests, available-component hashes, class distribution, and explicit blocked hash records instead.

No data were downloaded. No test data were used for training, delta optimization, or model selection.
