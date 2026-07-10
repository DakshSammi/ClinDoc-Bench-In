# Dataset

The final benchmark contains 90 patients, 125 documents, and 150 images.

Documents include heterogeneous prescription and clinical record layouts from Indian healthcare settings. The dataset intentionally preserves realistic challenges such as handwriting, printed templates, scanned artifacts, multi-page encounters, and variable clinical detail.

## Manifest

The canonical benchmark manifest is:

`benchmark/data/benchmark_manifest.csv`

Important fields include:

| Field | Meaning |
| --- | --- |
| `document_id` | Unique document identifier |
| `patient_id` | Patient grouping identifier |
| `image_paths` | Ordered source images |
| `ground_truth_path` | Canonical annotation path |
| `hospital` | Site or hospital metadata |
| `department` | Department or specialty metadata |
| `num_images` | Number of source images for the document |
| `benchmark_include` | Inclusion flag |

## Privacy

Raw clinical material must not be published without the appropriate approvals. Paper examples are generated as anonymized copies under `paper_assets/examples/` using opaque black redaction boxes.
