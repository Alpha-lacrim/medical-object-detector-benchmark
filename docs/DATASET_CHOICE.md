# Dataset choice

## Decision

This project selects the **RSNA Pneumonia Detection Challenge 2018, Stage 2
training set**. The lung-opacity detection task has one foreground class,
`Lung Opacity`; normal
and abnormal-without-opacity studies are negative images, not additional object
classes. The full labeled set is reduced to a deterministic, patient-grouped,
stratified subset of 5,000 studies for the predeclared RTX 4060 Laptop / 16 GB
RAM compute budget.

The decision prioritizes trustworthy localization annotations and a defensible
leakage-safe split over the convenience of the two smaller MRI compilations.
Both MRI candidates discard the patient/study identifiers needed to determine
whether repeated-subject slices cross splits. One also represents
`No Tumor` with non-empty detection labels, which is not a coherent foreground
object definition.

## Inspection method

All candidates were inspected on 2026-08-02. Counts below distinguish publisher
claims from the file inventory that could be verified through Kaggle metadata
and public source artifacts. Official project pages, source papers, and license
documents take precedence over downstream descriptions. The selected RSNA
annotations and official exam-to-original-image mapping were parsed locally;
the preparation command records SHA-256 hashes and a machine-readable audit.

| Criterion | RSNA Stage 2 lung-opacity task | pkdarabi Brain Tumor, v5 | ahmedsorour Brain Tumor, v1 |
|---|---|---|---|
| Image inventory | 26,684 labeled frontal chest-radiograph studies; 30,227 label rows | Data card advertises 3,903 MRI images and a 70/20/10 split; the region-blocked archive could not be fully inventoried | 5,249 JPG images: 4,737 train and 512 validation; no publisher-provided test split |
| Declared/verified detection classes | Verified: one, `Lung Opacity`; background is implicit | Kaggle data card advertises four categories—Glioma, Meningioma, Pituitary, and No Tumor—but the archive class map could not be read locally; a paper citing the set is internally inconsistent about inventory and taxonomy | Verified from labels: four YOLO IDs, `Glioma`, `Meningioma`, `No Tumor`, and `Pituitary` |
| Annotation format | CSV, one row per box or negative study; pixel-space top-left `x, y, width, height`; DICOM images | Roboflow-style YOLO text labels and a YAML class map; raster MRI images | YOLO normalized center `x, y, width, height` text files; JPG images |
| Class/study balance | 6,012 opacity; 11,821 no-opacity/not-normal; 8,851 normal. The positive studies contain 9,555 boxes | Not verifiable without the region-blocked archive; no arithmetic split estimates are treated as observations | Glioma 1,289; Meningioma 1,589; No Tumor 811; Pituitary 1,560 images (largest:smallest = 1.96) |
| Annotation audit | Among 9,555 positive box rows, none is missing/non-numeric, non-positive-area, outside the documented 1024×1024 frame, or an exact duplicate; blank coordinates on 20,672 negative rows are expected | A full annotation audit was impossible without the region-blocked archive; patient identity is absent | Archive audit: 5,873 label rows; two images lack matching labels, two labels are orphaned, one Glioma label is empty, and every no-tumor image has a non-empty detection annotation |
| Multiple images per patient/study | Yes. The official mapping shows 11,452 patients among the Stage 2 labeled studies; 4,355 have multiple exams (maximum 65) | The export retains no usable patient/study key. Slice-based sources make repeated subjects plausible, but multiplicity cannot be reconstructed | The export retains no usable patient/study key. Slice-based sources make repeated subjects plausible, but multiplicity cannot be reconstructed |
| Patient-safe split possible? | Yes, through the official RSNA mapping and original NIH patient prefix | No, not from the published filenames/labels | No, not from the published filenames/labels |
| License/terms | Bespoke RSNA terms allow academic research, education, and other commercial or non-commercial purposes; re-identification is prohibited and specified attribution applies to sharing/redistribution | Kaggle metadata declares CC BY 4.0 | Kaggle declares CC0, but file-level provenance and compatibility with the cited source collections are undocumented |
| Scale on target hardware | Too large for iterative two-detector experiments without the fixed 5,000-study subset required below | Small enough without subsampling | Small enough without subsampling |

The candidate pages are the [official RSNA challenge page](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018),
[pkdarabi Kaggle page](https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection),
and [ahmedsorour Kaggle page](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes).
RSNA cohort and annotation details are corroborated by the
[challenge publication](https://pubs.rsna.org/doi/10.1148/ryai.2019180041)
and [RSNA Atlas record](https://atlas.rsna.org/cards/74777c92-9b0b-4f06-a788-71a8a12c4e93).

## Why RSNA is the strongest benchmark

### Annotation quality

Eighteen board-certified radiologists from 16 academic institutions
participated, including 12 Society of Thoracic Radiology experts. Of the full
30,000-case challenge cohort, 4,527 cases were triple-read and 1,380 were
individually adjudicated; the images themselves came from one NIH institution.
The task is imperfect—an opacity is a radiographic observation, not proof of
pneumonia—but its boxes have clearer clinical provenance than the MRI
compilations. The local coordinate audit found no malformed, off-image, or
duplicate positive boxes.

### Balance and task definition

RSNA is imbalanced at the study level, but the imbalance is measurable and can
be preserved during selection and splitting. Negative radiographs remain valid
negative examples. By contrast, treating `No Tumor` as a boxed foreground class
mixes image classification with object detection and makes false-positive and
localization metrics difficult to interpret.

### Suitability for the comparison

The set is large enough to support Faster R-CNN/YOLO accuracy, robustness, and
explainability comparisons, while a fixed subset makes iteration feasible. Its
one foreground category also isolates detector behavior from uncertain subtype
taxonomy. DICOM-to-PNG conversion and COCO JSON provide one shared input
contract for both detectors.

### Licensing

The selected data are governed by bespoke
[RSNA challenge terms](https://www.rsna.org/-/media/files/rsna/education/ai-resources-and-training/ai-image-challenge/pneumonia-detection-challenge-terms-of-use-and-attribution.pdf).
They allow academic research, education, and other commercial or non-commercial
purposes; re-identification is prohibited, and sharing or redistribution
triggers the specified NIH/RSNA attribution requirements. Raw images are
excluded from Git.

## Hardware-scoped subset

The full labeled set is RSNA-scale, so the selected benchmark contains exactly
5,000 studies. A seeded SHA-256 ordering and grouped stratified allocation:

1. groups all exams by the true NIH patient key recovered from the official
   RSNA mapping;
2. preserves the three study strata (`Lung Opacity`,
   `No Lung Opacity / Not Normal`, and `Normal`) as closely as group constraints
   permit; and
3. assigns 70%/15%/15% to train/validation/test without placing a patient in
   more than one split.

The subset supports a controlled laptop-scale comparison; conclusions cannot
automatically be generalized to all 26,684 labeled studies, other institutions,
or other radiograph populations. The measured selection contains 2,136 NIH
patient groups and 1,136 opacity, 2,216 no-opacity/not-normal, and 1,648 normal
studies. Its exact train/validation/test counts are 3,500/750/750, with zero
patient-key overlap. Exact manifests are committed under
`data/splits/rsna-pneumonia-5000/` and can be regenerated with the commands in
`README.md`.
