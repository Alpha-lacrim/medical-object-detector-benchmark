# Citation Verification Audit

## Current focused manuscript — 2026-09-08

This section supersedes the manuscript-location and citation-coverage statements
in the archived audit below. It audits `report/paper_draft.md` only. All 16
current citation keys were checked against primary/authoritative sources;
37 unique BibTeX entries remain because the database also serves the unchanged
historical report and long alternate. No uncited entry is injected into the
canonical manuscript. DOI/publisher/official sources are preferred; access
limitations and the one author-uploaded-paper fallback are explicit.

| Key | Verified source / authority | Current bounded use | Verification and limits |
|---|---|---|---|
| `kuzucu2024calibration` | Kuzucu, Oksuz, Sadeghi, Dokania, ECCV 2024, pp. 185--204; [Springer DOI](https://doi.org/10.1007/978-3-031-72664-4_11), [official ECVA full text](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03148.pdf) | §§1, 2.1, 5: threshold/population pitfalls already documented | VERIFIED: especially pp. 1--4. Supports prior-art attribution, not novelty or a new calibration experiment here. |
| `kuppers2020calibration` | Küppers et al., CVPR Workshops 2020; [DOI](https://doi.org/10.1109/CVPRW50498.2020.00171), [CVF paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w20/Kuppers_Multivariate_Confidence_Calibration_for_Object_Detection_CVPRW_2020_paper.pdf) | §§2.1, 3.7: multivariate detection-confidence framework | VERIFIED. Our minimum-cell convention and support sensitivities are repository implementation choices, not clinical-risk calibration. IEEE pagination is 1322--1330; CVF's landing-page listing uses a different short pagination. |
| `shih2019rsna` | Shih et al., Radiology: AI 2019, e180041; [publisher](https://doi.org/10.1148/ryai.2019180041) | §§1, 2.2: expert opacity boxes, categories, adjudication | VERIFIED from publisher-indexed resource/annotation text. Source-cohort numbers are not substituted for our audited Stage 2 counts or selected patient groups. |
| `rsna2022datasetdescription` | RSNA, *2018 Pneumonia Detection Challenge Dataset Description*, document marked V1 02/07/2022; [official PDF](https://www.rsna.org/rsnai/-/media/Files/RSNA/Education/AI%20resources%20and%20training/AI%20image%20challenge/RSNA-2018-Pneumonia-Detection-Challenge-Dataset-Description.ashx?hash=8331DDFA52AA5EE9C3C99B5983BB680D7BD2ABBF&la=en) | §2.2: annotation workflow and DICOM conversion/header additions | VERIFIED, both pages. The template's option lists are not treated as completed answers. It supplies no local ethics/consent determination. |
| `rsna2018challenge` | RSNA/Kaggle 2018; [official evaluation](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview/evaluation) | §2.2: original challenge score differs from internal COCO AP | VERIFIED from official indexed competition evaluation text; direct page extraction was empty. IoU range is 0.40--0.75. No leaderboard metric is relabeled as our COCO AP. |
| `wu2024pneumonia` | Wu et al., Scientific Reports 14:1929 (2024); [publisher](https://doi.org/10.1038/s41598-024-52156-7) | §§2.2, 5: prior RSNA anchor-free comparison and ablations | VERIFIED, Tables 2--3 and methods. Protocol-specific context only; not a reproduced result or architecture-family effect. |
| `chinnam2026modern` | Chinnam, Gogineni, Rao, Suryanarayana, Nagaraj, Sunitha, CCIC 2026; [DOI](https://doi.org/10.1109/CCIC68129.2026.11485895), [IEEE record](https://ieeexplore.ieee.org/document/11485895), [author-uploaded paper](https://www.researchgate.net/publication/404406332_Automated_Pneumonia_Detection_in_Medical_X-Rays_Using_Modern_Object_Detectors) | §§2.2, 5: recent direct RSNA comparison including YOLOv8l, Faster R-CNN, Deformable DETR | VERIFIED from the primary paper uploaded by coauthor Seema Nagaraj; IEEE extraction was unavailable. Title/DOI/authors appear in the paper. Table 1 gives different input resolutions; §III instead describes resizing all images alike. We say “reported” resolutions, do not resolve that inconsistency for the authors, and import no numerical performance or clinical-use claim. |
| `ren2015fasterrcnn` | Ren et al., NeurIPS 2015; [official proceedings](https://proceedings.neurips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html) | §2.3: proposals and second-stage detection | VERIFIED from proceedings text; no general accuracy claim. |
| `lin2017fpn` | Lin et al., CVPR 2017; [CVF proceedings](https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html), DOI 10.1109/CVPR.2017.106 | §2.3: multiscale features | VERIFIED from official indexed abstract; direct landing-page fetch returned 403. |
| `ultralytics2026release` | Ultralytics 8.4.110; [PyPI version record](https://pypi.org/project/ultralytics/8.4.110/) | §2.3: pinned implementation identity | VERIFIED against release page and local resolved configs; no assertion about the latest available model. |
| `ultralytics2026yolo11config` | Ultralytics, pinned v8.4.110; [official raw model configuration](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.110/ultralytics/cfg/models/11/yolo11.yaml) | §2.3: YOLO11s configuration authority | VERIFIED. Raw source was readable when the GitHub HTML route failed. |
| `tejani2024claim` | Tejani et al., Radiology: AI 6(4):e240300 (2024); [publisher](https://doi.org/10.1148/ryai.240300) | §2.3: reporting, reference-standard and internal/external-testing terminology | VERIFIED from full publisher text. Directly applicable; the local crosswalk remains an audit, not certification. |
| `lin2014coco` | Lin et al., ECCV 2014, pp. 740--755; [publisher](https://doi.org/10.1007/978-3-319-10602-1_48) | §3.3: COCO evaluation framework | VERIFIED resource identity. Exact IoU grid, interpolation and cap are supported by the official evaluator and the pinned local pycocotools/code/config, not inferred solely from the original dataset paper. |
| `efron1993bootstrap` | Efron and Tibshirani, *An Introduction to the Bootstrap*; [DOI](https://doi.org/10.1201/9780429246593), [publisher listing](https://www.routledge.com/An-Introductionto-the-Bootstrap/Efron-Tibshirani/p/book/9780412042317) | §3.5: bootstrap framework | VERIFIED method source; the publisher lists a 1994 edition while its series list identifies the original 1993 book retained in the bibliography. Patient/run design and actual results are ours. |
| `phipson2010permutation` | Phipson and Smyth, 2010; [DOI](https://doi.org/10.2202/1544-6115.1585), [PubMed author abstract](https://pubmed.ncbi.nlm.nih.gov/21044043/) | §3.5: nonzero Monte Carlo permutation p-values | VERIFIED through author abstract/identifier record; direct DOI route was unavailable. |
| `holm1979simple` | Holm, Scandinavian Journal of Statistics 6(2):65--70 (1979); [JSTOR source](https://www.jstor.org/stable/4615733), [issue contents](https://www.jstor.org/stable/i412579) | §3.5: multiplicity adjustment | Source identity VERIFIED from the official issue listing. Full text was unavailable on this route; the existing implementation/tests establish the named adjustment used locally. |

Searches on 2026-09-08 covered detector calibration/threshold pitfalls and
direct RSNA modern-detector comparisons, with 2024--2026 queries and checks of
older retained methods. This was a focused literature refresh, not a systematic
review. Reviews, leaderboards, unrelated RSNA bone-age/CT studies, and
classification-only COVID/pneumonia comparisons were not used as direct
lung-opacity detector evidence. The old broad “comparability gap” and brain-MRI
detour were removed from journal prose. Prior saliency, corruption, DICOM,
F-beta/loss, and raw-score decision-analysis references remain in the shared
bibliography for historical/supplementary use.

Offline syntax, duplicate-key/DOI, required-field, and used-key resolution
checks are documented in README and implemented by
`scripts/check_bibliography.py`. They do not substitute for the source-support
review above. Current and preserved manuscript citations all resolve.

## Historical audit retained verbatim (through 2026-09-07)

The text below is a dated provenance record of the broader draft. Its uses of
“current manuscript,” old section numbers, and citation totals describe that
earlier version; the current mapping is exclusively the table above. The
reporting-guideline references retained below still document the crosswalk's
by-analogy use of STARD-AI/TRIPOD+AI and are not new primary frameworks.

# Citation Audit

Audit date: 2026-08-31

Scope: every citation key used by the current manuscript,
[`report/paper_draft.md`](../report/paper_draft.md), that supports an
architecture/method, dataset, DICOM, statistical, calibration, decision-curve,
robustness, or XAI statement. All cited background sources were also reviewed;
none of the manuscript's cited references was sampled out. The final section
separately audits the three reporting-guideline sources used by
[`REPORTING_CHECKLIST.md`](REPORTING_CHECKLIST.md).

Verdicts:

- **VERIFIED** — the source exists and directly supports the bounded claim.
- **PARTIAL** — the source exists but supports only part of the claim or is a
  mutable implementation source that needs the paired pinned/local evidence.
- **UNSUPPORTED** — the source exists but does not support the stated claim.
- **WRONG ATTRIBUTION** — the work, authorship, identifier, or claim is
  attributed to the wrong source.

After the corrections listed below, no manuscript row remains `UNSUPPORTED` or
`WRONG ATTRIBUTION`. `PARTIAL` rows are retained to make source limitations
visible; the manuscript wording and paired evidence are already bounded so
that those rows do not carry unsupported conclusions.

## Architecture, software, and evaluation methods

| Citation key | Full source and year | Persistent identifier / authoritative location | Exists | Exact manuscript claim supported | Verdict |
|---|---|---|---|---|---|
| `ren2015fasterrcnn` | Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks* (2015). | [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html); arXiv:1506.01497. | Yes | §2.1: Faster R-CNN couples an RPN with an RoI-based classifier/regressor while sharing convolutional features. | **VERIFIED** |
| `lin2017fpn` | Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. *Feature Pyramid Networks for Object Detection* (2017). | DOI [10.1109/CVPR.2017.106](https://doi.org/10.1109/CVPR.2017.106); arXiv:1612.03144. | Yes | §2.1: the ResNet-50 FPN represents objects using semantically strong features at multiple scales. | **VERIFIED** |
| `redmon2016yolo` | Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. *You Only Look Once: Unified, Real-Time Object Detection* (2016). | DOI [10.1109/CVPR.2016.91](https://doi.org/10.1109/CVPR.2016.91); arXiv:1506.02640. | Yes | §2.1: YOLO introduced a contrasting one-network-pass dense detection formulation. | **VERIFIED** |
| `li2020gfl` | Xiang Li, Wenhai Wang, Lijun Wu, Shuo Chen, Xiaolin Hu, Jun Li, Jinhui Tang, and Jian Yang. *Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection* (2020). | [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html); arXiv:2006.04388. | Yes | §2.1: distributed box regression follows the general distributional-regression approach introduced with GFL. | **PARTIAL** — verifies the general DFL/GFL method, not the exact YOLO11s implementation; the pinned Ultralytics config/source supplies that implementation evidence. |
| `ultralytics2024yolo11` | Glenn Jocher, Jing Qiu, and Ayush Chaurasia. *Ultralytics YOLO11* (software, 2024). | [Official repository](https://github.com/ultralytics/ultralytics) and its `CITATION.cff`; no DOI for YOLO11. | Yes | §2.1: YOLO11 is the software family used and lacks a standalone peer-reviewed architecture paper, so implementation sources are authoritative. | **PARTIAL** — the repository is mutable and family-wide; authorship was corrected in `references.bib`, and the release/config/local graph pin the actual implementation. |
| `ultralytics2026release` | Ultralytics. *ultralytics 8.4.110* (software release, 2026). | [PyPI 8.4.110](https://pypi.org/project/ultralytics/8.4.110/). | Yes | §§2.1 and 3.2: the study uses the exact `ultralytics` 8.4.110 release. | **VERIFIED** |
| `ultralytics2026yoloarchitecture` | Ultralytics. *YOLO Architecture Explained* (official documentation, accessed 2026). | [Official documentation](https://docs.ultralytics.com/guides/yolo-architecture/). | Yes | §2.1: YOLO11 uses a backbone/neck/head, multi-scale features, an anchor-free decoupled head, DFL-style distributed offsets, and NMS. | **PARTIAL** — directly supports the architecture description but is a mutable page; the pinned config and instantiated graph are the exact authority. |
| `ultralytics2026yolo11config` | Ultralytics. *YOLO11 Object-Detection Model Configuration* for `ultralytics` 8.4.110 (2026). | [Pinned release configuration](https://github.com/ultralytics/ultralytics/blob/v8.4.110/ultralytics/cfg/models/11/yolo11.yaml). | Yes | §2.1: the exact YOLO11s layer topology/scale and source-release authority used in the study. | **VERIFIED** |
| `lin2014coco` | Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. *Microsoft COCO: Common Objects in Context* (2014). | DOI [10.1007/978-3-319-10602-1_48](https://doi.org/10.1007/978-3-319-10602-1_48); arXiv:1405.0312. | Yes | §3.3: COCO AP is reported at IoU 0.50 and averaged across 0.50:0.95 under the common evaluator. | **VERIFIED** |

## Dataset and domain-background sources

| Citation key | Full source and year | Persistent identifier / authoritative location | Exists | Exact manuscript claim supported | Verdict |
|---|---|---|---|---|---|
| `shih2019rsna` | George Shih, Carol C. Wu, Safwan S. Halabi, Marc D. Kohli, Luciano M. Prevedello, Tessa S. Cook, Arjun Sharma, Judith K. Amorosa, Veronica Arteaga, Maya Galperin-Aizenberg, Ritu R. Gill, Myrna C. B. Godoy, Stephen Hobbs, Jean Jeudy, Archana Laroia, Palmi N. Shah, Dharshan Vummidi, Kavitha Yaddanapudi, and Anouk Stein. *Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia* (2019). | DOI [10.1148/ryai.2019180041](https://doi.org/10.1148/ryai.2019180041). | Yes | §§2.2 and 3.1: provenance of the RSNA/NIH chest-radiograph resource and expert pulmonary-opacity box workflow. | **VERIFIED** |
| `yao2021pneumonia` | Shangjie Yao, Yaowu Chen, Xiang Tian, and Rongxin Jiang. *Pneumonia Detection Using an Improved Algorithm Based on Faster R-CNN* (2021). | DOI [10.1155/2021/8854892](https://doi.org/10.1155/2021/8854892). | Yes | §2.2: a chest-radiograph Faster R-CNN study jointly changes backbone, FPN/anchors, preprocessing, and suppression choices. | **VERIFIED** |
| `wu2024pneumonia` | Linghua Wu, Jing Zhang, Yilin Wang, Rong Ding, Yueqin Cao, Guiqin Liu, Changsheng Liufu, Baowei Xie, Shanping Kang, Rui Liu, Wenle Li, and Furen Guan. *Pneumonia Detection Based on RSNA Dataset and Anchor-Free Deep Learning Detector* (2024). | DOI [10.1038/s41598-024-52156-7](https://doi.org/10.1038/s41598-024-52156-7). | Yes | §2.2: an anchor-free RSNA detector uses its own augmentations, focal-loss formulation, thresholds/NMS, and AP/AR definitions. | **VERIFIED** |
| `kang2023rcsyolo` | Ming Kang, Chee-Ming Ting, Fung Fung Ting, and Raphaël C.-W. Phan. *RCS-YOLO: A Fast and High-Accuracy Object Detector for Brain Tumor Detection* (2023). | DOI [10.1007/978-3-031-43901-8_57](https://doi.org/10.1007/978-3-031-43901-8_57). | Yes | §2.2: medical YOLO work frames a 2-D brain-MRI detector as an accuracy/speed problem, with a modality and comparison scope different from this study. | **VERIFIED** |
| `kang2025pkyolo` | Ming Kang, Fung Fung Ting, Raphael C.-W. Phan, and Chee-Ming Ting. *PK-YOLO: Pretrained Knowledge Guided YOLO for Brain Tumor Detection in Multiplanar MRI Slices* (2025). | [Official WACV paper](https://openaccess.thecvf.com/content/WACV2025/html/Kang_PK-YOLO_Pretrained_Knowledge_Guided_YOLO_for_Brain_Tumor_Detection_in_WACV_2025_paper.html). | Yes | §2.2: related multiplanar MRI work emphasizes domain pretraining and small-target/model choices under a different data organization. | **VERIFIED** |
| `hendrycks2019corruptions` | Dan Hendrycks and Thomas G. Dietterich. *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations* (2019). | [OpenReview](https://openreview.net/forum?id=HJz6tiCqYm); arXiv:1903.12261. | Yes | §2.3: clean accuracy need not predict performance under ordered lighting/noise/blur/compression corruptions. | **VERIFIED** |
| `michaelis2019detectionrobustness` | Claudio Michaelis, Benjamin Mitzkus, Robert Geirhos, Evgenia Rusak, Oliver Bringmann, Alexander S. Ecker, Matthias Bethge, and Wieland Brendel. *Benchmarking Robustness in Object Detection: Autonomous Driving When Winter Is Coming* (2019). | arXiv:[1907.07484](https://arxiv.org/abs/1907.07484). | Yes | §2.3: common-corruption robustness analysis extends to object detection. | **VERIFIED** |

## Calibration, XAI, statistics, DICOM, and decision analysis

| Citation key | Full source and year | Persistent identifier / authoritative location | Exists | Exact manuscript claim supported | Verdict |
|---|---|---|---|---|---|
| `kuppers2020calibration` | Fabian Küppers, Jan Kronenberger, Amirhossein Shantia, and Anselm Haselhoff. *Multivariate Confidence Calibration for Object Detection* (2020). | DOI [10.1109/CVPRW50498.2020.00171](https://doi.org/10.1109/CVPRW50498.2020.00171). | Yes | §§2.3 and 3.6: D-ECE conditions correctness on confidence and predicted box geometry using a multivariate binning framework. | **VERIFIED** |
| `selvaraju2017gradcam` | Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization* (2017). | DOI [10.1109/ICCV.2017.74](https://doi.org/10.1109/ICCV.2017.74); arXiv:1610.02391. | Yes | §2.3: Grad-CAM uses gradients of a target score to weight convolutional features and produce a coarse localization map. | **VERIFIED** |
| `adebayo2018sanity` | Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, and Been Kim. *Sanity Checks for Saliency Maps* (2018). | [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2018/file/294a8ed24b1ad22ec2e7efea049b8737-Paper.pdf); arXiv:1810.03292. | Yes | §§2.3 and 3.9: visually plausible maps can survive model randomization; the distinct data-randomization control permutes labels and retrains, which this study did not perform. | **VERIFIED** |
| `arun2021saliency` | Nishanth Arun, Nathan Gaw, Praveer Singh, Ken Chang, Mehak Aggarwal, Bryan Chen, Katharina Hoebel, Sharut Gupta, Jay Patel, Mishka Gidwani, Julius Adebayo, Matthew D. Li, and Jayashree Kalpathy-Cramer. *Assessing the Trustworthiness of Saliency Maps for Localizing Abnormalities in Medical Imaging* (2021). | DOI [10.1148/ryai.2021200267](https://doi.org/10.1148/ryai.2021200267). | Yes | §2.3: saliency-map plausibility/localization alone is insufficient to establish trustworthy or clinically appropriate model reasoning in medical imaging. | **VERIFIED** |
| `zhang2016excitation` | Jianming Zhang, Zhe Lin, Jonathan Brandt, Xiaohui Shen, and Stan Sclaroff. *Top-Down Neural Attention by Excitation Backprop* (2016). | DOI [10.1007/978-3-319-46493-0_33](https://doi.org/10.1007/978-3-319-46493-0_33); arXiv:1608.00507. | Yes | §2.3: the pointing-game localization measure is an established way to compare an attention maximum with a target box. | **VERIFIED** |
| `wang2004ssim` | Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. *Image Quality Assessment: From Error Visibility to Structural Similarity* (2004). | DOI [10.1109/TIP.2003.819861](https://doi.org/10.1109/TIP.2003.819861); PMID 15376593. | Yes | §3.9: SSIM is used alongside Pearson and Spearman similarity for normalized Grad-CAM maps. | **VERIFIED** |
| `efron1993bootstrap` | Bradley Efron and Robert J. Tibshirani. *An Introduction to the Bootstrap* (1993). | DOI [10.1201/9780429246593](https://doi.org/10.1201/9780429246593). | Yes | §3.10: resampling is used to construct pointwise bootstrap intervals; the manuscript then specifies its project-specific two-stage patient-cluster/run procedure. | **VERIFIED** |
| `phipson2010permutation` | Belinda Phipson and Gordon K. Smyth. *Permutation P-values Should Never Be Zero: Calculating Exact P-values When Permutations Are Randomly Drawn* (2010). | DOI [10.2202/1544-6115.1585](https://doi.org/10.2202/1544-6115.1585); PMID 21044043. | Yes | §3.10: randomly drawn permutation p-values use a plus-one correction rather than permitting zero. | **VERIFIED** |
| `holm1979simple` | Sture Holm. *A Simple Sequentially Rejective Multiple Test Procedure* (1979). | DOI [10.2307/4615733](https://doi.org/10.2307/4615733); JSTOR 4615733. | Yes | §3.10: permutation p-values are adjusted using Holm's sequentially rejective familywise procedure. | **VERIFIED** |
| `dicom2026ps33` | National Electrical Manufacturers Association. *DICOM PS3.3 2026c: Information Object Definitions* (2026). | Official sections [A.8.1, Secondary Capture Image IOD](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.8.html), C.8.6 Secondary Capture modules, and C.11.2.1.2 default `LINEAR` window behavior. | Yes | §3.8: Secondary Capture objects are modality-independent converted images; the audited tags constrain what can be inferred, and the exact default-`LINEAR` display transforms are synthetic sensitivities. | **VERIFIED** |
| `dicom2026ps34` | National Electrical Manufacturers Association. *DICOM PS3.4 2026c: Service Class Specifications* (2026). | Official [§B.5 Standard SOP Classes](https://dicom.nema.org/medical/dicom/current/output/chtml/part04/sect_b.5.html), including Secondary Capture Image Storage UID `1.2.840.10008.5.1.4.1.1.7`. | Yes | §3.8: the audited SOP Class UID is Secondary Capture Image Storage rather than a native CR/DX storage SOP class. | **VERIFIED** |
| `vickers2006decisioncurve` | Andrew J. Vickers and Elena B. Elkin. *Decision Curve Analysis: A Novel Method for Evaluating Prediction Models* (2006). | DOI [10.1177/0272989X06295361](https://doi.org/10.1177/0272989X06295361); PMID 17099194. | Yes | §6: conventional DCA requires a threshold probability with the corresponding odds weight; a raw detector score is not automatically that quantity. | **VERIFIED** |
| `vickers2008decisioncurveextensions` | Andrew J. Vickers, Angel M. Cronin, Elena B. Elkin, and Mithat Gonen. *Extensions to Decision Curve Analysis, a Novel Method for Evaluating Diagnostic Tests, Prediction Models and Molecular Markers* (2008). | DOI [10.1186/1472-6947-8-53](https://doi.org/10.1186/1472-6947-8-53); PMID 19036144. | Yes | §6: the probability/threshold semantics required for conventional decision-curve net benefit do not follow from uncalibrated detector scores. | **VERIFIED** |

## Reporting-standard sources used by the audit

These sources support reporting-governance statements in
[`REPORTING_CHECKLIST.md`](REPORTING_CHECKLIST.md), not a scientific result in
the manuscript. They are included because standards sources were explicitly in
scope.

| Audit key | Full source and year | Persistent identifier / authoritative location | Exists | Exact reporting claim supported | Verdict |
|---|---|---|---|---|---|
| `claim2024` | Ali S. Tejani, Michail E. Klontzas, Anthony A. Gatti, John T. Mongan, Linda Moy, Seong Ho Park, and Charles E. Kahn Jr., for the CLAIM 2024 Update Panel. *Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update* (2024). | DOI [10.1148/ryai.240300](https://doi.org/10.1148/ryai.240300); PMID 38809149; [official RSNA checklist page](https://pubs.rsna.org/page/ai/claim). | Yes | CLAIM is the direct medical-imaging AI framework; its official responses are Yes/No/NA; Yes should cite manuscript location and No/NA should explain; “internal testing” and “external testing” replace ambiguous validation terminology. | **VERIFIED** |
| `stardai2025` | Viknesh Sounderajah, Ahmad Guni, Xiaoxuan Liu, Gary S. Collins, Alan Karthikesalingam, Sheraz R. Markar, Robert M. Golub, Alastair K. Denniston, Shravya Shetty, David Moher, Patrick M. Bossuyt, Ara Darzi, and Hutan Ashrafian, with the STARD-AI Steering Committee. *The STARD-AI Reporting Guideline for Diagnostic Accuracy Studies Using Artificial Intelligence* (2025). | DOI [10.1038/s41591-025-03953-8](https://doi.org/10.1038/s41591-025-03953-8); PMID 40954311; author correction DOI 10.1038/s41591-026-04570-9. | Yes | The final version of record contains 40 items for AI-centered diagnostic-accuracy studies. It supersedes the 2020 announcement and 2021 protocol for checklist use, and its scope supports the decision that this localization benchmark is not a clean primary fit. | **VERIFIED** |
| `tripodai2024` | Gary S. Collins, Karel G. M. Moons, Paula Dhiman, Richard D. Riley, Andrew L. Beam, Ben Van Calster, Marzyeh Ghassemi, Xiaoxuan Liu, Johannes B. Reitsma, Maarten van Smeden, Anne-Laure Boulesteix, Jennifer Catherine Camaradou, Leo Anthony Celi, Spiros Denaxas, Alastair K. Denniston, Ben Glocker, Robert M. Golub, Hugh Harvey, Georg Heinze, Michael M. Hoffman, André Pascal Kengne, Emily Lam, Naomi Lee, Elizabeth W. Loder, Lena Maier-Hein, Bilal A. Mateen, Melissa D. McCradden, Lauren Oakden-Rayner, Johan Ordish, Richard Parnell, Sherri Rose, Karandeep Singh, Laure Wynants, and Patricia Logullo. *TRIPOD+AI Statement: Updated Guidance for Reporting Clinical Prediction Models That Use Regression or Machine Learning Methods* (2024). | DOI [10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378); [official BMJ article](https://www.bmj.com/content/385/bmj-2023-078378). | Yes | TRIPOD+AI concerns clinical prediction models for individual prognosis/diagnosis, supporting its use here only by analogy rather than as the primary checklist for object detection. | **VERIFIED** |

## Corrections made during this audit

1. Added canonical manuscript citations for COCO evaluation
   (`lin2014coco`), SSIM (`wang2004ssim`), bootstrap methodology
   (`efron1993bootstrap`), Monte Carlo permutation plus-one correction
   (`phipson2010permutation`), Holm adjustment (`holm1979simple`), and the
   DICOM Storage SOP-class table (`dicom2026ps34`).
2. Added the exact `ultralytics` 8.4.110 release citation to the YOLO11
   implementation-authority sentence.
3. Corrected the software-author record for `ultralytics2024yolo11` to include
   Ayush Chaurasia, consistent with the official repository citation metadata.
4. Bounded implementation claims so the mutable Ultralytics repository/docs
   are not treated as sole authority; the pinned configuration, exact package
   release, local instantiated graph, and run configs jointly carry the claim.
5. Verified the final STARD-AI 2025 version of record and its 2026 author
   correction; the old protocol is not used as the checklist source.

## Bibliography entries not cited by the current manuscript

`chattopadhay2018gradcampp`, `muhammad2020eigencam`, and `petsiuk2021drise`
remain in `references.bib` as historical reading-list entries but are not cited
by `paper_draft.md`; therefore they support no current manuscript claim and are
outside the required used-reference verdict table. They should be removed only
if the final journal's bibliography pipeline emits uncited database entries.
