# Leaf Disease Classifier

A supervised + unsupervised deep learning project that diagnoses plant leaf diseases from images. Built as a hands-on learning project to move from foundational ML (scikit-learn) into deep learning (TensorFlow), and further into unsupervised deep learning and PyTorch.

This project is not a single model — it's a **learning arc**, documented phase by phase. Each phase has its own notebook, its own deliverable, and its own write-up. Both contributors build the full pipeline independently (see [Branch Strategy](#branch-strategy) below) rather than splitting phases between people, so both of us get full end-to-end reps on every concept.

---

## Project Goal

Given a photo of a plant leaf, predict:
- Whether it's healthy or diseased
- If diseased, which specific disease (multi-class, ~38 classes across multiple crops)
- A confidence score for the prediction

Secondary goal: use the same problem to deliberately practice supervised deep learning, unsupervised deep learning, transfer learning, and cross-framework implementation (TensorFlow → PyTorch).

---

## Dataset

**PlantVillage Dataset** (Kaggle) — ~54,000 labeled leaf images across healthy and diseased classes for crops including tomato, potato, corn, and others.

- Source: Kaggle, search "PlantVillage Dataset"
- Format: folder-per-class structure, RGB images
- Split: train / validation / test (test set held out and untouched until final evaluation, across all phases)

---

## Phases

### Phase 1 — CNN From Scratch (TensorFlow)
**Duration:** Week 1

**Deliverable:**
- A custom-built CNN (no pretrained weights) trained on PlantVillage
- Training/validation accuracy & loss curves (before and after adding augmentation/regularization)
- Written note on overfitting observed and what fixed it

**Learning Outcomes:**
- Convolution, pooling, and dense layers — what each is actually doing
- Diagnosing overfitting from train/val curves
- Data augmentation (flip, rotation, zoom) and why each choice makes sense for leaf images
- Batch normalization and dropout as regularization
- Handling class imbalance (`class_weight` or resampling)

**Tech Used:** TensorFlow, Keras, `tf.data` / `image_dataset_from_directory`, matplotlib, numpy

---

### Phase 2 — Transfer Learning (TensorFlow)
**Duration:** Week 2

**Deliverable:**
- MobileNetV2 (or ResNet50) with frozen base + custom classification head
- Fine-tuned version (top layers unfrozen, low learning rate)
- Comparison table: scratch CNN vs frozen transfer model vs fine-tuned transfer model (accuracy, training time, parameter count)

**Learning Outcomes:**
- What pretrained ImageNet weights actually encode
- Feature extraction vs fine-tuning
- Why fine-tuning needs a much lower learning rate (catastrophic forgetting)
- Correct preprocessing for pretrained models (not just `/255.`)
- Precision, recall, F1, and confusion matrix reading for multi-class problems

**Tech Used:** TensorFlow, Keras Applications (MobileNetV2/ResNet50), scikit-learn (metrics), matplotlib

---

### Phase 3 — Autoencoder for Anomaly Detection (TensorFlow, Unsupervised)
**Duration:** Week 3

**Deliverable:**
- Convolutional autoencoder trained only on healthy leaf images
- Reconstruction-error-based healthy/diseased classification
- Comparison: autoencoder anomaly detection accuracy vs Phase 1/2 supervised accuracy
- Bonus: PCA/t-SNE visualization of encoder bottleneck features, color-coded by true disease label

**Learning Outcomes:**
- Encoder-bottleneck-decoder architecture and representation learning
- Reconstruction loss as a training signal without labels
- Anomaly detection logic: the model is bad at reconstructing what it's never seen
- Dimensionality reduction: PCA vs t-SNE, when to use which
- Concretely understanding why supervised beats unsupervised when labels are available

**Tech Used:** TensorFlow, Keras, scikit-learn (PCA, t-SNE), matplotlib

---

### Phase 4 — PyTorch Rebuild
**Duration:** Week 4

**Deliverable:**
- Phase 1 (from-scratch CNN) rebuilt in PyTorch, same architecture, same dataset
- Phase 2 (transfer learning) rebuilt using `torchvision.models`
- Comparison: do TensorFlow and PyTorch versions land on similar accuracy? Document any discrepancies and why.

**Learning Outcomes:**
- `nn.Module` class-based model design
- Manual training loops (`optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`)
- `Dataset` / `DataLoader` custom data pipelines
- Device management (`.to(device)`) for GPU training
- Logits vs probabilities in `CrossEntropyLoss` (a common first-time PyTorch mistake)

**Tech Used:** PyTorch, torchvision, numpy, matplotlib

---

### Phase 5 — Deployment (Stretch Goal)
**Duration:** Week 5 (optional, post core-learning)

**Deliverable:**
- Minimal inference API (Flask or FastAPI) serving the best-performing model
- Basic frontend (can leverage existing MERN skills) for image upload → prediction display
- Deployed demo link in README

**Learning Outcomes:**
- Model serialization and loading for inference
- Separating training code from inference code
- Basic MLOps hygiene (model versioning, clean inference pipeline)

**Tech Used:** Flask/FastAPI, React (optional), model files (`.h5`/SavedModel or `.pt`)

---

## Branch Strategy

Both contributors build the **entire pipeline independently**, not split by phase. This is intentional: the goal of this project is individual skill-building for both of us, not fastest possible delivery.

- `main` — stable, holds the final merged/best-performing version of each phase once both branches are reviewed
- `muntaha` — full independent build of all phases
- `hamid` — full independent build of all phases

**Weekly sync (end of each phase):**
- Both branches pushed and compared
- Walk each other through decisions made (architecture choices, hyperparameters, what broke and why)
- Best-performing/cleanest implementation of that phase gets merged into `main`, with a note in the commit/README crediting the independent build it came from

**Why this structure:** it preserves individual accountability (you can't show up to a sync with nothing built) and full learning coverage (neither of us skips a phase), while still producing one coherent shared repo at the end.

---

## Progress Log

*(Update this section as phases are completed)*

| Phase | Status | Best Accuracy | Notes |
|-------|--------|---------------|-------|
| Phase 1 — CNN from scratch | Not started | — | — |
| Phase 2 — Transfer learning | Not started | — | — |
| Phase 3 — Autoencoder | Not started | — | — |
| Phase 4 — PyTorch rebuild | Not started | — | — |
| Phase 5 — Deployment | Not started | — | — |

---

## Repo Structure

```
/data                          # dataset download script / instructions (not raw data)
/notebooks
  01_cnn_from_scratch_tf.ipynb
  02_transfer_learning_tf.ipynb
  03_autoencoder_anomaly_tf.ipynb
  04_pytorch_rebuild.ipynb
/app                           # Phase 5 deployment code (Flask/FastAPI + frontend)
/results
  comparison_table.md          # accuracy, params, train time across all approaches
  training_curves/             # saved plots
README.md
```

---

## Key Learnings

*(This section grows as we go — what broke, what fixed it, what surprised us.)*

-
-
-
