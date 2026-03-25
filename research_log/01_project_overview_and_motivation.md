# Chapter 01 — Project Overview and Motivation

**Previous chapter:** [00 Index](00_INDEX.md)
**Next chapter:** [02 Architecture](02_architecture.md)

---

## 1. Research Objective and Core Hypothesis

### What we are trying to do

We want to synthesise a realistic chest X-ray containing **both** cardiomegaly and pleural effusion without ever having trained a model jointly on comorbid data. Instead, we train separate generative models — one per disease — and **compose** them at inference time.

This is clinically valuable because:
- Comorbid cases (both diseases present) are rare and expensive to label cleanly
- Controlled synthesis of combined pathologies is useful for data augmentation and robustness testing
- Disentangled latent representations allow attribute-level editing, which supports radiological explainability

### The SepVAE as a prerequisite

Before any composition can happen, we need a latent space where disease variation is separated from shared anatomy. The **Separable VAE (SepVAE)** is the first building block: a VAE with a frozen pretrained backbone (CheSS, ResNet-50) and three separate encoder heads producing:

$$z = [z_{\text{common}} \;|\; z_{\text{cardio}} \;|\; z_{\text{effusion}}]$$

- $z_{\text{common}}$ (4 channels): captures shared anatomy, patient position, acquisition settings
- $z_{\text{cardio}}$ (2 channels): should capture **only** cardiomegaly-specific variation
- $z_{\text{effusion}}$ (2 channels): should capture **only** effusion-specific variation

Everything downstream — LDM training, score composition, comorbid synthesis — depends on these heads being genuinely disentangled. If $z_{\text{cardio}}$ leaks effusion information or $z_{\text{common}}$ absorbs cardiomegaly variation, composition will produce globally inconsistent images.

### Informal hypothesis (starting point)

> "If disease-related variation can be approximately factorised from shared anatomical and acquisition variation, then score composition in latent space should better approximate multi-pathology generation than composition in a fully entangled latent space."

This hypothesis is refined into a falsifiable form in [Chapter 10](10_verification_and_hypothesis.md) after we identified what "composition" concretely means and what must be measured to test it.

---

## Supplementary: Background and Clinical Motivation

*The following section provides additional context on why this problem is worth solving and what the SepVAE must provide.*

The broader goal of this project is to enable **compositional multi-pathology generation** in chest X-rays. Concretely: given a model that has learned what cardiomegaly looks like and a separate model that has learned what pleural effusion looks like, can we synthesise a realistic image containing *both* without ever having jointly trained on comorbid data?

This is clinically valuable because:
- Comorbid cases are rare and expensive to collect with clean labels
- Controlled synthesis allows studying pathology interactions
- Disentangled representations could allow attribute-level editing for explainability

The SepVAE is the first building block: a variational autoencoder that produces **factorized latent representations** — one shared head $z_{\text{common}}$ capturing anatomy and acquisition, and one head per disease ($z_{\text{cardio}}$, $z_{\text{effusion}}$) capturing only disease-specific variation.

The SepVAE training uses the VinBigData chest X-ray dataset. The triplet dataloader provides `(x_norm, x_effusion, x_cardiomegaly)` batches. The flag `exclude_cross_disease_overlap=True` removes patients who have both diseases from training, reducing statistical correlation between disease heads — though it does not make the underlying pathophysiology independent (right heart failure, for example, causes both cardiomegaly and pleural effusion).

The key question the training is designed to answer: can we create a representation where $z_{\text{cardio}}$ and $z_{\text{effusion}}$ are conditionally independent given $z_{\text{common}}$? This is the conditional independence assumption required for the downstream composition strategy (Strategy A) to be mathematically sound. See [Chapter 09](09_composition_theory.md) for the full mathematical treatment.

---

*End of Chapter 01. Continue to [Chapter 02: Architecture](02_architecture.md).*
