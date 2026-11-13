# Guide to Hyperparameter Experiments for LDM Training

When searching for a stable training configuration, especially to combat issues like **mode collapse**, it's best to be systematic.
This guide provides a set of ordered experiments. Start with **Experiment 1** and proceed sequentially.

The goal is to balance:

* **Learning Rate (LR):** high enough to learn efficiently, but low enough to prevent instability.
* **Weight Decay (WD):** strong enough to regularize, but not too high to underfit.
* **Model Capacity:** matched to dataset size and complexity.

---

## Recommended Experiments

| Experiment                           | Learning Rate (LR) | Weight Decay (WD) | LDM Base CH | Rationale                                                                                                                                                           |
| ------------------------------------ | ------------------ | ----------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Safe & Stable Start**           | `3e-5`             | `0.01`            | `96`        | Most conservative and highest priority. Lower LR prevents collapse. Smaller model (`ldm_base_ch=96`) reduces complexity → easier to stabilize.                      |
| **2. Increase Model Capacity**       | `3e-5`             | `0.01`            | `128`       | Once stability is confirmed at Exp. 1, test a larger model. May improve representation. If this collapses → dataset too small or requires more regularization.      |
| **3. Add More Regularization**       | `3e-5`             | `0.05`            | `128`       | If Exp. 2 shows instability or overfitting, increase WD. Stronger regularization penalizes large weights → prevents collapse by discouraging single-mode solutions. |
| **4. Cautious Speed-Up**             | `5e-5`             | `0.01`            | `96`        | If Exp. 1 is stable but too slow, cautiously increase LR. Tests if convergence can accelerate without instability, while keeping smaller model size.                |
| **5. The Original (For Comparison)** | `1e-4`             | `0.01`            | `128`       | Your original (collapsing) configuration. Serves as baseline. If others succeed, confirms this combo was too aggressive.                                            |

---

## How to Use

Run these configurations with your `train_ldm_[task].sh` launcher.

**Example — Run Experiment 1:**

```bash
./launchers/single_runs/ldm/train_ldm_tb.sh full_train --lr 3e-5 --ldm-base-ch 96 --weight-decay 0.01
```

---

## What to Monitor

Watch **`sample_diversity/metrics`** in your logs or on W&B:

* ✅ **High & stable `pairwise_mse_mean`** → Healthy, diverse generation
* ⚠️ **Low or trending to 0** → Mode collapse

---
