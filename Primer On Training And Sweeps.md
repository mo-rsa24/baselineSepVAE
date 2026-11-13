# How to Use the Training & Sweeps Framework

This guide explains how to run individual training jobs, calculate the necessary scaling factor, and launch hyperparameter sweeps.

---

## 1. VAE Training (Single Runs)

First, train an autoencoder for your dataset. The script takes `z_channels` as an argument.

**Example:** Train a VAE for TB with `z_channels=1`

```bash
./launchers/single_runs/vae/train_ae_tb.sh 1
```

---

## 2. Calculate Latent Scale Factor (Crucial Step)

After each VAE model finishes training, you must calculate its **latent scale factor**.
This value is **essential for stable LDM training**.

**How to Run:**

```bash
python scripts/find_latent_scale.py \
    --ae_config_path "runs/ae_tb_z1_.../run_meta.json" \
    --ae_ckpt_path "runs/ae_tb_z1_.../ckpts/last.flax" \
    --data_root "/datasets/mmolefe/cleaned" \
    --task "TB"
```

**Note:** Repeat this step for every VAE you train.

---

## 3. LDM Training (Single Runs)

The LDM launchers use sensible presets for different training modes and allow full customization via command-line arguments.

### Recommended Presets by Mode

| Parameter            | full_train | overfit_one | overfit_16 | overfit_32 |
| -------------------- | ---------- | ----------- | ---------- | ---------- |
| **BATCH_PER_DEVICE** | 16         | 1           | 16         | 16         |
| **EPOCHS**           | 1000       | 1000        | 1500       | 1500       |
| **LOG_EVERY**        | 100        | 10          | 25         | 50         |
| **SAMPLE_EVERY**     | 10         | 5           | 10         | 10         |
| **USE_WANDB**        | Yes        | No          | Yes        | Yes        |

---

### How to Run

⚠️ **Important:** Before your first run, edit the script (e.g., `launchers/single_runs/ldm/train_ldm_tb.sh`) and update the `AE_CKPT_PATH` to point to your trained VAE.

* **Run a full training job with default presets:**

```bash
./launchers/single_runs/ldm/train_ldm_tb.sh full_train
```

* **Run an overfitting job on 16 samples**
  (uses recommended presets: 1500 epochs, log every 25 steps, etc.):

```bash
./launchers/single_runs/ldm/train_ldm_tb.sh overfit_16
```

* **Override presets for a custom experiment**
  (e.g., overfit on 32 samples but reduce epochs and disable W&B logging):

```bash
./launchers/single_runs/ldm/train_ldm_tb.sh overfit_32 --epochs 500 --no-wandb
```

---

## 4. W&B Sweeps

Sweeps automate running many experiments to find optimal models and hyperparameters.

**Step 1: Initialize the Sweep**
Choose the `.yaml` configuration for your desired experiment.

Example — Sweep LDM Hyperparameters for TB (Full Training):

```bash
wandb sweep sweeps/ldm/hyperparams/tb_full_train.yaml
```

**Step 2: Run the Agent**
Use the Sweep ID from the previous command to start the agent on your cluster’s login node.

```bash
wandb agent <your_username>/<your_project>/<sweep_id>
```

---