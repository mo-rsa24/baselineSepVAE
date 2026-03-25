# Head Drift from `e1eca19`

This note documents the meaningful changes between the known-good SepVAE
curriculum commit

- `e1eca19bf6670a9ec495d7f78a63b9d283a73a3b`

and the current `main` head

- `97f678113b2770417db1c126b839905103f1766c`

The successful `d0 -> d1 -> d2 -> d4` run sequence was code-stable: all four
successful runs were launched from `e1eca19`. That means any reproduction gap
today comes from two sources:

1. code drift since `e1eca19`
2. curriculum / hyperparameter drift

## Summary

The current head is not a minor variant of the known-good code. The main drift
lands in the exact files that control latent geometry and reconstruction:

- `run/train_sep_vae.py`
- `models/sep_vae_v2.py`
- `losses/sep_vae_losses.py`

These changes are large enough that a failed current run cannot be interpreted
as a clean reproduction attempt of the successful `e1eca19` setup.

## Trainer Drift

File:

- `run/train_sep_vae.py`

Main changes since `e1eca19`:

- The trainer moved from two optimizers to three optimizers.
- A PatchGAN discriminator path is now created in the main training script.
- PatchGAN state is instantiated even in phases where `weight_gan=0.0`.
- Resume logic now supports partial parameter merging across architecture drift.
- Checkpoints now carry extra discriminator state.

Why this matters:

- Resume semantics are no longer the same as the successful run.
- A resumed checkpoint from an older architecture can now be partially merged
  into a newer model instead of failing fast.
- This is useful for development, but it weakens reproducibility because the
  resumed model is no longer guaranteed to be architecturally identical to the
  original run.

## Model Drift

File:

- `models/sep_vae_v2.py`

Main changes since `e1eca19`:

- The layer-4 branches now stay at `16x16` throughout instead of downsampling
  and resizing back up.
- The decoder channel schedule was widened.
- Decoder squeeze-excitation settings changed.
- A new decoder self-attention block was added at `32x32`.

Observed consequence:

- The parameter count changed between the successful and current runs.
- Successful run log: `87,241,351`
- Current curriculum log: `88,555,719`

Why this matters:

- D1/D2/D4 checkpoints from `e1eca19` are not checkpoints of the same model as
  the current head.
- Latent manifolds and reconstructions can change even under identical CLI
  arguments because the encoder and decoder are no longer the same network.

## Loss Drift

File:

- `losses/sep_vae_losses.py`

Main changes since `e1eca19`:

- The perceptual loss path was modified.
- The preferred CheSS layers changed from the older later-stage emphasis to a
  layer selection that prefers layers `1-3` when available.
- Optional GAN and total-variation terms were added to the VAE loss function.

Why this matters:

- Even when `weight_gan=0.0` and `weight_tv=0.0`, the training and resume code
  around those paths is no longer the same as in the successful run.
- D4 is especially sensitive because perceptual supervision is one of the main
  phase objectives.

## Curriculum Drift

The current curriculum also diverged from the successful run recipe:

- D1 successful run:
  - `epochs=20`
  - `batch_size=16`
  - `lr_vae=2e-4`
- D1 current curriculum:
  - `epochs=30`
  - `batch_size=16`
  - `lr_vae=1e-4`

- D2 successful run:
  - resumed from D1 final
  - `epochs=120`
  - `batch_size=16`
  - `weight_mi_factor=1.0`
- D2 current curriculum:
  - resumed from D1 final
  - target epoch `65`, not `120`

- D4 successful run:
  - resumed from D2 final
  - `epochs=160`
  - `batch_size=16`
  - `weight_perceptual=0.3`
- D4 current curriculum:
  - resumed from D2 final
  - target epoch `85`, not `160`
  - `batch_size=12`
  - `weight_perceptual=0.05`

Why this matters:

- The current run is not a close reproduction even before considering code
  drift.
- The largest curriculum differences are exactly in D2 and D4, which are the
  phases most responsible for latent separation and perceptual refinement.

## Practical Conclusion

The successful `d0 -> d1 -> d2 -> d4` result should be treated as the outcome
of:

- the `e1eca19` codebase
- the longer D2 schedule
- the stronger D4 perceptual schedule

The clean reproduction path is therefore:

1. run from `e1eca19`
2. use the original successful hyperparameters
3. compare only after that against current `main`

Anything else mixes code drift and curriculum drift into the same experiment.
