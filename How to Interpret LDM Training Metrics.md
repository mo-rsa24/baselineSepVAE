# How to Interpret LDM Training Metrics

This guide explains how to interpret the key statistics logged during the training and sampling of your Latent Diffusion Model. Understanding these numbers is crucial for diagnosing issues like training instability, slow convergence, and mode collapse.

---

1. `train/metrics`

These metrics are logged at every `log_every` step and give you a real-time view of the model's learning process.

| Metric            | What it Measures                                                                                                         | Good Sign                                                                                                                | Bad Sign (and What it Means)                                                                                                                                                                                                                                                                  |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **loss**          | The value of the denoising loss function. It measures how well the model predicts the noise that was added to an image.  | The loss should **steadily decrease** over time, especially in the early stages of training. It will eventually plateau. | - **NaN** or **inf**: Training has collapsed, likely due to a learning rate that is too high (exploding gradients). <br> - **Stays high / fluctuates wildly**: Model is not learning effectively. Could be caused by too low learning rate, poor data quality, or architectural issues.       |
| **grad_norm**     | The overall magnitude (L2 norm) of the gradients before they are clipped. It indicates how large the weight updates are. | A stable, non-zero value. Normal for it to be higher at the start of training and decrease over time.                    | - **Very high (> 100)**: Exploding gradients. Updates are too large → instability. `grad_clip` prevents a crash, but the root cause (likely high learning rate) must be fixed. <br> - **Very low (< 1e-6)**: Vanishing gradients. Model isn’t learning; gradients too weak to update weights. |
| **learning_rate** | The current learning rate being used by the optimizer.                                                                   | Should match the value you set. If using a scheduler (e.g., warmup/decay), you’ll see it change over time.               | N/A                                                                                                                                                                                                                                                                                           |

---

2. `sample_diversity/metrics`

This block is the most important diagnostic for **mode collapse**. It is calculated during sampling by measuring the pairwise Mean Squared Error (MSE) between all generated latents in a batch.

| Metric                | What it Measures                                                                 | Good Sign (Healthy Model)                                                                                                      | Bad Sign (Mode Collapse)                                                                                                                   |
| --------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **pairwise_mse_mean** | Average pixel-wise difference between all pairs of generated samples in a batch. | A high and stable value. For normalized latents, > **0.1** is good. Higher = more varied outputs from different noise vectors. | Very low (< **0.01**) and trending toward zero. Clear sign of mode collapse → sampler produces the same output regardless of random noise. |
| **pairwise_mse_std**  | Standard deviation of the pairwise MSE values.                                   | Healthy, non-zero value.                                                                                                       | Trends toward zero during mode collapse.                                                                                                   |
| **pairwise_mse_min**  | The smallest MSE between any two samples.                                        | Should be significantly > 0.                                                                                                   | Extremely close to zero.                                                                                                                   |
| **pairwise_mse_max**  | The largest MSE between any two samples.                                         | Should be a high value.                                                                                                        | Becomes very small → no meaningful diversity between samples.                                                                              |

---

3. `final_latent_stats`

These stats describe the distribution of the final latent vectors (`z`) produced by the sampler, right before they are passed to the VAE decoder.

| Metric        | What it Measures                                                  | Good Sign                                                                                                   | Bad Sign (and What it Means)                                                                              |
| ------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **mean**      | Average value across all dimensions of latent vectors in a batch. | Close to **0.0**.                                                                                           | Large positive/negative shift → latents not centered. Can lead to dull or discolored images.              |
| **std**       | Standard deviation of the latent vectors.                         | Close to target std = **1.0 / latent_scale_factor**. <br> Example: scale factor = 1.86 → target std ≈ 0.53. | - **Very low**: Latents too similar (mode collapse). <br> - **Very high**: Sampler unstable or too noisy. |
| **min / max** | Minimum and maximum values in the latent vectors.                 | Within reasonable range (e.g., [-3.0, 3.0]).                                                                | Extreme values (±20) suggest instability in sampling → artifacts or blank images.                         |

---
