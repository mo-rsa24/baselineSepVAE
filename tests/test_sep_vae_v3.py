import jax
import jax.numpy as jnp

from losses.sep_vae_losses import CTRAdversary, FactorDiscriminator
from losses.sep_vae_v3_losses import SepVAEV3LossConfig, sepvae_v3_loss
from models.sep_vae_v2 import SepVAEV2
from models.sep_vae_v3 import SepVAEV3


IMG_SIZE = 128


def _make_circle_mask(radius: float, center_x: float, center_y: float):
    yy, xx = jnp.meshgrid(
        jnp.linspace(0.0, 1.0, IMG_SIZE),
        jnp.linspace(0.0, 1.0, IMG_SIZE),
        indexing="ij",
    )
    return (((xx - center_x) ** 2 + (yy - center_y) ** 2) <= radius ** 2).astype(jnp.float32)


def _make_v3_batch():
    x_norm = jnp.full((1, IMG_SIZE, IMG_SIZE, 1), -0.2, dtype=jnp.float32)
    x_cardio = jnp.full((1, IMG_SIZE, IMG_SIZE, 1), 0.25, dtype=jnp.float32)
    heart_mask = jnp.stack(
        [
            _make_circle_mask(0.14, 0.5, 0.52),
            _make_circle_mask(0.20, 0.52, 0.52),
        ],
        axis=0,
    )
    return {
        "x_norm": x_norm,
        "x_disease1": x_cardio,
        "heart_mask": heart_mask,
        "ctr": jnp.array([0.46, 0.61], dtype=jnp.float32),
        "has_mask": jnp.ones((2,), dtype=jnp.float32),
        "disease_labels": jnp.array([0, 1], dtype=jnp.int32),
    }


def test_sepvae_v3_encode_decode_shapes():
    model = SepVAEV3(
        z_channels_common=4,
        z_channels_heart=4,
        decoder_res_blocks=1,
        feature_channels=64,
    )
    key = jax.random.PRNGKey(0)
    batch = _make_v3_batch()
    x = jnp.concatenate([batch["x_norm"], batch["x_disease1"]], axis=0)[:1]
    heart_mask = batch["heart_mask"][:1]

    variables = model.init(key, x, heart_mask, key=key)
    encoded = model.apply(variables, x, heart_mask, method=model.encode)

    assert encoded["common"][0].shape[0] == 1
    assert encoded["heart"][0].shape[0] == 1
    assert encoded["ctr_pred"].shape == (1,)

    x_hat, alpha, aux = model.apply(
        variables,
        encoded["common"][0],
        encoded["heart"][0],
        encoded["ctr_pred"],
        skip_common=encoded["skip_common"],
        skip_heart=encoded["skip_heart"],
        method=model.decode,
    )

    assert x_hat.shape == (1, IMG_SIZE, IMG_SIZE, 1)
    assert alpha.shape == (1, IMG_SIZE, IMG_SIZE, 1)
    assert aux["x_common"].shape == (1, IMG_SIZE, IMG_SIZE, 1)
    assert aux["x_heart"].shape == (1, IMG_SIZE, IMG_SIZE, 1)


def test_sepvae_v3_loss_path_keeps_heart_branch_active_for_both_classes():
    model = SepVAEV3(
        z_channels_common=4,
        z_channels_heart=4,
        decoder_res_blocks=1,
        feature_channels=64,
    )
    key = jax.random.PRNGKey(1)
    batch = _make_v3_batch()
    x = jnp.concatenate([batch["x_norm"], batch["x_disease1"]], axis=0)

    variables = model.init(key, x, batch["heart_mask"], key=key)
    cfg = SepVAEV3LossConfig(
        weight_rec=1.0,
        weight_kl_common=1e-4,
        weight_kl_heart=1e-4,
        weight_alpha=1.0,
        weight_common_out=1.0,
        weight_heart_in=1.0,
        weight_ctr=1.0,
    )
    total_loss, logs, _, _, x_hat = sepvae_v3_loss(
        model, variables["params"], batch, key, cfg
    )

    assert jnp.isfinite(total_loss)
    assert x_hat.shape == (2, IMG_SIZE, IMG_SIZE, 1)
    assert float(logs["metrics/z_heart_norm_normal"]) > 0.0
    assert float(logs["metrics/z_heart_norm_cardio"]) > 0.0


def test_sepvae_v3_loss_defaults_backward_compat():
    """Default SepVAEV3LossConfig has all new fields at off-values."""
    cfg = SepVAEV3LossConfig()
    assert cfg.weight_mi_factor == 0.0
    assert cfg.weight_heart_supcon == 0.0
    assert cfg.weight_ctr_adv == 0.0
    assert cfg.use_conditional_kl_heart == False
    assert cfg.sigma_inactive == 1.0

    model = SepVAEV3(
        z_channels_common=4,
        z_channels_heart=4,
        decoder_res_blocks=1,
        feature_channels=64,
    )
    key = jax.random.PRNGKey(4)
    batch = _make_v3_batch()
    x = jnp.concatenate([batch["x_norm"], batch["x_disease1"]], axis=0)
    variables = model.init(key, x, batch["heart_mask"], key=key)
    # No disc or adversary passed — all new losses must silently zero out
    total_loss, logs, _, _, _ = sepvae_v3_loss(model, variables["params"], batch, key, cfg)
    assert jnp.isfinite(total_loss)
    assert float(logs["loss/mi_factor"]) == 0.0
    assert float(logs["loss/heart_supcon"]) == 0.0
    assert float(logs["loss/ctr_adv_confuse"]) == 0.0


def test_sepvae_v3_loss_with_separation_mechanisms():
    """V3 loss path with all separation mechanisms active."""
    z_ch = 4
    model = SepVAEV3(
        z_channels_common=z_ch,
        z_channels_heart=z_ch,
        decoder_res_blocks=1,
        feature_channels=64,
    )
    key = jax.random.PRNGKey(5)
    batch = _make_v3_batch()
    x = jnp.concatenate([batch["x_norm"], batch["x_disease1"]], axis=0)
    variables = model.init(key, x, batch["heart_mask"], key=key)

    disc = FactorDiscriminator(hidden_dim=64)
    disc_vars = disc.init(key, jnp.ones((1, z_ch + z_ch)))

    adv = CTRAdversary(hidden_dim=64)
    adv_vars = adv.init(key, jnp.ones((1, z_ch)))

    cfg = SepVAEV3LossConfig(
        weight_mi_factor=1.0,
        weight_heart_supcon=0.05,
        supcon_temperature=0.1,
        use_conditional_kl_heart=True,
        sigma_inactive=0.3,
        weight_ctr_adv=0.1,
    )
    total_loss, logs, z_c, z_h, x_hat = sepvae_v3_loss(
        model, variables["params"], batch, key, cfg,
        disc_params=disc_vars["params"], discriminator=disc,
        ctr_adv_params=adv_vars["params"], ctr_adversary=adv,
    )
    assert jnp.isfinite(total_loss)
    assert x_hat.shape == (2, IMG_SIZE, IMG_SIZE, 1)
    assert "loss/mi_factor" in logs
    assert "loss/heart_supcon" in logs
    assert "loss/ctr_adv_confuse" in logs
    assert jnp.isfinite(logs["loss/mi_factor"])
    assert jnp.isfinite(logs["loss/heart_supcon"])
    assert jnp.isfinite(logs["loss/ctr_adv_confuse"])


def test_sepvae_v2_forward_still_initializes():
    model = SepVAEV2(
        z_channels_common=4,
        z_channels_disease=4,
        decoder_res_blocks=1,
        use_bbox_cross_attn=False,
        heart_out_zc=True,
        heart_in_zd=True,
    )
    key = jax.random.PRNGKey(2)
    batch = _make_v3_batch()
    x = jnp.concatenate([batch["x_norm"], batch["x_disease1"]], axis=0)
    labels = batch["disease_labels"]

    variables = model.init(key, x, labels, key=key, heart_mask=batch["heart_mask"])
    x_rec, latents_dict, z_c, z_d, ctr_pred = model.apply(
        variables,
        x,
        labels,
        key=key,
        heart_mask=batch["heart_mask"],
    )

    assert x_rec.shape == (2, IMG_SIZE, IMG_SIZE, 1)
    assert latents_dict["common"][0].shape[0] == 2
    assert latents_dict["cardiomegaly"][0].shape[0] == 2
    assert z_c.shape[0] == 2
    assert z_d.shape[0] == 2
    assert ctr_pred.shape == (2,)
