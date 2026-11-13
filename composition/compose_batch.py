import jax
import numpy as np
from operator import itemgetter
from composition.utils import parse_args, decode_image, save_image_grid, setup_config, save_filmstrip
from composition.dynamics import generate_batched_baseline, run_composition_batched
from composition.output_manager import OutputManager
from composition.visualize import (
    plot_x0_manifold_diagnostics,
    plot_dynamics_suite,
    plot_geometry_suite,
    plot_logs_and_kappa, plot_disease_difference
)


def main():
    # 1. Setup & Config
    args = parse_args()
    manager = OutputManager(args.output_path)  # Initialize Output Manager
    ae_model, ae_params, normal, tb = setup_config(args)

    model_n, params_n = itemgetter("ldm_model", "params")(normal)
    model_t, params_t = itemgetter("ldm_model", "params")(tb)
    lsize, zch = itemgetter("latent_size", "z_channels")(normal)

    # 2. Generate Baselines (If needed for Diagnostics)
    traj_n, traj_t = None, None
    latents_n, latents_t = None, None

    if args.sample_images:
        print("--- Phase 1: Generating Baselines ---")
        # Note: 'return_trajectory=True' captures pred_x0 for rigorous analysis
        latents_n, traj_n = generate_batched_baseline(
            jax.random.PRNGKey(args.seed + 100), normal,
            args.num_samples, args.batch_size, sampler=args.sampler, n_steps=args.steps,
            latent_scale_factor=args.latent_scale_factor, return_trajectory=True
        )
        latents_t, traj_t = generate_batched_baseline(
            jax.random.PRNGKey(args.seed + 200), tb,
            args.num_samples, args.batch_size, sampler=args.sampler, n_steps=args.steps,
            latent_scale_factor=args.latent_scale_factor, return_trajectory=True
        )

        # Save Reference Images
        vis_n = decode_image(ae_model, ae_params, latents_n[:args.num_visual_samples], args.latent_scale_factor)
        vis_t = decode_image(ae_model, ae_params, latents_t[:args.num_visual_samples], args.latent_scale_factor)
        save_image_grid(np.array(vis_n), manager.get_path("images", "ref_normal.png"))
        save_image_grid(np.array(vis_t), manager.get_path("images", "ref_tb.png"))

    # 3. Run Composition
    print(f"--- Phase 2: Running Composition ({args.sampler}) ---")
    final_latents, kappas, logs_a, logs_b, traj_comp = run_composition_batched(
        jax.random.PRNGKey(args.seed), args,
        model_n, params_n, model_t, params_t, (lsize, zch)
    )

    # 4. Save Final Results
    print("--- Phase 3: Saving Results ---")
    vis_comp = decode_image(ae_model, ae_params, final_latents[:args.num_visual_samples], args.latent_scale_factor)
    save_image_grid(np.array(vis_comp), manager.get_path("images", "composition_result.png"))

    # 5. Run Diagnostics (The "Three Angles")
    if not args.sweep and latents_n is not None:
        print("--- Phase 4: Running Diagnostic Suite ---")

        # Angle 1: Semantic Validity (Is it a real lung?)
        if traj_n is not None:
            plot_x0_manifold_diagnostics(traj_n, traj_t, traj_comp, manager.get_path("validity", "diag"))
            if 'vis_n' in locals() and 'vis_comp' in locals():
                n_img = np.array(vis_n)[0]
                c_img = np.array(vis_comp)[0]
                plot_disease_difference(
                    n_img,
                    c_img,
                    manager.get_path("validity", "level1_difference_map.png")
                )
        # Angle 2: Temporal Dynamics (Process)
        if traj_n is not None:
            plot_dynamics_suite(
                traj_n, traj_t, traj_comp, manager,
                ae_model, ae_params, args.latent_scale_factor
            )
            # Filmstrip
            save_filmstrip(
                ae_model, ae_params, traj_comp,
                manager.get_path("images", "composition_filmstrip.png"),
                args.latent_scale_factor
            )
            # Logs
            plot_logs_and_kappa(logs_a, logs_b, kappas, args.steps, manager)

        # Angle 3: Static Geometry (Distribution)
        plot_geometry_suite(latents_n, latents_t, final_latents, manager)

    print(f"\nExperiment Complete. All outputs saved to: {manager.base_dir}")


if __name__ == "__main__":
    main()