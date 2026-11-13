from composition.dynamics import generate_ldm_samples, run_sampler, prepare_latents
from composition.utils import parse_args, setup_run,save_results
import jax
from composition.visualize import plot_log_trajectories, plot_kappa_trajectory, plot_latent_pca

LIFT_VALUES = [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]  # Sorted for logical interpolation
NUM_ROWS = 4
NUM_COLS = len(LIFT_VALUES)

def main():
    args = parse_args()
    ae_model, ae_params, model_n, params_n, model_t, params_t, latent_size, z_channels = setup_run(args)
    latents, lift_param = prepare_latents(args, latent_size, z_channels)
    latents_normal, latents_tb = None, None
    if args.sample_images:
        rng_vis = jax.random.PRNGKey(args.seed + 1)
        print("Generating independent samples for visualization...")
        latents_normal, latents_tb = generate_ldm_samples(
            rng_vis,
            model_n, params_n,
            model_t, params_t,
            ae_model, ae_params,
            latent_size, z_channels,
            batch_size=args.batch_size,
            sampler=args.sampler,
            n_steps=args.steps,
            latent_scale_factor=args.latent_scale_factor,
            output_path=args.output_path
        )

    final_latents, kappas, log_hist_a, log_hist_b = run_sampler(
        args.sampler, latents, model_n, params_n, model_t, params_t,
        args.steps, lift_param, args.score
    )
    # 4. Save Output Images
    save_results(final_latents, ae_model, ae_params, args)

    # 5. Visual Diagnostics (Only if not sweeping, or if desired)
    if not args.sweep:
        print("Generating Diagnostic Plots...")
        base_name = f"{args.sampler.lower()}_diag"

        # Trajectories
        plot_log_trajectories(log_hist_a, log_hist_b, steps=args.steps, output_path=f"{base_name}_logs.png")

        # Kappa (Only relevant for SuperDiff samplers, PoE returns empty/dummy kappa usually)
        if args.sampler in ['Ancestral', 'Euler']:
            plot_kappa_trajectory(kappas, steps=args.steps, output_path=f"{base_name}_kappa.png")

        # Manifold Visualization (PCA / t-SNE)
        # Requires latents_normal and latents_tb to be populated
        if latents_normal is not None:
            print("Generating Manifold Visualizations...")
            plot_latent_pca(latents_normal, latents_tb, final_latents, output_path=f"{base_name}_pca.png")
            # plot_latent_tsne(latents_normal, latents_tb, final_latents, output_path=f"{base_name}_tsne.png")

if __name__ == "__main__":
    main()