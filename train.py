import torch
import numpy as np
from pathlib import Path
from nanophoto.meep_compute_fom import compute_FOM_parallele
from scipy.spatial.distance import cdist
import argparse
import yaml


class ActiveLearningDDPM:
    """Active Learning loop for DDPM-based nanophotonic design generation."""

    def __init__(self, config):
        """
        Initialize Active Learning DDPM.
        
        Args:
            config: Dictionary or path to YAML config file
        """
        if isinstance(config, (str, Path)):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)

        self.config = config

        # Extract parameters from config
        self.fom_threshold = config['active_learning']['fom_threshold']
        self.distance_threshold = config['active_learning']['distance_threshold']
        self.n_samples_per_iter = config['active_learning']['n_samples_per_iter']
        self.max_iterations = config['active_learning']['max_iterations']

        self.output_dir = Path(config['data']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # DDPM parameters
        self.ddpm_config = config.get('ddpm') or {}
        self.gen_config = config.get('generation') or {}
        self.fom_config = config.get('fom') or {}

        # Load initial training data
        initial_data_path = config['data']['initial_data_path']
        data = np.load(initial_data_path)
        self.initial_training_data = torch.from_numpy(data)
        if isinstance(self.initial_training_data, dict):
            self.initial_training_data = self.initial_training_data['data']

        # Keep track of new samples added during active learning
        self.new_samples = []  # List of tensors, one per iteration

        print(f"Loaded initial training data: {self.initial_training_data.shape}")

    @property
    def training_data(self):
        """Combine initial data with all new samples."""
        if len(self.new_samples) == 0:
            return self.initial_training_data
        return torch.cat([self.initial_training_data] + self.new_samples, dim=0)

    def train_and_generate_samples(self, iteration):
        """Train DDPM model and generate samples using photo_gen with Hydra config."""
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}: Training DDPM model and generating samples")
        print(f"Training data shape: {self.training_data.shape}")
        print(f"{'='*60}")

        # Prepare checkpoint and output paths
        checkpoint_dir = self.output_dir / f"iter_{iteration}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "checkpoint.pt"
        images_dir = checkpoint_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Use current training data (initial + all new samples)
        data = self.training_data.numpy()
        if data.ndim == 3:
            data = data[:, None, :, :]  # Add channel dimension

        # Import required modules
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from photo_gen.models.unet_fast import train_fast
        from photo_gen.models.unet import inference
        import photo_gen

        # Get photo_gen config directory
        config_dir = str(Path(photo_gen.__file__).parent / "config")

        # Build overrides
        train_size = len(data)

        overrides = [
            f"train_set_size={train_size}",
            f"n_to_generate={self.n_samples_per_iter}",
            "~evaluation",  # Disable evaluation to avoid missing dataset error
        ]
        if 'max_epochs' in self.ddpm_config:
            n_compute_steps = self.ddpm_config['max_epochs'] * train_size
            overrides.append(f"n_compute_steps={n_compute_steps}")

        # Add DDPM config overrides
        if 'batch_size' in self.ddpm_config:
            overrides.append(f"model.batch_size={self.ddpm_config['batch_size']}")
        if 'learning_rate' in self.ddpm_config:
            overrides.append(f"model.lr={self.ddpm_config['learning_rate']}")

        print(f"Hydra overrides: {overrides}")

        # Initialize Hydra and compose config
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=config_dir, version_base="1.1", job_name="active_learning"):
            cfg = compose(config_name="config", overrides=overrides)

            # Disable struct mode to allow adding n_epochs
            from omegaconf import OmegaConf
            OmegaConf.set_struct(cfg, False)

            # Calculate n_epochs from n_compute_steps
            n_epochs = int(cfg.n_compute_steps / train_size)
            cfg.model.n_epochs = n_epochs
            cfg.n_epochs = n_epochs

            # Copy training parameters from model config to top level (like main.py does)
            training_params = ['lr', 'batch_size', 'num_time_steps', 'ema_decay', 'seed']
            for param in training_params:
                if hasattr(cfg.model, param):
                    setattr(cfg, param, getattr(cfg.model, param))

            print(f"Training DDPM for {n_epochs} epochs...")

            # Train the model
            train_fast(
                data=data,
                cfg=cfg,
                checkpoint_path=checkpoint_path,
                savedir=checkpoint_dir,
                run=None
            )

            print(f"\nGenerating {self.n_samples_per_iter} samples...")

            # Generate samples
            samples = inference(
                cfg=cfg,
                checkpoint_path=checkpoint_path,
                savepath=images_dir,
                meep_eval=False
            )
        
        breakpoint()
        # Convert to torch tensor
        samples = torch.from_numpy(samples).float()
        
        print(f"Generated {len(samples)} samples with shape {samples.shape}")
        
        return samples
    
    def compute_fom_scores(self, samples):
        """Compute FOM scores using nanophoto.meep_compute_fom."""
        print(f"\nComputing FOM scores for {len(samples)} samples...")
        
        # Check if we should skip actual FOM computation
        if self.fom_config.get('skip_meep', False):
            print("Skipping MEEP FOM computation (using random scores for testing)")
            fom_scores = torch.rand(len(samples)) * 0.6  # Random scores between 0 and 0.6
            print(f"FOM scores (random) - Min: {fom_scores.min():.4f}, "
                  f"Max: {fom_scores.max():.4f}, "
                  f"Mean: {fom_scores.mean():.4f}")
            return fom_scores
        
        # Convert to numpy for FOM computation
        samples_np = samples.cpu().numpy()
        
        # Verify shape before computing FOM
        print(f"Sample shape before FOM: {samples_np.shape}")
        assert samples_np.shape[1:] == (101, 91), f"Expected shape (N, 101, 91), got {samples_np.shape}"
        
        # Compute FOM in parallel with error handling
        debug_mode = self.fom_config.get('debug', False)
        try:
            fom_scores = compute_FOM_parallele(samples_np, debug=debug_mode)
        except (ValueError, AssertionError) as e:
            print(f"Warning: FOM computation failed with error: {e}")
            print(f"Returning default FOM scores of 0.0")
            fom_scores = np.zeros(len(samples_np))
        
        # Convert back to torch tensor
        fom_scores = torch.tensor(fom_scores, dtype=torch.float32)
        
        print(f"FOM scores - Min: {fom_scores.min():.4f}, "
              f"Max: {fom_scores.max():.4f}, "
              f"Mean: {fom_scores.mean():.4f}")
        
        return fom_scores
    
    def compute_distances(self, samples):
        """Compute minimum distance from each sample to training data."""
        print(f"\nComputing distances to training data...")
        
        # Flatten samples and training data for distance computation
        samples_flat = samples.reshape(len(samples), -1).cpu().numpy()
        training_flat = self.training_data.reshape(len(self.training_data), -1).cpu().numpy()
        
        # Compute pairwise distances
        distances = cdist(samples_flat, training_flat, metric='euclidean')
        
        # Get minimum distance for each sample
        min_distances = distances.min(axis=1)
        
        print(f"Min distances - Min: {min_distances.min():.4f}, "
              f"Max: {min_distances.max():.4f}, "
              f"Mean: {min_distances.mean():.4f}")
        
        return torch.tensor(min_distances, dtype=torch.float32)
    
    def select_samples(self, samples, fom_scores, distances):
        """Select samples based on FOM threshold and distance threshold."""
        print(f"\nSelecting samples with FOM > {self.fom_threshold} "
              f"and distance > {self.distance_threshold}...")
        
        # Apply both filters
        fom_mask = fom_scores > self.fom_threshold
        distance_mask = distances > self.distance_threshold
        combined_mask = fom_mask & distance_mask
        
        selected_samples = samples[combined_mask]
        selected_fom = fom_scores[combined_mask]
        selected_distances = distances[combined_mask]
        
        print(f"Samples passing FOM threshold: {fom_mask.sum()}/{len(samples)}")
        print(f"Samples passing distance threshold: {distance_mask.sum()}/{len(samples)}")
        print(f"Samples passing both thresholds: {len(selected_samples)}/{len(samples)}")
        
        if len(selected_samples) > 0:
            print(f"Selected samples - FOM mean: {selected_fom.mean():.4f}, "
                  f"Distance mean: {selected_distances.mean():.4f}")
        
        return selected_samples
    
    def update_training_data(self, new_samples):
        """Add selected samples to training data."""
        if len(new_samples) == 0:
            print("\nNo new samples to add to training data.")
            return False
        
        print(f"\nAdding {len(new_samples)} new samples to training data...")
        print(f"Previous training data size: {len(self.training_data)}")
        
        # Add to list of new samples (kept separate from initial data)
        self.new_samples.append(new_samples)
        
        print(f"New training data size: {len(self.training_data)}")
        
        return True
    
    def load_previous_selected_samples(self):
        """Load all previously selected samples from disk."""
        all_selected = []
        iteration = 0
        while True:
            selected_path = self.output_dir / f"selected_samples_iter_{iteration}.pt"
            if not selected_path.exists():
                break
            selected = torch.load(selected_path)
            all_selected.append(selected)
            iteration += 1
        
        if all_selected:
            self.new_samples = all_selected
            print(f"Loaded {len(all_selected)} iterations of selected samples ({len(self.training_data) - len(self.initial_training_data)} total new samples)")
    
    def run(self):
        """Run the active learning loop."""
        print("\n" + "="*60)
        print("Starting Active Learning Loop")
        print("="*60)
        print(f"Configuration:")
        print(f"  FOM threshold: {self.fom_threshold}")
        print(f"  Distance threshold: {self.distance_threshold}")
        print(f"  Samples per iteration: {self.n_samples_per_iter}")
        print(f"  Max iterations: {self.max_iterations}")
        print(f"  Output directory: {self.output_dir}")
        print("="*60)
        
        # Load any previously selected samples
        self.load_previous_selected_samples()
        
        for iteration in range(self.max_iterations):
            print(f"\n{'#'*60}")
            print(f"# ITERATION {iteration + 1}/{self.max_iterations}")
            print(f"{'#'*60}")
            
            # Step 1: Train DDPM model and generate samples
            samples = self.train_and_generate_samples(iteration)
            
            # Step 2: Compute FOM scores
            fom_scores = self.compute_fom_scores(samples)
            
            # Step 3: Compute distances to training data
            distances = self.compute_distances(samples)
            
            # Step 4: Select samples
            selected_samples = self.select_samples(samples, fom_scores, distances)
            
            # Save selected samples
            if len(selected_samples) > 0:
                selected_samples_path = self.output_dir / f"selected_samples_iter_{iteration}.pt"
                torch.save(selected_samples, selected_samples_path)
                print(f"Saved {len(selected_samples)} selected samples to {selected_samples_path}")
            
            # Step 5: Update training data
            if not self.update_training_data(selected_samples):
                print("\nNo new samples selected. Stopping active learning.")
                break
            
            # Save checkpoint with only new samples
            if len(self.new_samples) > 0:
                # Concatenate all new samples from all iterations
                all_new_samples = torch.cat(self.new_samples, dim=0)
                checkpoint = {
                    'iteration': iteration,
                    'new_samples': all_new_samples,
                    'n_new_samples': len(all_new_samples),
                    'n_total_samples': len(self.training_data)
                }
                checkpoint_path = self.output_dir / f"checkpoint_iter_{iteration}.pt"
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint with {len(all_new_samples)} new samples to {checkpoint_path}")
        
        # Save only new samples at the end
        if len(self.new_samples) > 0:
            all_new_samples = torch.cat(self.new_samples, dim=0)
            new_samples_path = self.output_dir / "new_samples.pt"
            np.save(all_new_samples, new_samples_path)
            print(f"\n{'='*60}")
            print(f"Active Learning Complete!")
            print(f"New samples saved to: {new_samples_path}")
            print(f"Number of new samples: {len(all_new_samples)}")
            print(f"Initial dataset size: {len(self.initial_training_data)}")
            print(f"Final total dataset size: {len(self.training_data)}")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"Active Learning Complete!")
            print(f"No new samples were added.")
            print(f"Dataset size remains: {len(self.initial_training_data)}")
            print(f"{'='*60}")


def load_config(config_path, overrides=None):
    """Load configuration from YAML file with optional overrides."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if overrides:
        # Apply command-line overrides
        for key, value in overrides.items():
            if value is not None:
                keys = key.split('.')
                d = config
                for k in keys[:-1]:
                    d = d.setdefault(k, {})
                d[keys[-1]] = value
    
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Active Learning for DDPM-based Nanophotonic Design")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")

    args = parser.parse_args()
    config = load_config(args.config)

    al = ActiveLearningDDPM(config)
    al.run()
