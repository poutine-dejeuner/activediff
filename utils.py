from nanophoto.meep_compute_fom import compute_FOM
import multiprocessing
from tqdm import tqdm


def compute_distances(samples, training_data):
    """Compute minimum distance from each sample to training data."""
    print(f"\nComputing distances to training data...")

    # Flatten samples and training data for distance computation
    samples_flat = samples.reshape(len(samples), -1).cpu().numpy()
    training_flat = training_data.reshape(len(training_data), -1).cpu().numpy()

    # Compute pairwise distances
    distances = cdist(samples_flat, training_flat, metric='euclidean')

    # Get minimum distance for each sample
    min_distances = distances.min(axis=1)

    print(f"Min distances - Min: {min_distances.min():.4f}, "
          f"Max: {min_distances.max():.4f}, "
          f"Mean: {min_distances.mean():.4f}")

    return torch.tensor(min_distances, dtype=torch.float32)


def dist_select(samples, distances, distance_threshold):
    """Select samples based on distance threshold."""
    print(f"\nSelecting samples with distance > {distance_threshold}...")

    # Apply distance filter
    distance_mask = distances > distance_threshold
    selected_samples = samples[distance_mask]
    selected_distances = distances[distance_mask]

    print(f"Samples passing distance threshold: {distance_mask.sum()}/{len(samples)}")

    if len(selected_samples) > 0:
        print(f"Selected samples - Distance mean: {selected_distances.mean():.4f}")

    return selected_samples


def fom_select(samples, fom_scores, fom_threshold):
    """Select samples based on FOM threshold."""
    print(f"\nSelecting samples with FOM > {fom_threshold}...")

    # Apply FOM filter
    fom_mask = fom_scores > fom_threshold
    selected_samples = samples[fom_mask]
    selected_fom = fom_scores[fom_mask]

    print(f"Samples passing FOM threshold: {fom_mask.sum()}/{len(samples)}")

    if len(selected_samples) > 0:
        print(f"Selected samples - FOM mean: {selected_fom.mean():.4f}")

    return selected_samples


def compute_FOM_parallele_safe(samples_np):
    print(f"Computing FOM for {len(samples_np)} samples in parallel...")
    images = [samples_np[i] for i in range(samples_np.shape[0])]

    with multiprocessing.Pool() as pool:
        fom_scores = list(tqdm(pool.imap(compute_FOM_safe, images), total=len(images)))
    return fom_scores


def compute_FOM_safe(image):
    """Wrapper to catch exceptions in multiprocessing workers."""
    try:
        return compute_FOM(image, debug=False)
    except Exception as e:
        import traceback
        error_msg = f"Error in worker: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(error_msg)
        return 0.0  # Return default value on error


def compute_fom_scores(samples, cfg):
    """Compute FOM scores using nanophoto.meep_compute_fom."""
    print(f"\nComputing FOM scores for {len(samples)} samples...")

    # Check if we should skip actual FOM computation
    skip_meep = OmegaConf.select(cfg, 'fom.skip_meep', default=False)
    if skip_meep:
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
    try:
        fom_scores = compute_FOM_parallele_safe(samples_np)
        fom_scores = np.array(fom_scores)
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
