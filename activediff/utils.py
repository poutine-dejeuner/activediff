from pathlib import Path
import numpy as np
import torch
from torch import cdist
from activediff.meep_compute_fom import compute_FOM
import multiprocessing
from tqdm import tqdm
from omegaconf import OmegaConf

def binarisation(images: torch.Tensor) -> torch.Tensor:
    """Computes the average binarisation of a batch of images. The
    binarization of an image is the mean of min(abs(z), abs(1-z)) for each
    pixel, where z is the pixel value."""
    min_values = torch.min(torch.abs(images), torch.abs(1 - images))
    binarisation_scores = min_values.mean(dim=[1, 2])
    return binarisation_scores

def compute_distances(samples: torch.Tensor, training_data: torch.Tensor) -> torch.Tensor:
    """Compute minimum distance from each sample to training data."""
    print(f"\nComputing distances to training data...")

    samples_flat = samples.reshape(len(samples), -1)
    training_flat = training_data.reshape(len(training_data), -1)

    distances = cdist(samples_flat, training_flat, p=2)

    min_distances = torch.min(distances, dim=1).values  # .values to get tensor from named tuple

    print(f"Min distances - Min: {min_distances.min():.4f}, "
          f"Max: {min_distances.max():.4f}, "
          f"Mean: {min_distances.mean():.4f}")

    return min_distances


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


def filter_similar_samples(samples: torch.tensor, fom_scores:torch.Tensor| np.ndarray, distance_threshold:float):
    """Filter samples by keeping only the best FOM for similar samples.
    
    Args:
        samples: Tensor of shape (N, H, W)
        fom_scores: Tensor of FOM scores for each sample
        distance_threshold: Distance threshold below which samples are considered similar
    
    Returns:
        Filtered samples and their FOM scores
    """
    if isinstance(fom_scores, np.ndarray):
        fom_scores = torch.from_numpy(fom_scores)
    print(f"\nFiltering similar samples with distance threshold {distance_threshold}...")
    
    # Flatten samples for distance computation
    samples_flat = samples.reshape(len(samples), -1)
    
    # Compute pairwise distances between samples
    pairwise_distances = cdist(samples_flat, samples_flat, p=2)
    
    # Sort samples by FOM score (descending)
    sorted_indices = torch.argsort(fom_scores, descending=True)
    
    # Keep track of which samples to keep
    keep_mask = torch.ones(len(samples), dtype=torch.bool)
    
    # Iterate through sorted samples
    for _, idx_i in enumerate(sorted_indices):
        if not keep_mask[idx_i]:
            continue
            
        # Find similar samples (distance below threshold)
        similar_mask = pairwise_distances[idx_i] < distance_threshold
        
        # Mark lower-FOM similar samples for removal
        for idx_j in range(len(samples)):
            if idx_j != idx_i and similar_mask[idx_j] and keep_mask[idx_j]:
                # If scores are very close, keep both; otherwise keep the better one
                if fom_scores[idx_j] < fom_scores[idx_i]:
                    keep_mask[idx_j] = False
    
    filtered_samples = samples[keep_mask]
    filtered_fom = fom_scores[keep_mask]
    
    print(f"Samples after similarity filtering: {keep_mask.sum()}/{len(samples)}")
    if len(filtered_samples) > 0:
        print(f"Filtered samples - FOM mean: {filtered_fom.mean():.4f}")
    
    return filtered_samples, filtered_fom


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

    return selected_samples, selected_fom


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


def set_seed(seed: int = 42):
    if seed == -1:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    # random.seed(seed)


def fom_dist_select(imagespath, fom_threshold=0.48, dist_threshold=10):
    import activediff

    imagesdir = imagespath.parent
    training_data_path = Path(activediff.__file__).parent.parent / "data/imagesnorm.npy"
    
    training_data = torch.from_numpy(np.load(training_data_path))

    samples = torch.from_numpy(np.load(imagespath))
    distances = compute_distances(samples.to(float), training_data.to(float))
    samples_after_dist = dist_select(samples, distances, dist_threshold)

    fompath = imagesdir / "fom.npy"
    if fompath.exists():
        fom_scores = torch.from_numpy(np.load(fompath))
    else:
        fom_scores = compute_FOM_parallele_safe(samples_after_dist)
        np.save(fompath, fom_scores)

    fom_scores = np.array(fom_scores)
    samples_after_fom, fom_scores = filter_similar_samples(samples_after_dist, fom_scores, dist_threshold)
    selected_samples, selected_fom = fom_select(samples_after_fom, fom_scores, fom_threshold)
    selected_dist = compute_distances(selected_samples.to(float), training_data.to(float))
    imagesdir = Path(imagespath).parent
    np.save(imagesdir / 'selected_images.npy', selected_samples)
    np.save(imagesdir / 'selecded_fom.npy', selected_fom)
    print(selected_dist.mean())
    print(selected_fom.mean())

if __name__ == "__main__":
    imagespath = Path("active_learning_output/iter_0/images/images.npy")
    fom_dist_select(imagespath)
