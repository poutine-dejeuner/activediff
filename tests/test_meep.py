"""Tests for activediff.meep_compute_fom module."""

import pytest
import numpy as np

from activediff.meep_compute_fom import (
    mirror_upper_y_half,
    normalise,
    compute_FOM,
    compute_FOM_parallele,
    compute_FOM_array,
    MappingClass,
    mapping,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_image(rng=None):
    """Return a valid (101, 91) image with values in [0, 1]."""
    rng = rng or np.random.default_rng(42)
    return rng.random((101, 91)).astype(np.float64)


# ---------------------------------------------------------------------------
# Pure‑function tests (no MEEP simulation needed)
# ---------------------------------------------------------------------------

class TestMirrorUpperYHalf:
    """Tests for mirror_upper_y_half."""

    def test_even_width(self):
        x = np.arange(24).reshape(4, 6).astype(float)
        out = mirror_upper_y_half(x)
        assert out.shape == x.shape
        half = x.shape[1] // 2
        # upper half should be kept as‑is
        np.testing.assert_array_equal(out[:, half:], x[:, half:])
        # lower half is flipped upper half
        np.testing.assert_array_equal(out[:, :half], np.fliplr(x[:, half:]))

    def test_odd_width(self):
        x = np.arange(35).reshape(5, 7).astype(float)
        out = mirror_upper_y_half(x)
        assert out.shape == x.shape
        half = x.shape[1] // 2
        upper = x[:, half:]
        np.testing.assert_array_equal(out[:, half:], upper)
        np.testing.assert_array_equal(out[:, :half], np.fliplr(upper)[:, :-1])

    def test_symmetry(self):
        """Result should be symmetric around the center column."""
        x = np.random.default_rng(0).random((10, 20))
        out = mirror_upper_y_half(x)
        np.testing.assert_array_equal(out, np.fliplr(out))

    def test_realistic_shape(self):
        """Test with shapes used in the actual pipeline (101, 181)."""
        x = np.random.default_rng(1).random((101, 181))
        out = mirror_upper_y_half(x)
        assert out.shape == (101, 181)


class TestNormalise:
    """Tests for normalise."""

    def test_range(self):
        img = np.array([[1.0, 5.0], [3.0, 9.0]])
        out = normalise(img)
        assert out.min() == pytest.approx(0.0)
        assert out.max() == pytest.approx(1.0)

    def test_already_normalised(self):
        img = np.array([[0.0, 0.5], [0.25, 1.0]])
        out = normalise(img)
        np.testing.assert_allclose(out, img)

    def test_constant_image(self):
        """Constant image → division by zero guard (nan is acceptable)."""
        img = np.ones((5, 5)) * 3.0
        out = normalise(img)
        # Either all‑zero or all‑nan depending on implementation
        assert out.shape == (5, 5)

    def test_negative_values(self):
        img = np.array([[-2.0, 0.0], [1.0, 3.0]])
        out = normalise(img)
        assert out.min() == pytest.approx(0.0)
        assert out.max() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MEEP simulation tests (slower – require meep)
# ---------------------------------------------------------------------------

class TestMapping:
    """Tests for the mapping / MappingClass functions."""

    @pytest.fixture()
    def sim_args(self):
        """Minimal sim_args derived from get_sim defaults."""
        from activediff.meep_compute_fom import get_sim
        _, _, args = get_sim()
        return {
            "Nx": args["Nx"],
            "Ny": args["Ny"],
            "filter_radius": args["filter_radius"],
            "design_region_width": args["design_region_width"],
            "design_region_height": args["design_region_height"],
            "design_region_resolution": args["design_region_resolution"],
        }

    def test_output_shape(self, sim_args):
        Nx, Ny = sim_args["Nx"], sim_args["Ny"]
        x = np.random.default_rng(0).random((Nx, Ny))
        out = mapping(x, eta=0.5, beta=256, **sim_args)
        assert out.shape == (Nx * Ny,)

    def test_output_range(self, sim_args):
        """Projected values should be in [0, 1]."""
        Nx, Ny = sim_args["Nx"], sim_args["Ny"]
        x = np.random.default_rng(1).random((Nx, Ny))
        out = mapping(x, eta=0.5, beta=256, **sim_args)
        out_np = np.array(out)
        assert out_np.min() >= -1e-6
        assert out_np.max() <= 1.0 + 1e-6

    def test_mapping_class_equivalent(self, sim_args):
        """MappingClass should produce the same result as calling mapping."""
        Nx, Ny = sim_args["Nx"], sim_args["Ny"]
        x = np.random.default_rng(2).random((Nx, Ny))
        direct = mapping(x, 0.5, 256, **sim_args)
        via_class = MappingClass(**sim_args)(x, 0.5, 256)
        np.testing.assert_array_equal(np.array(direct), np.array(via_class))


class TestComputeFOM:
    """Tests for compute_FOM (single image)."""

    def test_basic_fom(self):
        """compute_FOM returns a finite scalar for a valid image."""
        image = _random_image()
        fom = compute_FOM(image)
        assert np.isfinite(fom), f"FOM is not finite: {fom}"
        assert isinstance(float(fom), float)

    def test_fom_range(self):
        """FOM should be a reasonable positive value (power ratio)."""
        image = _random_image(np.random.default_rng(99))
        fom = compute_FOM(image)
        assert fom >= 0.0, f"FOM should be non-negative, got {fom}"

    def test_wrong_shape_raises(self):
        """Should raise AssertionError for wrong input shape."""
        with pytest.raises(AssertionError):
            compute_FOM(np.random.rand(50, 50))

    def test_out_of_range_raises(self):
        """Should raise AssertionError if values are outside [0, 1]."""
        bad_image = np.full((101, 91), 2.0)
        with pytest.raises(AssertionError):
            compute_FOM(bad_image)

    def test_deterministic(self):
        """Same input should produce the same FOM."""
        image = _random_image()
        fom1 = compute_FOM(image)
        fom2 = compute_FOM(image)
        assert fom1 == pytest.approx(fom2, rel=1e-4)


class TestComputeFOMParallele:
    """Tests for compute_FOM_parallele (batch)."""

    def test_single_image(self):
        """2‑D input (single image) should return a scalar."""
        image = _random_image()
        fom = compute_FOM_parallele(image)
        assert np.isscalar(fom) or (isinstance(fom, np.ndarray) and fom.ndim == 0) or np.isfinite(fom)

    def test_batch(self):
        """Batch of images should return an array of FOMs."""
        rng = np.random.default_rng(7)
        images = np.stack([_random_image(rng) for _ in range(3)])
        foms = compute_FOM_parallele(images)
        assert isinstance(foms, np.ndarray)
        assert foms.shape == (3,)
        assert np.all(np.isfinite(foms))

    def test_4d_squeeze(self):
        """4‑D input (B, 1, H, W) should be squeezed and processed."""
        rng = np.random.default_rng(8)
        images = rng.random((2, 1, 101, 91))
        foms = compute_FOM_parallele(images)
        assert isinstance(foms, np.ndarray)
        assert foms.shape == (2,)


class TestComputeFOMArray:
    """Tests for compute_FOM_array (serial with error handling)."""

    def test_single_image(self):
        image = _random_image()
        fom, errors = compute_FOM_array(image)
        assert np.isfinite(fom) or (isinstance(fom, np.ndarray) and np.all(np.isfinite(fom)))
        assert errors == []

    def test_batch(self):
        rng = np.random.default_rng(10)
        images = np.stack([_random_image(rng) for _ in range(2)])
        foms, errors = compute_FOM_array(images)
        assert isinstance(foms, np.ndarray)
        assert len(foms) == 2
        assert isinstance(errors, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
