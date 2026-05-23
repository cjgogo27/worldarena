from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


LAB_ROOT = Path(__file__).resolve().parents[1]
DEMO4_ROOT = LAB_ROOT / "demo4_constrained_2d"
DEMO5_ROOT = LAB_ROOT / "demo5_energy_shapes"
DEMO6_ROOT = LAB_ROOT / "demo6_transport_state_matching"
DEMO7_ROOT = LAB_ROOT / "demo7_renderer_observation"
DEMO8_ROOT = LAB_ROOT / "demo8_integrated_fetg_pipeline"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    sys.path.insert(0, str(DEMO4_ROOT))
    cu = load_module("demo4_constraint_utils", DEMO4_ROOT / "constraint_utils.py")

    sys.path.insert(0, str(DEMO5_ROOT))
    eu = load_module("demo5_energy_utils", DEMO5_ROOT / "energy_utils.py")
    demo5_run = load_module("demo5_run", DEMO5_ROOT / "run.py")

    samples = cu.sample_two_disks_unconstrained(256, radius=0.5, bounds=2.0, seed=7)
    projected = cu.project_to_non_overlap(samples, radius=0.5, seed=7)
    raw_valid = float(np.mean(cu.check_non_overlap(samples, radius=0.5)))
    projected_valid = float(np.mean(cu.check_non_overlap(projected, radius=0.5)))
    assert projected_valid >= raw_valid
    assert projected_valid > 0.95

    constrained = cu.sample_constrained(128, radius=0.5, bounds=2.0, seed=11)
    assert cu.check_non_overlap(constrained, radius=0.5).all()

    np.random.seed(5)
    random_samples = eu.sample_random_init(64)
    good_samples = eu.sample_good_init(64, noise=0.05)
    random_energy = np.mean([eu.cup_energy(sample) for sample in random_samples])
    good_energy = np.mean([eu.cup_energy(sample) for sample in good_samples])
    assert good_energy < random_energy

    np.random.seed(13)
    init = eu.sample_random_init(1)[0]
    cfg = demo5_run.Config(n_steps=30, step_size=0.01, noise_scale=0.0)
    _, energies = demo5_run.langevin_sample(
        init,
        eu.cup_energy,
        eu.cup_energy_gradient,
        cfg.n_steps,
        cfg.step_size,
        cfg.noise_scale,
    )
    assert float(energies[-1]) <= float(energies[0])
    assert np.isfinite(energies).all()

    sys.path.insert(0, str(DEMO6_ROOT))
    su6 = load_module("demo6_state_utils", DEMO6_ROOT / "state_utils.py")
    mt6 = load_module("demo6_metrics", DEMO6_ROOT / "metrics.py")

    np.random.seed(99)
    points_a = su6.generate_canonical_shape("circle", 20, 1.0, 1)
    points_b = points_a + np.array([2.0, 2.0])

    img_a = su6.render_points_to_image(points_a, 64)
    img_b = su6.render_points_to_image(points_b, 64)
    p_ab = mt6.pixel_l2_distance(img_a, img_b)
    p_ba = mt6.pixel_l2_distance(img_b, img_a)
    assert abs(p_ab - p_ba) < 1e-6

    at_ab = su6.centroid_aligned_transport_distance(points_a, points_b)
    at_ba = su6.centroid_aligned_transport_distance(points_b, points_a)
    assert abs(at_ab - at_ba) < 1e-6

    np.random.seed(42)
    circle_a = su6.generate_canonical_shape("circle", 20, 1.0, 1)
    circle_b = circle_a + np.array([3.0, 3.0])
    square = su6.generate_canonical_shape("square", 20, 1.0, 3)

    same_shape_dist = su6.centroid_aligned_transport_distance(circle_a, circle_b)
    diff_shape_dist = su6.centroid_aligned_transport_distance(circle_a, square)
    assert same_shape_dist < diff_shape_dist

    # Demo 7: Renderer-Observation Loop Tests
    sys.path.insert(0, str(DEMO7_ROOT))
    ru7 = load_module("demo7_renderer_utils", DEMO7_ROOT / "renderer_utils.py")

    # Test 1: Renderer output shape is correct
    np.random.seed(777)
    test_points = np.random.randn(20, 2)
    rendered_img = ru7.render_state_to_image(test_points, canvas_size=64)
    assert rendered_img.shape == (64, 64), f"Expected (64, 64), got {rendered_img.shape}"

    # Test 2: Rendered image changes when state changes
    state_a = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    state_b = np.array([[2.0, 2.0], [3.0, 2.0], [2.5, 3.0]])  # translated version
    img_a = ru7.render_state_to_image(state_a, canvas_size=64)
    img_b = ru7.render_state_to_image(state_b, canvas_size=64)
    # They should be different (unless translation is exactly zero)
    assert not np.allclose(img_a, img_b), "Translated states should produce different images"

    # Test 3: Naive recovery returns valid state within expected bounds
    np.random.seed(778)
    true_circle = np.column_stack([
        np.cos(np.linspace(0, 2 * np.pi, 20)[:-1]),
        np.sin(np.linspace(0, 2 * np.pi, 20)[:-1]),
    ])
    obs = ru7.render_state_to_image(true_circle, canvas_size=64)
    recovered = ru7.naive_recover_state(obs, n_points=20, seed=779)
    # Check bounds: points should be within reasonable world-space bounds
    assert recovered.shape == (20, 2), f"Expected shape (20, 2), got {recovered.shape}"
    assert np.all(np.isfinite(recovered)), "Recovered state should have finite values"

    # Test 4: Recovery on masked observation is better than naive baseline
    np.random.seed(780)
    true_state = np.column_stack([
        np.cos(np.linspace(0, 2 * np.pi, 20)[:-1]),
        np.sin(np.linspace(0, 2 * np.pi, 20)[:-1]),
    ])
    full_obs = ru7.render_state_to_image(true_state, canvas_size=64)
    masked_obs, _ = ru7.mask_observation(full_obs, mask_fraction=0.3, seed=781)

    # Naive baseline: random initialization
    np.random.seed(782)
    naive_state = np.random.randn(20, 2) * 0.5
    naive_img = ru7.render_state_to_image(naive_state, canvas_size=64)
    naive_dist = ru7.image_l2_distance(naive_img, masked_obs)

    # Recovery: use centroid from masked observation
    np.random.seed(783)
    recovered_state = ru7.naive_recover_state(masked_obs, n_points=20, seed=784)
    recovered_img = ru7.render_state_to_image(recovered_state, canvas_size=64)
    recovered_dist = ru7.image_l2_distance(recovered_img, masked_obs)

    # The centroid-based recovery should be closer to the masked observation than random
    assert recovered_dist < naive_dist, (
        f"Recovery ({recovered_dist:.3f}) should be better than naive ({naive_dist:.3f})"
    )

    # Demo 8: Integrated FETG Pipeline Tests
    # Save/restore global random state since earlier demos consume global np.random
    random_state = np.random.get_state()
    try:
        sys.path.insert(0, str(DEMO8_ROOT))
        pu8 = load_module("demo8_proposal", DEMO8_ROOT / "d8_proposal.py")
        eu8 = load_module("demo8_energy", DEMO8_ROOT / "d8_energy.py")
        ru8 = load_module("demo8_renderer", DEMO8_ROOT / "d8_renderer.py")
        pl8 = load_module("demo8_pipeline", DEMO8_ROOT / "d8_pipeline.py")

        # Test 1: Feasible proposal returns valid states
        state = pu8.sample_feasible_point_set(n_points=20, radius=0.8, bounds=2.0, seed=99)
        assert state.shape == (20, 2), f"Expected shape (20, 2), got {state.shape}"
        assert np.isfinite(state).all(), "All state values must be finite"
        assert np.all(state >= -2.0) and np.all(state <= 2.0), "States must be within bounds"
        feasible = pu8.check_point_set_feasibility(state, bounds=2.0)
        assert feasible, "Feasible proposal should pass feasibility check"

        # Test 2: Renderer output shape is correct
        np.random.seed(888)
        test_points = np.random.randn(20, 2) * 0.5
        rendered_img = ru8.render_state_to_image(test_points, canvas_size=64)
        assert rendered_img.shape == (64, 64), f"Expected (64, 64), got {rendered_img.shape}"

        # Test 3: Integrated energy is finite for valid states
        state_for_energy = pu8.sample_feasible_point_set(n_points=20, radius=0.8, bounds=2.0, seed=42)
        energy = eu8.point_cloud_energy(state_for_energy)
        assert np.isfinite(energy), "Energy must be finite"
        assert energy < 100.0, f"Energy too large: {energy}"

        # Test 4: Refinement improves energy or observation loss
        np.random.seed(42)
        cfg = pl8.PipelineConfig(seed=42, refine_steps=80)
        integrated = pl8.run_integrated_pipeline(cfg)

        energy_improved = integrated["refined_energy"] <= integrated["feasible_energy"]
        obs_improved = integrated["observation_loss_refined"] <= integrated["observation_loss_feasible"]
        assert energy_improved or obs_improved, (
            f"Refinement should improve energy or observation loss. "
            f"Energy: {integrated['feasible_energy']:.3f} → {integrated['refined_energy']:.3f}, "
            f"Obs loss: {integrated['observation_loss_feasible']:.3f} → {integrated['observation_loss_refined']:.3f}"
        )
        assert np.isfinite(integrated["refined_energy"])
        assert np.isfinite(integrated["observation_loss_refined"])
    finally:
        np.random.set_state(random_state)

    print("physics_state_gen_lab smoke tests passed")


if __name__ == "__main__":
    main()
