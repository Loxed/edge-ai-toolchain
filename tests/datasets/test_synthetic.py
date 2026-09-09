from edgekit.datasets.synthetic import sample_noisy_points


def test_sample_count_matches_n_per_class_times_corners():
    dataset = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]

    samples = sample_noisy_points(dataset, n_per_class=10, seed=0)

    assert len(samples) == 40


def test_reproducible_with_same_seed():
    dataset = [([0, 0], 0), ([1, 1], 1)]

    a = sample_noisy_points(dataset, n_per_class=5, seed=42)
    b = sample_noisy_points(dataset, n_per_class=5, seed=42)

    assert a == b


def test_label_matches_originating_corner():
    dataset = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
    n_per_class = 15

    samples = sample_noisy_points(dataset, n_per_class=n_per_class, seed=0)

    # points are generated corner by corner, in dataset order
    for i, (corner, expected_label) in enumerate(dataset):
        block = samples[i * n_per_class:(i + 1) * n_per_class]
        assert all(label == expected_label for _, label in block)


def test_points_stay_centered_near_expected_value_per_axis():
    # sanity check that low/high land on the correct axis and aren't
    # swapped, not a claim about the exact noise distribution
    dataset = [([0, 1], 1)]

    samples = sample_noisy_points(dataset, n_per_class=50, noise_std=0.02, seed=0)

    for point, _ in samples:
        assert 0.3 < point[0] < 0.5  # centered on low=0.4 (bit was 0)
        assert 0.5 < point[1] < 0.7  # centered on high=0.6 (bit was 1)