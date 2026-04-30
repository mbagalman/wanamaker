from wanamaker.benchmarks.loaders import load_synthetic_ground_truth

def test_load_synthetic_ground_truth():
    df, gt = load_synthetic_ground_truth()
    assert df is not None
    assert gt is not None
    assert 'channels' in gt
