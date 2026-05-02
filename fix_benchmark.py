import re
with open('tests/benchmarks/test_candidate_generation_safety.py', 'r') as f:
    content = f.read()

content = content.replace(
    'def test_benchmark_naive_optimizer_unsafe_move_rejected(base_config, baseline_data, posterior_summary):',
    'def test_benchmark_naive_optimizer_unsafe_move_rejected(benchmark, base_config, baseline_data, posterior_summary):'
).replace(
    '    result = suggest_scenarios(',
    '    result = benchmark(suggest_scenarios,'
).replace(
    '''        engine=engine,
    )''',
    '''        engine=engine,
    )'''
).replace(
    'def test_benchmark_constrained_to_locked_channels(base_config, baseline_data, posterior_summary):',
    'def test_benchmark_constrained_to_locked_channels(benchmark, base_config, baseline_data, posterior_summary):'
).replace(
    'def test_benchmark_no_plans_if_all_unsafe(base_config, baseline_data, posterior_summary):',
    'def test_benchmark_no_plans_if_all_unsafe(benchmark, base_config, baseline_data, posterior_summary):'
).replace(
    'def test_benchmark_spend_invariant_channels(base_config, baseline_data, posterior_summary):',
    'def test_benchmark_spend_invariant_channels(benchmark, base_config, baseline_data, posterior_summary):'
).replace(
    'def test_benchmark_compatibility_with_ramp_recommendations(base_config, baseline_data, posterior_summary):',
    'def test_benchmark_compatibility_with_ramp_recommendations(benchmark, base_config, baseline_data, posterior_summary):'
).replace(
    'def test_benchmark_weak_trust_card_dimensions_reduce_candidate_moves(base_config, baseline_data, posterior_summary):',
    'def test_benchmark_weak_trust_card_dimensions_reduce_candidate_moves(benchmark, base_config, baseline_data, posterior_summary):'
)

with open('tests/benchmarks/test_candidate_generation_safety.py', 'w') as f:
    f.write(content)
