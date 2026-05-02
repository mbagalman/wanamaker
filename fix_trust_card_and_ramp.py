import re
with open('tests/benchmarks/test_candidate_generation_safety.py', 'r') as f:
    content = f.read()

# Fix ramp_rec UnboundLocalError
content = content.replace(
    '''    # We can also attempt a ramp on one of the candidates vs the baseline
    if len(result.candidates) > 0:
        ramp_rec = recommend_ramp(
            posterior_summary,
            baseline_data,
            result.candidates[0].plan,
            seed=42,
            engine=engine,
            trust_card=None,
        )

    # Make sure ramp generation works on the same config
    assert ramp_rec is not None''',
    '''    # We can also attempt a ramp on one of the candidates vs the baseline
    assert len(result.candidates) > 0
    ramp_rec = recommend_ramp(
        posterior_summary,
        baseline_data,
        result.candidates[0].plan,
        seed=42,
        engine=engine,
        trust_card=None,
    )

    # Make sure ramp generation works on the same config
    assert ramp_rec is not None'''
)

# Fix trust card assertions
content = content.replace(
    '''        # Depending on ramp logic, weak dimensions might block moves or reduce fractional_kelly.
        # Check that weak trust card acts as a constraint in ramp.
        # This is enough to show integration.
        pass''',
    '''        # Depending on ramp logic, weak dimensions block moves or reduce fractional_kelly.
        # Ensure that trust card has an effect
        assert ramp_rec_weak.candidates[0].fractional_kelly < ramp_rec_strong.candidates[0].fractional_kelly or \
               not ramp_rec_weak.candidates[0].passes or \
               len(ramp_rec_weak.candidates[0].failed_gates) > len(ramp_rec_strong.candidates[0].failed_gates)'''
)

with open('tests/benchmarks/test_candidate_generation_safety.py', 'w') as f:
    f.write(content)
