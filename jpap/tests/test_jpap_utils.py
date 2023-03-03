import pytest
import random

import jpap

def simulate_string():
    string_length = random.randint(1, 10)
    random_string = "".join([chr(c) for c in random.choices(population=range(97,122), k = string_length)])
    return random_string

# tests: -----
def test_spf_input():
    wrong_inputs = [True, 1, 1.2]
    for i in wrong_inputs:
        with pytest.raises(TypeError):
            jpap.sql_pattern_formatting(patterns = i)

def test_spf_output():
    random.seed(1)
    input_patterns = [simulate_string() for x in range(10)]
    formatted_strings = jpap.sql_pattern_formatting(input_patterns)
    for i, fs in enumerate(formatted_strings):
        assert fs[:3] == "'% " and fs[-2:] == "%'" # test encoding
        assert fs[3:-2] == input_patterns[i] # test that nothing else changes