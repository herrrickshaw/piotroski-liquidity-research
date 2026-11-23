"""combo_by_tier_us, size_vs_liquidity_us and verify_f_band_us hold their findings as a module
docstring and no executable code. Importing them must stay side-effect free and keep the
documented headline numbers readable; there is nothing else to execute."""
import types

import pytest


@pytest.mark.parametrize("name,needles", [
    ("combo_by_tier_us", ["UNDERPOWERED", "darvas x piotroski7", "SMALL n=1"]),
    ("size_vs_liquidity_us", ["ILLIQUIDITY, NOT SIZE", "LARGE_CAP  ILLIQ", "+33.7%"]),
    ("verify_f_band_us", ["INVERTED-U", "WINSORIZATION", "F 7-9 strong"]),
])
def test_module_imports_cleanly_and_documents_its_finding(load, name, needles):
    m = load(name)
    public = [k for k, v in vars(m).items() if not k.startswith("_") and not isinstance(v, types.ModuleType)]
    assert public == []                                  # no functions, no constants, no work on import
    for n in needles:
        assert n in m.__doc__
