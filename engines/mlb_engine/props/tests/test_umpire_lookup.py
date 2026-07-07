from baseball_props.analysis.edge_row_builder import (
    UmpireModifiers,
    lookup_umpire_modifiers,
    normalize_umpire_name,
)


def test_normalize_umpire_name() -> None:
    assert normalize_umpire_name("Marvin Hudson Jr.") == "marvin hudson"
    assert normalize_umpire_name("  CB Bucknor  ") == "cb bucknor"


def test_lookup_unknown_umpire_defaults_to_one() -> None:
    mods = lookup_umpire_modifiers("Unknown Umpire XYZ")
    assert mods == UmpireModifiers(umpire_name="Unknown Umpire XYZ", zone_size_modifier=1.0, run_environment_modifier=1.0)


def test_lookup_empty_string_defaults_to_one() -> None:
    mods = lookup_umpire_modifiers("")
    assert mods.umpire_name == ""
    assert mods.run_environment_modifier == 1.0
    assert mods.zone_size_modifier == 1.0


def test_lookup_tbd_defaults_to_one() -> None:
    mods = lookup_umpire_modifiers("TBD")
    assert mods.run_environment_modifier == 1.0


def test_lookup_known_umpire_multiplier() -> None:
    mods = lookup_umpire_modifiers("Marvin Hudson")
    assert mods.umpire_name == "Marvin Hudson"
    assert mods.run_environment_modifier == 0.97
    assert mods.zone_size_modifier == 1.03
