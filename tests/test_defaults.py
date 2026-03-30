"""Tests for svmorph.core.defaults – spatial constant sanity checks."""

from svmorph.core import defaults


def test_stent_constants_are_positive():
    assert defaults.STENT_DIAMETER_DEFAULT_CM > 0
    assert defaults.STENT_LENGTH_DEFAULT_CM > 0
    assert defaults.MAX_STENT_SIZE_CM > 0
    assert defaults.MIN_STENT_SIZE_CM > 0
    assert defaults.MAX_STENT_LENGTH_CM > 0
    assert defaults.MIN_STENT_LENGTH_CM > 0
    assert defaults.UNDEPLOYED_STENT_DIAMETER_CM > 0
    assert defaults.SMOOTHING_K_CM > 0
    assert defaults.STENT_UNIT_SECTION_HALFLENGTH_CM > 0
    assert defaults.STENT_SEGMENT_LENGTH_CM > 0


def test_stent_size_bounds():
    assert defaults.MIN_STENT_SIZE_CM < defaults.MAX_STENT_SIZE_CM
    assert defaults.MIN_STENT_LENGTH_CM < defaults.MAX_STENT_LENGTH_CM
    assert defaults.UNDEPLOYED_STENT_DIAMETER_CM < defaults.STENT_DIAMETER_DEFAULT_CM


def test_influence_and_contact_distances():
    assert defaults.INFLUENCE_RADIUS_CM > 0
    assert defaults.CONTACT_DISTANCE_CM > 0
    assert defaults.CONTACT_DISTANCE_CM < defaults.INFLUENCE_RADIUS_CM


def test_stenosis_defaults():
    assert defaults.STENOSIS_RADIUS_DEFAULT_CM > 0
    assert defaults.STENOSIS_LENGTH_DEFAULT_CM > 0


def test_aneurysm_defaults():
    assert defaults.ANEURYSM_MAX_RADIUS_DEFAULT_CM > 0


def test_foreshortening_percentage():
    assert 0 < defaults.FORESHORTENING_PERCENTAGE < 1
