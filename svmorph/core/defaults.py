"""Shared spatial default constants for svMorph.

All values are expressed in **centimetres** and should be multiplied by
:func:`svmorph.core.units.L` at runtime to convert into the active unit
system.  Both the GUI (``main_window``) and the simulation interactor
import from here so that the defaults stay in sync.
"""

# Stent geometry
STENT_DIAMETER_DEFAULT_CM = 0.8
STENT_LENGTH_DEFAULT_CM = 1.7
MAX_STENT_SIZE_CM = 2.0
MIN_STENT_SIZE_CM = 0.1
MAX_STENT_LENGTH_CM = 8.0
MIN_STENT_LENGTH_CM = 1.0
UNDEPLOYED_STENT_DIAMETER_CM = 0.1
SMOOTHING_K_CM = 0.01
STENT_UNIT_SECTION_HALFLENGTH_CM = 0.2
STENT_SEGMENT_LENGTH_CM = 0.1
FORESHORTENING_PERCENTAGE = 0.1

# Stent deployment physics
INFLUENCE_RADIUS_CM = 0.65
CONTACT_DISTANCE_CM = 0.001

# Stenosis defaults
STENOSIS_RADIUS_DEFAULT_CM = 0.1
STENOSIS_LENGTH_DEFAULT_CM = 0.5

# Aneurysm defaults
ANEURYSM_MAX_RADIUS_DEFAULT_CM = 0.5
