import redpipy as rpp
import yaml

EMISSION_MONO_DRIVER = yaml.full_load(
    "/root/.local/refurbishedPTI/configs/emission_init.yaml"
)

EXCITATION_MONO_DRIVER = yaml.full_load(
    "/root/.local/refurbishedPTI/configs/excitation_init.yaml"
)

# PEAK_THRESHOLD = 0.5
# OSC_CHANNEL = "ch1"
#
## Decay configurations
# PULSE_WIDTH = 10e-9  # s
# RP_MAX_SR = 125e6  # hz
# RP_BUF_SIZE = 2**14
# LASER_FREQ = 80  # hz
# LASER_DC = 50  # %
