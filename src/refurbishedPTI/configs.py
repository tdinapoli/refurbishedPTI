from dataclasses import dataclass
import yaml

path_test = ""#"/home/tomi/Documents/academicos/facultad/tesis/packages/refurbishedPTI/.rp_dirs_simulation"
path_emission = path_test+"/root/.local/refurbishedPTI/configs/emission_init.yaml"
path_excitation = path_test+"/root/.local/refurbishedPTI/configs/excitation_init.yaml"
with open(path_emission, 'r') as f:
    EMISSION_MONO_DRIVER = yaml.full_load(f)

with open(path_excitation, 'r') as f:
    EXCITATION_MONO_DRIVER = yaml.full_load(f)


PEAK_THRESHOLD = 0.5

## Decay configurations
# PULSE_WIDTH = 10e-9  # s
# RP_MAX_SR = 125e6  # hz
# RP_BUF_SIZE = 2**14
# LASER_FREQ = 80  # hz
# LASER_DC = 50  # %

# Gui
# Adjust values
@dataclass
class Gui:
    MAX_WL_STEP: float = 20
    MIN_WL_STEP: float = 0.5
    DEFAULT_INTEGRATION_TIME: float = 0.1
    MIN_INTEGRATION_TIME: float = 0.01
    MAX_INTEGRATION_TIME: float = 30
    MEASUREMENT_PATH: str = '/root/.local/refurbishedPTI/measurements'
    #MEASUREMENT_PATH: str = '/home/tomi/.local/refurbishedPTI/measurements'
    MIN_COUNTS: int = 0
    MAX_COUNTS: int = 1e6
    min_freq: float = 1
    max_freq: float = 1e3
    min_duty_cycle: float = 0
    max_duty_cycle: float = 100
    lifetime_default_counts: int = 1000
    lifetime_min_counts: int = 10
    lifetime_max_counts: int = 1e6
