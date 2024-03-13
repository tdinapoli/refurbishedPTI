
import redpipy as rpp

EMISSION_MONO_DRIVER = {
'pin_step'  :  ('p', 4),
'pin_direction'  :  ('p', 5),
'limit_switch'  : ('p', 3),
'calibration_path' : '/root/refurbishedPTI-files/emission_calibration.yaml'
        }

EXCITATION_MONO_DRIVER = {
'pin_step'  :  ('p', 6),
'pin_direction'  :  ('p', 7),
'limit_switch'  :  ('p', 2),
'calibration_path' : '/root/refurbishedPTI-files/excitation_calibration.yaml'
        }

PEAK_THRESHOLD = -3.5
OSC_CHANNEL = "ch1"

# Decay configurations
PULSE_WIDTH = 10e-9 # s
RP_MAX_SR = 125e6 # hz
RP_BUF_SIZE = 2**14
LASER_FREQ = 80 # hz
LASER_DC = 50 # %