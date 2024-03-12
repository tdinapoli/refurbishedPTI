
import redpipy as rpp

EMISSION_MONO_DRIVER = {
'pin_step'  :  ('p', 4),
'pin_direction'  :  ('p', 5),
'limit_switch'  : ('p', 3),
        }

EXCITATION_MONO_DRIVER = {
'pin_step'  :  ('p', 6),
'pin_direction'  :  ('p', 7),
'limit_switch'  :  ('p', 2),
        }

PEAK_THRESHOLD = -3.5
OSC_CHANNEL = "ch1"