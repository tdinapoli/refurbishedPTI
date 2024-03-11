
import redpipy as rpp

EMISSION_MONO_DRIVER = {
'pin_step'  :  rpp.digital.RPDO((True, 'p', 4)),
'direction'  :  rpp.digital.RPDO((True, 'p', 5)),
'limit_switch'  :  rpp.digital.RPDO((True, 'p', 3)),
        }

EXCITATION_MONO_DRIVER = {
'pin_step'  :  rpp.digital.RPDO((True, 'p', 6)),
'direction'  :  rpp.digital.RPDO((True, 'p', 7)),
'limit_switch'  :  rpp.digital.RPDO((True, 'p', 2)),
        }

PEAK_THRESHOLD = -3.5
OSC_CHANNEL = "ch1"