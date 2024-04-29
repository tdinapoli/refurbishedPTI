"""
    refurbishedPTI.instruments
    ~~~~~~~~~~~~~~~

    Refurbished HoribaPTI instruments.


    :copyright: 2024 by redpipy Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from . import abstract, configs
from . import user_interface as ui
from typing import Literal

import redpipy as rpp
import yaml
import pandas as pd
import numpy as np

class DRV8825(abstract.MotorDriver):
    ttls: dict
    _MODES = (
                (False, False, False), #Full step
                (True, False, False) , #Half 
                (False, True, False) , #Quarter
                (True, True, False)  , #Eighth
                (True, True, True)   , #Sixteenth
               )

    def __init__(self, ttls: dict, stepping: int = 0):
        for ttl in ttls:
            setattr(self, ttl, ttls[ttl])
        self._stepping = stepping
        self.set_stepping(self._stepping)

    def set_stepping(self, stepping: int):
        m1, m2, m3 = self._MODES[stepping]
        try:
            self.m1.set_state(m1)
            self.m2.set_state(m2)
            self.m3.set_state(m3)
        except AttributeError:
            # TODO: add some kind of warning that this is the default.
            self._stepping = 0

    def get_stepping(self):
        try:
            self._stepping = self._MODES.index(
                (self.m1.state, self.m2.state, self.m3.state)
            )
            return self._stepping
        except AttributeError:
            return self._stepping

    # Obs: Minimum pulse duration is 1 micro second.
    # Obs2: slope has to be rising for step pin to work.
    def step(self, ontime=10e-3, offtime=10e-3, amount=1):
        self.pin_step.pulse(ontime, offtime, amount)


class M061CS02(abstract.Motor):
    _STEPS_MODE = (200, 400, 800, 1600, 3200)

    def __init__(self, driver, steps: int = 400, angle: float = 0.0):
        self._driver = driver
        self._angle = angle
        self._angle_relative = angle % 360
        self.steps = steps
        self._min_angle = 360.0 / self.steps
        self._min_offtime = 10e-3
        self._min_ontime = 10e-3

    def rotate(self, angle: float):
        relative_angle = angle - self._angle
        return self.rotate_relative(relative_angle)

    def rotate_relative(self, angle: float, change_angle: bool = True):
        cw, angle = self._cw_from_angle(angle), abs(angle)
        self._driver.direction.set_state(cw)
        steps = self._steps_from_angle(angle)
        self.rotate_step(steps, cw, change_angle=change_angle)
        angle_rotated = self._angle_sign(cw) * self._angle_from_steps(steps)
        return angle_rotated

    def rotate_step(self, steps: int, cw: bool, change_angle: bool = True):
        self._driver.direction.set_state(cw)
        self._driver.step(
            ontime=self._min_ontime, offtime=self._min_offtime, amount=steps
        )
        if change_angle:
            angle_change_sign = int(cw) * 2 - 1
            self._angle = round(
                self._angle + angle_change_sign * self.min_angle * steps, 5
            )

    def _angle_sign(self, cw: bool):
        return 2 * cw - 1

    def _angle_from_steps(self, steps: int):
        if steps >= 0:
            return self.min_angle * steps
        else:
            raise ValueError(f"steps should be greater than 0, not {steps}.")

    def _steps_from_angle(self, angle: float):
        if angle >= 0:
            return int(angle / self.min_angle)
        else:
            raise ValueError(f"angle should be greater than 0, not {angle}.")

    def _cw_from_angle(self, angle: float):
        return angle > 0

    @property
    def angle(self):
        return self._angle

    @property
    def angle_relative(self):
        return self._angle % 360

    @property
    def min_angle(self):
        return 360 / self.steps

    @property
    def steps(self):
        return self._STEPS_MODE[self._driver.get_stepping()]

    @steps.setter
    def steps(self, steps: int = 200):
        self._driver.set_stepping(self._STEPS_MODE.index(steps))

    def set_origin(self, angle: float = 0):
        self._angle = angle


class Monochromator:
    def __init__(
        self,
        motor: abstract.Motor,
        limit_switch: rpp.digital.RPDI,
        # TODO: improve path handling
        calibration_path: str = None,
    ):
        # TODO: move this to calibration.
        self.CALIB_ATTRS = [
            "_wl_step_ratio",
            "_greater_wl_cw",
            "_max_wl",
            "_min_wl",
            "_home_wavelength",
        ]
        self._motor = motor
        self._limit_switch = limit_switch
        if calibration_path is not None:
            self.load_calibration(calibration_path)

    @classmethod
    def constructor_default(
        cls,
        pin_step: tuple[Literal["n", "p"], int],
        pin_direction: tuple[Literal["n", "p"], int],
        limit_switch: tuple[Literal["n", "p"], int],
        MOTOR_DRIVER: abstract.MotorDriver = DRV8825,
        MOTOR: abstract.Motor = M061CS02,
        calibration_path: str = None,
    ):
        ttls = {
            "pin_step": rpp.digital.RPDO(pin_step, state=False),
            "direction": rpp.digital.RPDO(pin_direction, state=True),
        }

        driver = MOTOR_DRIVER(ttls)
        motor = MOTOR(driver)
        limit_switch = rpp.digital.RPDI(pin=limit_switch)
        return cls(motor, limit_switch=limit_switch, calibration_path=calibration_path)

    @property
    def wavelength(self):
        try:
            return self._wavelength
        except:
            return None

    @property
    def min_wl(self):
        return self._min_wl

    @property
    def max_wl(self):
        return self._max_wl

    @property
    def greater_wl_cw(self):
        return self._greater_wl_cw

    @property
    def wl_step_ratio(self):
        return self._wl_step_ratio

    @property
    def home_wavelength(self):
        return self._home_wavelength

    def set_wavelength(self, wavelength: float):
        if self.check_safety(wavelength):
            self._wavelength = wavelength
        else:
            print(f"Wavelength must be between {self._min_wl} and {self._max_wl}")

    def check_safety(self, wavelength):
        return self._min_wl <= wavelength <= self._max_wl

    def goto_wavelength(self, wavelength: float):
        if self.check_safety(wavelength):
            steps = self._steps_from_wl(wavelength)
            cw = self._cw_from_wl(wavelength)
            self._motor.rotate_step(steps, cw)
            self._wavelength = wavelength
        else:
            print(f"Wavelength must be between {self._min_wl} and {self._max_wl}")
        return self._wavelength

    def _steps_from_wl(self, wavelength: float):
        return abs(int((wavelength - self.wavelength) / self._wl_step_ratio))

    def _cw_from_wl(self, wavelength: float):
        cw = (wavelength - self.wavelength) > 0
        cw = not (cw ^ self.greater_wl_cw)
        return cw

    @property
    def limit_switch(self):
        return self._limit_switch

    def load_calibration(self, path: str):  # wavelength
        # TODO: make a better implementation of calibration loading.
        with open(path, "r") as f:
            self._calibration = yaml.full_load(f)
        for param in self._calibration:
            setattr(self, f"_{param}", self._calibration[param])
        calibration_complete = True
        for param in self.CALIB_ATTRS:
            if not hasattr(self, param):
                calibration_complete = False
                print(f"Calibration parameter {param[1:]} is missing.")
        if not calibration_complete:
            print("Calibration is incomplete.")

    def calibrate(self):
        ui.SpectrometerCalibrationInterface(self).cmdloop()
        # TODO: make a better implementation of calibration loading.
        self.load_calibration(self.calibration_path)

    def swipe_wavelengths(
        self,
        starting_wavelength: float = None,
        ending_wavelength: float = None,
        wavelength_step: float = None,
    ):
        if starting_wavelength is None:
            starting_wavelength = self.min_wl
        if ending_wavelength is None:
            ending_wavelength = self.max_wl
        if wavelength_step is None:
            wavelength_step = abs(self.wl_step_ratio)

        n_measurements = int(
            (ending_wavelength - starting_wavelength) / wavelength_step
        )
        for i in range(n_measurements):
            yield self.goto_wavelength(starting_wavelength + i * wavelength_step)

    def home(self, set_wavelength=True):
        steps_done = 0
        # This ensures ending wavelength is above 0.
        steps_limit = self.home_wavelength / self.wl_step_ratio
        while self.limit_switch.state and steps_done < abs(steps_limit):
            self._motor.rotate_step(1, not self._greater_wl_cw)
            steps_done += 1
        if set_wavelength and steps_done < steps_limit:
            self.set_wavelength(self.home_wavelength)
        elif steps_done >= abs(steps_limit):
            print("Danger warning:")
            print(
                f"Wavelength could not be set. Call home method again if and only if wavelength is greater than {self.home_wavelength}"
            )


class Spectrometer(abstract.Spectrometer):
    def __init__(
        self,
        emission_mono: Monochromator,
        osc: rpp.osci.Oscilloscope,
        excitation_mono: Monochromator,
    ):
        self.emission_mono = emission_mono

        self._osc = osc
        self._osc.channel1.enabled = True
        self._osc.channel1.set_gain(5)
        self._osc.configure_trigger()

        self.excitation_mono = excitation_mono

    @classmethod
    def constructor_default(
        cls,
        # TODO: pass monochromator instances.
        MONOCHROMATOR=Monochromator,
        OSCILLOSCOPE=rpp.Oscilloscope,
    ):
        emission_mono = MONOCHROMATOR.constructor_default(
            **configs.EMISSION_MONO_DRIVER
        )

        osc = OSCILLOSCOPE()
        excitation_mono = MONOCHROMATOR.constructor_default(
            **configs.EXCITATION_MONO_DRIVER
        )
        return cls(emission_mono, osc, excitation_mono)

    # TODO: leave this method here or directly call self.emission_mono.goto_wavelength
    def goto_wavelength(self, wavelength):
        return self.emission_mono.goto_wavelength(wavelength)

    def goto_excitation_wavelength(self, wavelength):
        return self.excitation_mono.goto_wavelength(wavelength)

    def get_emission(
        self,
        integration_time: float,
        excitation_wavelength: float = None,
        feed: callable = None,
        **kwargs,
    ):
        df = self.get_spectrum(
            integration_time=integration_time,
            static_wavelength=excitation_wavelength,
            emission=True,
            feed=feed,
            **kwargs,
        )
        df.attrs["type"] = "emission_spectrum"
        df.attrs["excitation_wavelength"] = excitation_wavelength
        return df

    def get_excitation(
        self,
        integration_time: float,
        emission_wavelength: float = None,
        feed: callable = None,
        **kwargs,
    ):
        df = self.get_spectrum(
            integration_time=integration_time,
            static_wavelength=emission_wavelength,
            emission=False,
            feed=feed,
            **kwargs,
        )
        df.attrs["type"] = "excitation_spectrum"
        df.attrs["emission_wavelength"] = emission_wavelength
        return df

    def get_spectrum(
        self,
        integration_time: float,
        static_wavelength: float = None,
        emission: bool = True,
        feed=None,
        **kwargs,
    ):
        if emission:
            static_mono = self.excitation_mono
        else:
            static_mono = self.emission_mono

        if static_wavelength is not None:
            static_mono.goto_wavelength(static_wavelength)
        spectrum_iterator = self._yield_spectrum(
            emission=emission, integration_time=integration_time, **kwargs
        )
        data = []
        for el in spectrum_iterator:
            data.append(el)
            if feed is not None:
                feed(el)
        return pd.DataFrame(data)

    def _yield_spectrum(
        self,
        integration_time: float,
        emission: bool = True,
        starting_wavelength: float = None,
        ending_wavelength: float = None,
        wavelength_step: float = None,
        rounds: int = 1,
        path=None,
    ):
        if emission:
            monochromator = self.emission_mono
        else:
            monochromator = self.excitation_mono
        for i, wl in enumerate(
            monochromator.swipe_wavelengths(
                starting_wavelength=starting_wavelength,
                ending_wavelength=ending_wavelength,
                wavelength_step=wavelength_step,
            )
        ):
            photons, time_measured = self.get_intensity(
                integration_time, rounds, path=path
            )
            yield dict(wavelength=wl, counts=photons, integration_time=time_measured)

    def get_intensity(self, seconds, rounds: int = 1, path=None):
        photons = 0
        # TODO: see if this loop can be moved to a lower level stage
        # so that it takes less time to complete.
        for _ in range(rounds):
            photons += self.integrate(seconds)
        # TODO: check if amount_datapoints works with new API. Res: works with _ at beginning
        # TODO: change osci API or find another solution to amount_datapoints
        time_measured = (
            rounds
            * self._osc._amount_datapoints
            / self._osc.get_timebase_settings()["sampling_rate"]
        )
        return photons, time_measured

    def integrate(self, seconds):
        # TODO: timebase should always be at maximum sampling rate.
        # change this function to integrate for any amount of seconds
        # but keep msr.
        self._osc.set_timebase(seconds)
        self._osc.trigger_now()
        osc_screen = self._osc.channel1.get_trace()
        # TODO: decide if i keep get_data (full dataframe) or just
        # get_trace.
        # osc_screen = self._osc.get_data()
        # photons = self._count_photons(osc_screen[configs.OSC_CHANNEL])
        return self._count_photons(osc_screen)

    def _count_photons(self, osc_screen):
        # TODO: save threshold in configuration.
        times = self._find_signal_peaks(-osc_screen, configs.PEAK_THRESHOLD)
        return len(times)

    def _find_signal_peaks(self, osc_screen, threshold, offset=0):
        # TODO: calibrate this
        peaks = np.where(np.diff(osc_screen) > threshold)[0]
        sampling_rate = self._osc.get_timebase_settings()["sampling_rate"]
        return peaks / sampling_rate + offset

    def set_wavelength(self, wavelength: float):
        return self.emission_mono.set_wavelength(wavelength)

    def set_excitation_wavelength(self, wavelength: float):
        return self.excitation_mono.set_wavelength(wavelength)

    def home(self):
        self.excitation_mono.home()
        self.emission_mono.home()
        print(f"Lamp wavelength should be {self.excitation_mono.home_wavelength}")
        print(
            f"Monochromator wavelength should be {self.emission_mono.home_wavelength}"
        )
        print(
            f"If they are wrong, set them with spec.lamp.set_wavelength() and spec.monochromator.set_wavelength()"
        )

    # TODO: change this for new API.
    @property
    def decay_configuration(self):
        if self._osc.channel == 0 and self._osc.trig_src == 8:
            state = True
        elif self._osc.channel == 1 and self._osc.trig_src == 4:
            state = True
        else:
            state = False
        return state

    # TODO: change this for new API.
    @decay_configuration.setter
    def decay_configuration(self, val):
        self._osc.decimation = 0
        self._osc.trigger_pre = 0
        self._osc.trigger_post = self._osc.buffer_size
        if val:
            self._osc.set_trigger(channel=1, edge="pos", level=1.0)
        else:
            self._osc.set_trigger(channel=None)

    # TODO: Change this for new API.
    def acquire_decay(self, amount_windows=1, amount_buffers=1):
        self.decay_configuration = True
        counts = np.array([])
        for window in range(amount_windows):
            buff_offset = window * self._osc.amount_datapoints
            for _ in range(amount_buffers):
                screen = np.array(self._osc.get_triggered())
                # Aca también, cambiar los números por calibracion/configuracion
                times = self._find_signal_peaks(screen, 0.16, 0.18)
                times = (
                    peak_positions + buff_offset
                ) / self._osc.get_timebase_settings()["sampling_rate"]
                counts = np.hstack((counts, times))
        return counts
