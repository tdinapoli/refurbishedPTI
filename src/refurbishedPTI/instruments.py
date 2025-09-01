"""
refurbishedPTI.instruments
~~~~~~~~~~~~~~~

Refurbished HoribaPTI instruments.


:copyright: 2024 by redpipy Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

import time
from typing import Literal

import numpy as np
import pandas as pd
import pyvisa
import redpipy as rpp
import yaml

from . import abstract, configs
from . import user_interface as ui


class ITC4020:
    def __init__(self, config_path, verbose=False):
        self.RESOURCE_STRING = "USB0::4883::32842::M00404162::0::INSTR"
        self.TIMEOUT = 100000
        self.ID = "Thorlabs,ITC4001,M00404162,1.8.0/1.5.0/2.3.0\n"
        rm = pyvisa.ResourceManager("@py")
        self.itc = rm.open_resource(self.RESOURCE_STRING)
        self._establish_connection()
        self.load_configuration(config_path, verbose=verbose)

    def query(self, *args, **kwargs):
        return self.itc.query(*args, **kwargs)

    def _establish_connection(self):
        start = time.time()
        while time.time() - start < self.TIMEOUT:
            try:
                msg = self.itc.query("*IDN?")
                if msg == self.ID:
                    print(f"Connection established as expected\n{msg}")
                    return True
                else:
                    print(f"Connection established to unknown instrument\n{msg}")
            except Exception as e:
                print(e)
        raise TimeoutError

    def load_configuration(self, path, verbose=False):
        try:
            with open(path, "r") as f:
                self._configuration = yaml.full_load(f)
        except FileNotFoundError:
            print(f"Error: configuration file '{path}' not found.")
            return
        for menu_name, menu_vals in self._configuration.items():
            if verbose:
                print(f"Setting {menu_name} vals:")
            for parameter, value in menu_vals.items():
                print(f"{parameter=} {value=}")
                if hasattr(self, parameter):
                    setattr(self, parameter, value)
                    if verbose:
                        print(f"{parameter} = {value}")
                else:
                    print(f"ITC4020 has no attribute {parameter}")

    def turn_on_laser(self):
        if self.keylock_tripped:
            print("Please unlock safety key")
            return
        if self.modulation and self.qcw_mode == "PULS":
            print("Both modulation and QCW mode are on.")
            print("Turn one off before turning on the LASER.")
            return
        self.tec_output = True
        while self.temp_tripped:
            print("Waiting for laser to cool")
            time.sleep(1)
        self.ld_output = True
        time.sleep(2)

    # esto hay que cambiarlo para que no dependa de una config file
    def get_configuration(self, path):
        try:
            with open(path, "r") as f:
                config = yaml.full_load(f)
        except FileNotFoundError:
            print(f"Error: configuration file '{path}' not found.")
            return
        for menu_name, menu_params in config.items():
            print(f"{menu_name} values:")
            for param in menu_params:
                print(f"{param} = {getattr(self, param)}")

    @property
    def ld_output(self):
        return bool(int(self.itc.query("output:state?")))

    @ld_output.setter
    def ld_output(self, state):
        try:
            state = int(state)
            if state not in [0, 1]:
                print("state should be either 0 or 1, not {state}")
                return
            self.itc.write(f"output:state {int(state)}")
        except ValueError as e:
            print(e)

    @property
    def tec_output(self):
        return bool(int(self.itc.query("output2:state?")))

    @tec_output.setter
    def tec_output(self, state):
        try:
            state = int(state)
            if state not in [0, 1]:
                print("state should be either 0 or 1, not {state}")
                return
            self.itc.write(f"output2:state {int(state)}")
        except ValueError as e:
            print(e)

    @property
    def temp_tripped(self):
        return bool(int(self.itc.query("sense3:temperature:protection:tripped?")))

    @property
    def polarity(self):
        return self.itc.query("output:polarity?")

    @polarity.setter
    def polarity(self, polarity_value):
        print(f"setting polarity to {polarity_value}?")
        if polarity_value in ["CG", "AG"]:
            print("writing polarity")
            self.itc.write(f"output:polarity {polarity_value}")
            print("written")
        else:
            print("Wrong polarity value")

    @property
    def voltage_protection(self):
        return self.itc.query("output:protection:voltage?")

    @voltage_protection.setter
    def voltage_protection(self, voltage_limit):
        print("setting voltage_protection")
        self.itc.write(f"output:protection:voltage {voltage_limit}")
        print("setted")

    @property
    def operating_mode(self):
        return self.itc.query("source:function:mode?")

    @operating_mode.setter
    def operating_mode(self, mode):
        if mode in ["current", "power"]:
            self.itc.write(f"source:function:mode {mode}")
        else:
            print("Wrong operating mode")

    @property
    def laser_current_limit(self):
        return self.itc.query("source:current:limit?")

    @laser_current_limit.setter
    def laser_current_limit(self, limit):
        self.itc.write(f"source:current:limit {limit}")

    @property
    def optical_power_limit(self):
        return self.itc.query("sense:power:protection?")

    @optical_power_limit.setter
    def optical_power_limit(self, limit):
        self.itc.write(f"sense:power:protection {limit}")

    @property
    def optical_power(self):
        return float(self.itc.query("measure:power2?")[:-1])

    @property
    def temperature(self):
        return float(self.itc.query("measure:temperature?")[:-1])

    @property
    def laser_current(self):
        return float(self.itc.query("source:current?")[:-1])

    @laser_current.setter
    def laser_current(self, current):
        self.itc.write(f"source:current {current}")

    @property
    def modulation(self):
        return bool(int(self.itc.query("source:am:state?")[:-1]))

    @modulation.setter
    def modulation(self, state):
        # revisar: agregar para que apague qcw
        state = bool(state) * 1
        self.itc.write(f"source:am:state {state}")

    @property
    def qcw_mode(self):
        resp = self.itc.query("source:function:shape?")[:-1]
        if resp == "PULS":
            return True
        else:
            return False

    @qcw_mode.setter
    def qcw_mode(self, mode):
        # mode = False -> dc
        # mode = True -> pulse
        print(f"{self.qcw_mode=}, {mode=}")
        if mode == self.qcw_mode:
            print("Already in that mode")
            return
        # revisar: agregar para que apague modulation
        if mode:
            print(f"{self.qcw_mode=}, {mode=}")
            self.itc.write("source:function:shape pulse")
        else:
            print(f"{self.qcw_mode=}, {mode=}")
            self.itc.write("source:function:shape dc")

    @property
    def trigger_source(self):
        return self.itc.query("trigger:source?")

    @trigger_source.setter
    def trigger_source(self, source):
        if source in ["internal", "external"]:
            self.itc.write(f"trigger:source {source}")
        else:
            print(f"Source {source} not supported")

    @property
    def frequency(self):
        val = float(self.itc.query("source:pulse:period?"))
        return 1 / val

    @frequency.setter
    def frequency(self, value):
        self.itc.write(f"source:pulse:period {1/value}")

    @property
    def duty_cycle(self):
        return float(self.itc.query("source:pulse:dcycle?")[:-1])

    @duty_cycle.setter
    def duty_cycle(self, dc):
        self.itc.write(f"source:pulse:dcycle {dc}")

    @property
    def hold(self):
        return self.itc.query("source:pulse:hold?")

    @hold.setter
    def hold(self, value):
        if value in ["width", "dcycle"]:
            self.itc.write(f"source:pulse:hold {value}")
        else:
            print(f"Hold parameter {value} is not supported")

    @property
    def keylock_tripped(self):
        return bool(int(self.itc.query("output:protection:keylock:tripped?")))


class DRV8825(abstract.MotorDriver):
    ttls: dict
    _MODES = (
        (False, False, False),  # Full step
        (True, False, False),  # Half
        (False, True, False),  # Quarter
        (True, True, False),  # Eighth
        (True, True, True),  # Sixteenth
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
        limit_switch,#: rpp.digital.RPDI,
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

    def home(self, set_wavelength=True, ignore_limit=False):
        steps_done = 0
        # This ensures ending wavelength is above 0.
        if ignore_limit:
            print("WARNING: IGNORING LIMIT SWITCH SAFETY LIMIT")
            steps_limit = 2e3
        else:
            steps_limit = abs(self.home_wavelength / self.wl_step_ratio)
        # TODO: if you start at the home_wavelength, this doesn't work
        while self.limit_switch.state and steps_done < steps_limit:
            self._motor.rotate_step(1, not self._greater_wl_cw)
            steps_done += 1
        if set_wavelength and steps_done < steps_limit:
            self.set_wavelength(self.home_wavelength)
        elif steps_done >= steps_limit:
            print("Danger warning:")
            print(
                f"Wavelength could not be set. Call home method again if and only if wavelength is greater than {self.home_wavelength}"
            )


class Spectrometer(abstract.Spectrometer):
    def __init__(
        self,
        excitation_mono: Monochromator,
        emission_mono: Monochromator,
        osc,#: rpp.osci.Oscilloscope,
        home: bool=False
    ):
        self.excitation_mono = excitation_mono
        self.emission_mono = emission_mono

        self._osc = osc
        self._osc.channel1.enabled = True
        self._osc.channel1.set_gain(5)
        self._osc.configure_trigger()

        if home:
            self.emission_mono.home()
            self.excitation_mono.home()

    @classmethod
    def constructor_default(
        cls,
        excitation_mono: Monochromator = None,
        emission_mono: Monochromator = None,
        osc = None,#: rpp.osci.Oscilloscope = None,
        home: bool = False,
    ):
        if excitation_mono is None:
            excitation_mono = Monochromator.constructor_default(
                **configs.EXCITATION_MONO_DRIVER
            )
        if emission_mono is None:
            emission_mono = Monochromator.constructor_default(
                **configs.EMISSION_MONO_DRIVER
            )
        if osc is None:
            osc = rpp.Oscilloscope()
        return cls(excitation_mono, emission_mono, osc, home=home)

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
        feed_wl=None,
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
            if feed_wl is not None:
                feed_data = feed_wl(wl)
            else:
                feed_data = None
            photons, time_measured = self.get_intensity(
                integration_time, rounds, feed_data=feed_data
            )
            yield dict(wavelength=wl, counts=photons, integration_time=time_measured)

    def get_intensity(self, seconds, rounds: int = 1, feed_data=None):
        photons = 0
        # TODO: see if this loop can be moved to a lower level stage
        # so that it takes less time to complete.
        for _ in range(rounds):
            photons += self.integrate(seconds, feed_data=feed_data)
        # TODO: check if amount_datapoints works with new API. Res: works with _ at beginning
        # TODO: change osci API or find another solution to amount_datapoints
        time_measured = (
            rounds
            * self._osc._amount_datapoints
            / self._osc.get_timebase_settings()["sampling_rate"]
        )
        return photons, time_measured

    def integrate(self, seconds, feed_data: callable = None):
        # TODO: timebase should always be at maximum sampling rate.
        # change this function to integrate for any amount of seconds
        # but keep msr.
        t_2nd_dec = 0.00026 * 2
        reps = int(seconds/t_2nd_dec)
        photons = 0
        self._osc.set_timebase(t_2nd_dec)
        buffer = np.empty(self._osc._amount_datapoints, dtype=np.float32)
        for rep in range(reps):
            self._osc.trigger_now()
            data = self._osc.get_voltage_numpy("ch1", out=buffer)
            if feed_data is not None:
                feed_data(data, rep) 
            #TEST
            #data.to_pickle(f"/root/.local/refurbishedPTI/measurements/2024-06-25/tests/{self.emission_mono.wavelength}_{rep}.pickle")
            photons += self._count_pulses(data)
        # data = self._osc.channel1.get_trace()
        # TODO: decide if i keep get_data (full dataframe) or just
        # get_trace.
        # osc_screen = self._osc.get_data()
        # photons = self._count_photons(osc_screen[configs.OSC_CHANNEL])
        return photons

    def _count_pulses(self, data):
        return np.where(np.diff(data) > configs.PEAK_THRESHOLD)[0].shape[0]

    def _count_photons(self, data):
        # TODO: save threshold in configuration.
        times = self._find_arrival_times(data)
        return len(times)

    def _find_arrival_times(self, data):
        # TODO: calibrate this
        return data.iloc[np.where(np.diff(data.ch1) > configs.PEAK_THRESHOLD)[0]]

    def set_wavelength(self, wavelength: float):
        return self.emission_mono.set_wavelength(wavelength)

    def set_excitation_wavelength(self, wavelength: float):
        return self.excitation_mono.set_wavelength(wavelength)

    def home(self, excitation=True, emission=True):
        if excitation:
            self.excitation_mono.home()
            print(f"Lamp wavelength should be {self.excitation_mono.home_wavelength}")
        if emission:
            self.emission_mono.home()
            print(
                f"Monochromator wavelength should be {self.emission_mono.home_wavelength}"
            )
        print(
            "If they are wrong, set them with spec.lamp.set_wavelength() and spec.monochromator.set_wavelength()"
        )

    def set_decay_configuration(self, decimation=2):
        trace_duration = self._osc.set_decimation(decimation)
        # TODO: this has to be changed in the Osci API so that you don't have to specify
        # a time when you ask for full buffer
        #trace_duration = self._osc.set_timebase(decimation=decimation)
        self._osc.channel2.enabled = True
        self._osc.channel2.set_gain(5)
        self._osc.configure_trigger(source="ch2", level=1.0, positive_edge=False)
        self._osc.set_trigger_delay(1)
        return trace_duration

    def acquire_decay(self, max_delay=1, step: float = 1, amount_buffers=1, feed=None):
        self.set_decay_configuration()
        arrival_times = np.array([])
        for buff_offset in np.arange(1, max_delay + 1, step):
            print(f"{buff_offset=}")
            self._osc.set_trigger_delay(buff_offset)
            for buff in range(amount_buffers):
                print(f"{buff=}")
                self._osc.arm_trigger()
                data = self._osc.get_data()
                times = np.array(self._find_arrival_times(data).time)
                if feed:
                    feed(times)
                arrival_times = np.hstack((arrival_times, times))
        return pd.DataFrame(dict(arrival_times=arrival_times))
