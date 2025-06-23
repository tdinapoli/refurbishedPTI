import numpy as np
import pandas as pd

from . import abstract


class DummyITC:
    def __init__(self):
        self.frequency = 100
        self.duty_cycle = 80/100

class DummyMonochromator:
    def __init__(
        self,
        min_wl=200,
        max_wl=750,
        wl_step_ratio=0.5,
        home_wavelength=256.6,
    ):
        self._home_wavelength = home_wavelength
        self.min_wl = min_wl  
        self.max_wl = max_wl
        self.wl_step_ratio = wl_step_ratio

    def goto_wavelength(self, wl):
        #print(f"goin to {wl}")
        self.set_wavelength(wl)
        return wl

    def set_wavelength(self, wl):
        #print(f"setting wl to {wl}")
        self.wavelength = wl

    def home(self):
        #print(f"homing to {self._home_wavelength}")
        self.set_wavelength(self._home_wavelength)

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


class DummySpectrometer(abstract.Spectrometer):
    def __init__(self, excitation_mono, emission_mono):
        self.excitation_mono = excitation_mono
        self.emission_mono = emission_mono
        self.excitation_mono.home()
        self.emission_mono.home()

    def set_wavelength(self, wavelength: float):
        self.monochromator.set_wavelength(wavelength)

    def goto_wavelength(self, wavelength):
        self.monochromator.goto_wavelength(wavelength)

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
            photons, time_measured = self.get_intensity(integration_time, rounds)
            yield dict(wavelength=wl, counts=photons, integration_time=time_measured)

    def get_intensity(self, integration_time, rounds):
        return np.random.randint(0, 10), rounds * integration_time

    @classmethod
    def constructor_default(cls):
        lamp = DummyMonochromator()
        monochromator = DummyMonochromator(min_wl=150, max_wl=800, home_wavelength=176)
        return DummySpectrometer(lamp, monochromator)
