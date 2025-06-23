# from ucnpexp.instruments import Spectrometer
import copy
import datetime
import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Literal, Optional

import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ipyfilechooser import FileChooser
from IPython.display import HTML, clear_output, display

from . import configs
from .instruments import ITC4020, Monochromator, Spectrometer


@dataclass(frozen=True)
class Measurement:
    starting_wavelength: float
    ending_wavelength: float
    wavelength_step: float
    integration_time: float
    spectrum_type: str
    date: str
    time: str
    name: str
    df: pd.DataFrame
    emission_wavelength: Optional[float] = None
    excitation_wavelength: Optional[float] = None

    def to_csv(self, path):
        print(f"saving to csv at {path}")
        self.save_metadata()
        self.df.to_csv(f"{path}.csv")

    def to_excel(self, path):
        print(f"saving to excel at {path}")
        self.save_metadata()
        self.df.to_excel(f"{path}.xlsx")

    def to_pickle(self, path):
        print(f"saving to pickle at {path}")
        self.save_metadata()
        self.df.to_pickle(f"{path}.pickle")

    def to_dict(self):
        self_dict = asdict(self)
        return self_dict

    def save_metadata(self):
        for attr, value in self.metadata.items():
            self.df.attrs[attr] = value

    @property
    def metadata(self):
        self_dict = self.to_dict()
        del self_dict["df"]
        return self_dict

    def plot_line(self, ax, **kwargs):
        (line,) = ax.plot(
            self.df["wavelength"],
            self.df["counts"] / self.df["integration time"],
            "-o",
            **kwargs,
        )
        return line


class MonoBox(widgets.VBox):
    def __init__(self, mono: Monochromator, name: Literal["Emission", "Excitation"]):
        super().__init__()
        name_static = "Emission" if name == "Excitation" else "Excitation"
        self.mono_wl = widgets.BoundedFloatText(
            value=mono.wavelength,
            min=mono.min_wl,
            max=mono.max_wl,
            description=f"{name_static} monochromator wavelength (nm)",
        )
        self.starting_wl = widgets.BoundedFloatText(
            value=mono.min_wl,
            min=mono.min_wl,
            max=mono.max_wl,
            description="Starting wavelength (nm)",
        )
        self.ending_wl = widgets.BoundedFloatText(
            value=mono.max_wl,
            min=mono.min_wl,
            max=mono.max_wl,
            description="Ending wavelength (nm)",
        )
        self.wl_step = widgets.BoundedFloatText(
            value=mono.wl_step_ratio,
            min=mono.wl_step_ratio,
            max=configs.Gui.MAX_WL_STEP,
            description="Wavelength step (nm)",
        )
        self.name = name
        self.mono = mono
        self.children = [self.mono_wl, self.starting_wl, self.ending_wl, self.wl_step]


class SpectrumMeasurementBox(widgets.VBox):
    def __init__(self, spec: Spectrometer):
        super().__init__()
        self.spec = spec
        self.file_chooser = FileChooser(configs.Gui.MEASUREMENT_PATH)
        self.file_chooser.title = "Measurement filename"
        self.measure_button = widgets.Button(description="Acquire")
        self.measure_button.on_click(self.acquire)
        self.measurement_dropdown = widgets.Dropdown(
            options=["-"],
            value="-",
            description="Selected measurement",
            disabled=False,
        )
        self.save_button = widgets.Button(description="Save")
        self.save_button.on_click(self.save)
        self.delete_button = widgets.Button(description="Delete")
        self.delete_button.on_click(self.delete)
        self.filetype_dropdown = widgets.Dropdown(
            options=[
                ("csv", Measurement.to_csv),
                ("pickle", Measurement.to_pickle),
                ("excel", Measurement.to_excel),
            ],
            value=Measurement.to_pickle,
            description="Save to",
            disable=False,
        )
        self.output = widgets.Output()
        self.fig, self.ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        #self.set_style()
        #with self.output:
        #    self.output.clear_output()
        #    self.fig.show()
        self.children = [
            self.file_chooser,
            self.measure_button,
            self.measurement_dropdown,
            self.save_button,
            self.delete_button,
            self.filetype_dropdown,
            self.output,
        ]

    def set_style(self):
        self.ax.set_xlim(
            [
                min(self.spec.emission_mono.min_wl, self.spec.excitation_mono.min_wl),
                max(self.spec.emission_mono.max_wl, self.spec.excitation_mono.max_wl),
            ]
        )
        self.ax.set_ylim([configs.Gui.MIN_COUNTS, configs.Gui.MAX_COUNTS])
        self.ax.set_title("Spectrum plot")
        self.ax.set_xlabel("Wavelength (nm)", fontsize=15)
        self.ax.set_ylabel("Counts per second (1/s)", fontsize=15)
        self.ax.grid()

    def acquire(self, placeholder):
        print("executing SpectrumeMeasurementBox measure method")
        #self.disable_widgets(True, do_not_disable=["spectrum_type_widget"])
        df = pd.read_pickle("/root/.local/refurbishedPTI/measurements/2024-07-06/350.0_701.0_1.0_0.0003_0.265_0.400.pickle")
        name = str(datetime.datetime.now())
        self.measurement_dropdown.options = [name]
        self.measurement_dropdown.value = name
        self.plot(df, name)
        #self.file_chooser.reset(filename=name)
        #self.disable_widgets(False, do_not_disable=["spectrum_type_widget"])
    
    def plot(self, df, name):
        with self.output:        
            #self.output.clear_output()
            #self.fig.show()
            fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
            ax.plot(df.wavelength, df.counts, '.-', label=name)
            #ax.set_xlim(
            #    [
            #        min(self.spec.emission_mono.min_wl, self.spec.excitation_mono.min_wl),
            #        max(self.spec.emission_mono.max_wl, self.spec.excitation_mono.max_wl),
            #    ]
            #)
            #ax.set_ylim([configs.Gui.MIN_COUNTS, configs.Gui.MAX_COUNTS])
            ax.set_title("Spectrum plot")
            ax.set_xlabel("Wavelength (nm)", fontsize=15)
            ax.set_ylabel("Counts per second (1/s)", fontsize=15)
            ax.grid()
            plt.legend()
            plt.show()

    def save(self):
        print("executing SpectrumeMeasurementBox save method")
        pass

    def delete(self):
        print("executing SpectrumeMeasurementBox delete method")
        pass


class SpectrumTypeStack(widgets.Stack):
    def __init__(
        self, dropdown: widgets.Dropdown, emission_box: MonoBox, excitation_box: MonoBox
    ):
        super().__init__()

        self.dropdown = dropdown
        self.emission_box = emission_box
        self.excitation_box = excitation_box
        self.children = [self.emission_box, self.excitation_box]
        self.link = widgets.jslink((self.dropdown, "index"), (self, "selected_index"))


class SpectrumPlotBox(widgets.VBox):
    def __init__(self, spec: Spectrometer):
        super().__init__()
        self.output = widgets.Output()
        self.spec = spec

        with self.output:
            self.fig, self.ax = plt.subplots(
                constrained_layout=True,
                figsize=(10, 5),
            )
            #plt.close(self.fig)
        self.set_style()

    def set_style(self):
        self.ax.set_xlim(
            [
                min(self.spec.emission_mono.min_wl, self.spec.excitation_mono.min_wl),
                max(self.spec.emission_mono.max_wl, self.spec.excitation_mono.max_wl),
            ]
        )
        self.ax.set_ylim([configs.Gui.MIN_COUNTS, configs.Gui.MAX_COUNTS])
        self.ax.set_title("Spectrum plot")
        self.ax.set_xlabel("Wavelength (nm)", fontsize=15)
        self.ax.set_ylabel("Counts per second (1/s)", fontsize=15)
        self.ax.grid()


class LifetimePlotBox(widgets.VBox):
    def __init__(self, spec: Spectrometer, itc: ITC4020):
        super().__init__()
        self.spec = spec
        self.itc = itc
        self.output = widgets.Output()
        with self.output:
            self.fig, self.ax = plt.subplots(
                constrained_layout=True,
                figsize=(10, 5),
            )
            #plt.close(self.fig)
        self.set_style()

    def set_style(self):
        max_time_secs = 1 / self.itc.frequency * (1 - self.itc.duty_cycle)
        max_time_microsecs = max_time_secs * 1e6
        self.ax.set_xlim([0, max_time_microsecs])
        self.ax.set_ylim([configs.Gui.MIN_COUNTS, configs.Gui.MAX_COUNTS])
        self.ax.set_title("Lifetime plot")
        self.ax.set_xlabel("Time ($\mu$s)", fontsize=15)
        self.ax.set_ylabel("Counts", fontsize=15)
        self.ax.grid()


class SpectrumBox(widgets.VBox):
    def __init__(
        self,
        spec_type_dropdown: widgets.Dropdown,
        spec_type_stack: SpectrumTypeStack,
        spec_measurement_box: SpectrumMeasurementBox,
        spec_plot_box: SpectrumPlotBox,
    ):
        super().__init__()
        self.spec_type_dropdown = spec_type_dropdown
        self.spec_type_stack = spec_type_stack
        self.spec_measurement_box = spec_measurement_box
        #self.spec_plot_box = spec_plot_box
        self.output = widgets.Output()
        #with self.output:
        #    self.output.clear_output()
        #    plt.plot(np.linspace(0, 1, 100), np.linspace(0,1,100))
        #    plt.show()
        self.children = [
            self.spec_type_dropdown,
            self.spec_type_stack,
            self.spec_measurement_box,
            #self.output,
            #self.spec_plot_box,
        ]


class ITCBox(widgets.VBox):
    def __init__(self, itc: ITC4020):
        super().__init__()

        self.itc = itc
        self.power = widgets.BoundedFloatText(
            value=0,
            min=0,
            max=1,
            description="Pump power (mW)",
        )
        self.frequency = widgets.BoundedFloatText(
            value=configs.Gui.min_freq,
            min=configs.Gui.min_freq,
            max=configs.Gui.max_freq,
            description="Frequency (Hz)",
        )
        self.duty_cycle = widgets.BoundedFloatText(
            value=configs.Gui.min_duty_cycle,
            min=configs.Gui.min_duty_cycle,
            max=configs.Gui.max_duty_cycle,
            description="Duty Cycle (%)",
        )
        self.children = [
            self.power,
            self.frequency,
            self.duty_cycle,
        ]


class LifeMeasurementBox(widgets.VBox):
    def __init__(self, spec: Spectrometer, itc: ITC4020):
        super().__init__()
        self.file_chooser = FileChooser(configs.Gui.MEASUREMENT_PATH)
        self.file_chooser.title = "Measurement filename"
        self.acquire_button = widgets.Button(description="Acquire")
        self.acquire_button.on_click(self.acquire)
        self.measurement_dropdown = widgets.Dropdown(
            options=["-"],
            value="-",
            description="Selected measurement",
            disabled=False,
        )
        self.save_button = widgets.Button(description="Save")
        self.save_button.on_click(self.save)
        self.delete_button = widgets.Button(description="Delete")
        self.delete_button.on_click(self.delete)
        self.filetype_dropdown = widgets.Dropdown(
            options=[
                ("csv", Measurement.to_csv),
                ("pickle", Measurement.to_pickle),
                ("excel", Measurement.to_excel),
            ],
            value=Measurement.to_pickle,
            description="Save to",
            disable=False,
        )
        self.output = widgets.Output()
        self.children = [
            self.file_chooser,
            self.acquire_button,
            self.measurement_dropdown,
            self.save_button,
            self.delete_button,
            self.filetype_dropdown,
            self.output,
        ]

    def acquire(self, placeholder):
        print("executing SpectrumeMeasurementBox measure method")
        #self.disable_widgets(True, do_not_disable=["spectrum_type_widget"])
        df = pd.read_pickle("/root/.local/refurbishedPTI/measurements/2024-07-06/378_180_4_0.4_0.287982792.pickle")
        name = str(datetime.datetime.now())
        self.measurement_dropdown.options = [name]
        self.measurement_dropdown.value = name
        self.plot(df, name)
        #self.file_chooser.reset(filename=name)
        #self.disable_widgets(False, do_not_disable=["spectrum_type_widget"])
    
    def plot(self, df, name):
        with self.output:        
            #self.output.clear_output()
            #self.fig.show()
            fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
            ax.hist(df.arrival_times*1e6, label=name)
            #ax.set_xlim(
            #    [
            #        min(self.spec.emission_mono.min_wl, self.spec.excitation_mono.min_wl),
            #        max(self.spec.emission_mono.max_wl, self.spec.excitation_mono.max_wl),
            #    ]
            #)
            #ax.set_ylim([configs.Gui.MIN_COUNTS, configs.Gui.MAX_COUNTS])
            ax.set_title("Lifetime plot")
            ax.set_xlabel("time ($\mu$s)", fontsize=15)
            ax.set_ylabel("Counts", fontsize=15)
            ax.grid()
            plt.legend()
            plt.show()

    def save(self):
        print("executing SpectrumeMeasurementBox save method")
        pass

    def delete(self):
        print("executing SpectrumeMeasurementBox delete method")
        pass


class LifetimeParametersBox(widgets.VBox):
    def __init__(self, mono: Monochromator):
        super().__init__()

        self.life_emission_wl = widgets.BoundedFloatText(
            value=mono.wavelength,
            min=mono.min_wl,
            max=mono.max_wl,
            description="Emission monochromator wavelength (nm)",
        )
        self.amount_counts = widgets.BoundedIntText(
            value=configs.Gui.lifetime_default_counts,
            min=configs.Gui.lifetime_min_counts,
            max=configs.Gui.lifetime_max_counts,
            description="Amount of counts",
        )
        self.starting_time = widgets.BoundedFloatText(
            value=0,
            min=0,
            max=8,
            description="Starting time (ms)",
        )
        self.eding_time = widgets.BoundedFloatText(
            value=2,
            min=0,
            max=8,
            description="Ending time (ms)",
        )
        self.children = [
            self.life_emission_wl,
            self.amount_counts, 
            self.starting_time, 
            self.eding_time
            ]


class LifetimeBox(widgets.VBox):
    def __init__(
        self,
        itc_box: ITCBox,
        life_parameters_box: LifetimeParametersBox,
        life_measurement_box: LifeMeasurementBox,
        life_plot_box: LifetimePlotBox,
    ):
        super().__init__()
        self.itc_box = itc_box
        self.life_parameters_box = life_parameters_box
        self.life_measurement_box = life_measurement_box
        #self.life_plot_box = life_plot_box
        #self.output = widgets.Output()
        #with self.output:
        #    self.output.clear_output()
        #    plt.plot(np.linspace(0, 1, 100), np.linspace(0,1,100)**2)
        #    plt.show()
        self.children = [
            self.itc_box,
            self.life_parameters_box,
            self.life_measurement_box,
            #self.output,
            #self.life_plot_box,
        ]


#class MeasTypeStack(widgets.Stack):
#    def __init__(
#        self,
#        dropdown: widgets.Dropdown,
#        life_box: LifetimeBox,
#        spec_box: SpectrumBox,
#    ):
#        super().__init__()
#        self.output = widgets.Output()
#        self.dropdown = dropdown
#        self.life_box = life_box
#        self.spec_box = spec_box
#        self.children = [self.life_box, self.spec_box]
#        self.link = widgets.jslink((self.dropdown, "index"), (self, "selected_index"))
        #self.dropdown.observe = self.update_display()

    #def update_display(self):
    #    def _update_display(change):
    #        with self.output:
    #            clear_output(wait=True)
    #            if change["new"] == "Lifetime":
    #                display(self.life_box)
    #                display(self.life_box.life_plot_box.fig)
    #            elif change["new"] == "Spectrum":
    #                display(self.spec_box)
    #                display(self.spec_box.spec_plot_box.fig)

    #    return _update_display

class MeasTypeBox(widgets.VBox):
    def __init__(
        self,
        dropdown: widgets.Dropdown,
        life_box: LifetimeBox,
        spec_box: SpectrumBox,
    ):
        super().__init__()
        self.output = widgets.Output()
        self.dropdown = dropdown
        self.life_box = life_box
        self.spec_box = spec_box
        self.children = [self.life_box, self.spec_box]
        self.link = widgets.jslink((self.dropdown, "index"), (self, "selected_index"))
        #self.dropdown.observe = self.update_display()

    def update_display(self):
        def _update_display(change):
            with self.output:
                clear_output(wait=True)
                if change["new"] == "Lifetime":
                    display(self.life_box)
                    display(self.life_box.life_plot_box.fig)
                elif change["new"] == "Spectrum":
                    display(self.spec_box)
                    display(self.spec_box.spec_plot_box.fig)

        return _update_display


class Gui(widgets.VBox):
    def __init__(self):
        super().__init__()
        from .dummy_instruments import DummyITC as ITC4020
        from .dummy_instruments import DummySpectrometer as Spectrometer

        self.spec = Spectrometer.constructor_default()
        self.itc = ITC4020()

        self.emission_box = MonoBox(self.spec.emission_mono, name="Emission")
        self.excitation_box = MonoBox(self.spec.excitation_mono, name="Excitation")
        self.spec_type_dropdown = widgets.Dropdown(
            options=["Emission", "Excitation"], description="Spectrum type"
        )
        self.spec_type_stack = SpectrumTypeStack(
            self.spec_type_dropdown,
            self.emission_box,
            self.excitation_box,
        )

        self.spec_measurement_box = SpectrumMeasurementBox(self.spec)
        #self.spec_plot_box = SpectrumPlotBox(self.spec)
        self.spec_box = SpectrumBox(
            self.spec_type_dropdown,
            self.spec_type_stack,
            self.spec_measurement_box,
            0,
            #self.spec_plot_box,
        )

        self.life_itc_box = ITCBox(self.itc)
        self.life_parameters_box = LifetimeParametersBox(self.spec.emission_mono)
        self.life_measurement_box = LifeMeasurementBox(self.spec, self.itc)
        #self.life_plot_box = LifetimePlotBox(self.spec, self.itc)
        self.life_box = LifetimeBox(
            self.life_itc_box,
            self.life_parameters_box,
            self.life_measurement_box,
            0,
            #self.life_plot_box,
        )

        self.meas_type_options = ["Lifetime", "Spectrum"]
        self.meas_type_dropdown = widgets.Dropdown(
            options=self.meas_type_options,
            description="Measurement type",
            value="Lifetime",
        )

        self.meas_type_dropdown.observe(self.meas_type_dropdown_on_change(), names="value")
        #self.meas_type_stack = MeasTypeStack(
        #    self.meas_type_dropdown, self.life_box, self.spec_box
        #)
        self.meas_type_stack = widgets.Stack(children=[self.life_box, self.spec_box])
        #self.meas_type_dropdown.observe(self.meas_type_stack.update_display())

        self.children = [self.meas_type_dropdown, self.meas_type_stack]
        layout = widgets.Layout(width="auto")
        style = {"description_width": "initial"}
        self.meas_type_dropdown.style = style
        self.meas_type_dropdown.layout = layout
        set_style_children(self.meas_type_stack, style)
        set_layout_children(self.meas_type_stack, layout)
        display(self)

    def meas_type_dropdown_on_change(self):
        def _meas_type_dropdown_on_change(change):
            print(f"{self.meas_type_stack.selected_index=} {change.new=}")
            print(f"{self.meas_type_options=}")
            self.meas_type_stack.selected_index = self.meas_type_options.index(change.new)
        return _meas_type_dropdown_on_change


def set_style_children(widget: widgets.widget, style: dict):
    try:
        children = widget.children
        if not children:
            widget.style = style
        else:
            for child in children:
                set_style_children(child, style)
    except AttributeError:
        widget.style = style


def set_layout_children(widget: widgets.widget, layout: widgets.Layout):
    try:
        children = widget.children
        if not children:
            widget.layout = layout
        else:
            for child in children:
                set_layout_children(child, layout)
    except AttributeError:
        widget.layout = layout
