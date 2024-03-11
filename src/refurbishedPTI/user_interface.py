import cmd
from typing import IO
import yaml

class SpectrometerCalibrationInterface(cmd.Cmd):
    intro = '''
Spectrometer calibration interface. This interface aims to help you generate a calibration yaml file for the Spectrometer object. \n
Type help or ? to list commands.\n'''
    prompt = '(spec-calib)'
    file = None

    def __init__(self, spec):
        super(SpectrometerCalibrationInterface, self).__init__()
        self.spec = spec
        self.calibration = {"max_wl":None, "min_wl":None, "wl_step_ratio":None, "greater_wl_cw":None}

    def do_maxwl(self, max_wl: float):
        'Set maximum Spectrometer wavelength'
        try:
            max_wl = float(max_wl)
            #self.spec._max_wl = max_wl
            self.calibration["max_wl"] = max_wl
        except:
            print("Wrong argument")
            print("max_wl should be float")

    def do_minwl(self, min_wl: float):
        'Set maximum Spectrometer wavelength'
        try:
            min_wl = float(min_wl)
            #self.spec._min_wl = min_wl
            self.calibration["min_wl"] = min_wl
        except:
            print("Wrong argument")
            print("min_wl should be float")
    
    def do_growth_direction(self, arg):
        'Set wavelength growth direction for the monocromator motor'
        self.gd_ui = GrowthDirection(self.spec, self.calibration)
        self.gd_ui.cmdloop()

    def do_wavelength_to_step_ratio(self, arg):
        'Set wavelength to step ratio for the spectrometer'
        self.wl_to_step_ui = WavelengthStepRatio(self.spec, self.calibration)
        self.wl_to_step_ui.cmdloop()

    def do_quit(self, arg):
        'Quit calibration menu'
        return True
    
    def do_print_calibration(self, arg):
        'Prints currently configured calibartion'
        directions = {True:"Clockwise", False:"Counter Clockwise", None:"None"}
        for key in self.calibration:
            print(f"{key}:\t\t\t{self.calibration[key]}")
    
    def do_save_to_yaml(self, path):
        'Save calibration to yaml.\nInput filepath.'
        if None in self.calibration.values():
            print("At least one configuration parameter is not set.")
            print("Set them first.")
            return
        try: 
            path = str(path)
            # Esto no se hace pero bueno por ahora lo hago así
            self.spec.calibration_path = path
            with open(path, "w") as f:
                yaml.dump(self.calibration, f)
        except:
            print("Wrong argument")
            print("You must input the file path for the calibration file")

class WavelengthStepRatio(cmd.Cmd):
    intro='''
Wavelength to step ratio of the spectrometer configuration menu.
Type help or ? to list commands.

This menu aims to help you determine what is the value of the ratio
(wl2 - wl1)/steps
that determines how much does wavelength change (from wl1 to wl2)
for a given step change.
    '''
    prompt='(wavelength-step-ratio)'

    def __init__(self, spec, calibration_dict):
        super(WavelengthStepRatio, self).__init__()
        self.spec = spec
        self.calibration_dict = calibration_dict
        self._direction_dict = {True:"clockwise", False:"counter clockwise"}

    def do_start(self, arg):
        'Start calibration of the wavelength to step ratio'
        done = False
        while not done:
            self.initial_wavelength = self.ask_wavelength()
            self.rot_steps()
            self.final_wavelength = self.ask_wavelength()
            done = self.ask_done()
        print("Calculating...")
        wl_step_ratio = self.calculate_ratio()
        print(f"Setting wavelength to step ratio to {wl_step_ratio} nm/step")
        #self.spec._wl_step_ratio = wl_deg_ratio
        self.calibration_dict["wl_step_ratio"] = wl_step_ratio
        self.onecmd('quit')

    def do_quit(self, arg):
        'Go back to main calibration menu'
        return True

    def calculate_ratio(self):
        ratio = (self.final_wavelength - self.initial_wavelength)/self.rotation_steps
        print(self.final_wavelength, self.initial_wavelength, self.rotation_steps)
        return ratio

    def ask_done(self):
        print("Calibration parameters you input:")
        print(f"Initial wavelength:\t\t{self.initial_wavelength}")
        print(f"Rotation steps:\t\t{self.rotation_steps}")
        print(f"Final wavelength:\t\t{self.final_wavelength}")
        answer = input("Do you want to calculate wavelength to degree ratio?[Y/n]").lower()
        done = answer in ["y", ""]
        if not done:
            print("Aborting...")
        return done

    def rot_steps(self):
        steps = input("Input the amount of steps the motor will rotate clockwise (+) or anti clockwise (-): ")
        try:
            steps = int(steps)
            print(steps)
            self.cw, self.rotation_steps = steps > 0, abs(steps)
            self.spec._motor.rotate_step(self.rotation_steps, self.cw)
            print(f"Rotate {self.rotation_steps} steps {self._direction_dict[self.cw]}")
        except Exception as e:
            print(e)
            print("Wrong argument")
            print("Rotation steps should be an int")
    
    def ask_wavelength(self):
        keep_going  = True
        while keep_going:
            wavelength = input("Input current wavelength in nm: ")
            try:
                wavelength = float(wavelength)
                print(f"The spectrometer is now at {wavelength} nm")
                keep_going = False
            except:
                print("Wrong argument")
                print("wavelenght should be a float")
        return wavelength

    def do_set_ratio(self, arg):
        print("Calculating wavelength to degree ratio")
        print(f"Initial wavelength:\t {self.initial_wavelength}")
        print(f"Final wavelength:\t {self.final_wavelength}")
        print(f"Rotation steps:\t {self.rotation_steps}")
        ratio = (self.final_wavelength - self.initial_wavelength)/self.rotation_steps
        print(f"(final_wl - initial_wl)/steps = {ratio}")
        #self.spec._wl_deg_ratio = ratio
        self.calibration_dict["wl_step_ratio"] = ratio
        
    
class GrowthDirection(cmd.Cmd):
    intro='''
Growth direction of the monochromator motor configuration menu.
Type help or ? to list commands.

This menu aims to help you determine wether the spectrometer wavelength increases clockwise (True)
or counterclockwise (False).
Input l to turn 10 steps counter clockwise.
Input r to turn 10 steps clockwise.
Input L to turn 100 steps counter clockwise.
Input R to turn 100 steps clockwise.
Input set_growth_direction to set the growth direction.
    '''
    prompt=f'(growth-direction)'

    def __init__(self, spec, calibration_dict):
        super(GrowthDirection, self).__init__()
        self.spec = spec
        self.calibration_dict = calibration_dict
        self.steps = 1
        self.small_rotation = 10
        self.large_rotation = 100

    def do_r(self, arg):
        'Rotate monochromator motor 10 steps clockwise'
        self.spec._motor.rotate_step(self.small_rotation, True)

    def do_l(self, arg):
        'Rotate monochromator motor 10 steps clockwise'
        self.spec._motor.rotate_step(self.small_rotation, False)

    def do_R(self, arg):
        'Rotate monochromator motor 100 steps clockwise'
        self.spec._motor.rotate_step(self.large_rotation, True)

    def do_L(self, arg):
        'Rotate monochromator motor 100 steps clockwise'
        self.spec._motor.rotate_step(self.large_rotation, False)

    def do_set_growth_direction(self, cw: str):
        '''Set wavelength growth direction.\n
Usage: Input True for clowckwise or False for counter clockwise'''
        cw = cw.lower()
        if cw == "true":
            #self.spec._greater_wl_cw = True
            self.calibration_dict["greater_wl_cw"] = True
            return True
        elif cw == "false":
            #self.spec._greater_wl_cw = False
            self.calibration_dict["greater_wl_cw"] = False
            return True
        else:
            print("Wrong argument")
            self.onecmd("help set_growth_direction")

class TestMotor:
    def __init__(self):
        self.directions = {True:"cw", False:'ccw'}

    def rotate_step(self, steps, direction):
        print(f"rotate {steps} steps in {self.directions[direction]} direction")

    def rotate_relative(self, angle):
        print(f"rotate {angle}")
        return angle

class TestSpec:
    def __init__(self):
        self._motor = TestMotor()
        self._max_wl = None
        self._min_wl = None
        self._wl_deg_ratio = None
        self._greater_wl_cw = None

    def calibrate(self):
        ui = SpectrometerCalibrationInterface(self)
        ui.cmdloop()

if __name__ == '__main__':
    spec = TestSpec()
    hola = spec.calibrate()

