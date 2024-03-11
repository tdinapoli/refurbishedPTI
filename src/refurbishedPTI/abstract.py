from abc import ABC, abstractmethod
import time


class Motor(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def rotate(self, angle: float, cw: bool):
        pass
        # informarle al driver cuantos pulsos y direccion

class Spectrometer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def set_wavelength(self, wavelength: float):
        pass
    
    @abstractmethod
    def goto_wavelength(self, wavelength):
        pass

#    @abstractmethod
#    def _goto_wavelength(self, wavelength):
#        pass


class TTL:

    def __init__(self, state):
        self.state = state

    @property
    def state(self):
        return self._get_state()
    
    @state.setter
    def state(self, state):
        self._set_state(state)
    
    @abstractmethod
    def _get_state(self):
        pass
       
    @abstractmethod
    def _set_state(self, state):
        pass
    
    def toggle(self) -> None:
        self.state = not self.state

    def pulse(self, duration) -> None:
        self.toggle()
        time.sleep(duration)
        self.toggle()
    
    def __str__(self):
        return str(self.state)
    

class MotorDriver(ABC):
    step: TTL

    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def step(self):
        pass


