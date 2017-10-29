import threading

import matplotlib.figure
import time
import numpy as np
import pyaudio
from kivy.app import App
from kivy.clock import mainthread
from kivy.core.image import Image
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.graphics.context_instructions import Color
from kivy.graphics.vertex_instructions import Rectangle
from kivy.uix.slider import Slider
from kivy.uix.widget import Widget

CHUNKSIZE = 2048
RATE = 44100

avatar = None
volume = 0.7

figure = matplotlib.figure.Figure()
lines = []

axs = figure.subplots(2, 1)

lines.extend(axs[0].plot(np.zeros(CHUNKSIZE)))
axs[0].set_xlim((0, CHUNKSIZE))
axs[0].set_ylim((-.2, .2))

FFT_CHUNK = 10 * CHUNKSIZE

fftfreq = np.fft.fftfreq(FFT_CHUNK, d=RATE / CHUNKSIZE / 1000000)
lines.extend(axs[1].plot(fftfreq[:FFT_CHUNK // 2], np.zeros(FFT_CHUNK // 2)))
axs[1].plot([0, FFT_CHUNK], [1, 1], "r-")
axs[1].set_ylim((0, 2))
axs[1].set_xlim((0, 3000))


@mainthread
def plot(y):
    lines[0].set_ydata(y)

    fft = np.fft.fft(y, n=FFT_CHUNK)
    fft = np.sqrt(np.abs(fft))

    lines[1].set_ydata(fft[:FFT_CHUNK // 2])

    figure.canvas.draw()

    max_freq = fftfreq[np.argmax(fft[:FFT_CHUNK // 2])]
    max_freq_vol = fft[np.argmax(fft[:FFT_CHUNK // 2])]
    if max_freq_vol > 1 and max_freq > 100:
        avatar.mouth = "open"
    else:
        avatar.mouth = "closed"


class Avatar(Widget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        global avatar
        avatar = self

        self._textures = {}

        with self.canvas:
            Color(0., 1., 0.)
            self._background = Rectangle(pos=self.pos, size=(self.width, self.width))
            Color(1., 1., 1.)
            self._body = Rectangle(texture=self.get_texture("body"), pos=self.pos, size=(self.width, self.width))
            self._eyes = Rectangle(texture=self.get_texture("eyes.open"), pos=self.pos, size=(self.width, self.width))
            self._mouth = Rectangle(texture=self.get_texture("mouth.closed"), pos=self.pos, size=(self.width, self.width))

        self.bind(size=self._update_rect, pos=self._update_rect)

    def _update_rect(self, instance, value):
        self._background.pos = instance.pos
        self._background.size = instance.size

        mn = min(instance.width, instance.height)
        size = (mn, mn)
        pos = (instance.x + (instance.width - mn) / 2, instance.y)

        self._body.pos = pos
        self._body.size = size
        self._eyes.pos = pos
        self._eyes.size = size
        self._mouth.pos = pos
        self._mouth.size = size

    def get_texture(self, name):
        if name not in self._textures:
            self._textures[name] = Image("data/images/layers/{}.png".format(name)).texture
        return self._textures[name]

    @property
    @mainthread
    def eyes(self):
        raise NotImplementedError()

    @eyes.setter
    @mainthread
    def eyes(self, value):
        self._eyes.texture = self.get_texture("eyes.{}".format(value))

    @property
    @mainthread
    def mouth(self):
        raise NotImplementedError()

    @mouth.setter
    @mainthread
    def mouth(self, value):
        self._mouth.texture = self.get_texture("mouth.{}".format(value))


class Plot(FigureCanvasKivyAgg):
    def __init__(self, *args, **kwargs):
        super().__init__(figure, *args, **kwargs)


class ConceptApp(App):
    pass


class VolumeSlider(Slider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.value = volume

        self.bind(value=self.on_value_change)

    def on_value_change(self, instance, value):
        global volume
        volume = value


def plotter():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=RATE,
        input=True,
        output=True,
        frames_per_buffer=CHUNKSIZE,
    )

    i = 0
    while True:
        data = stream.read(CHUNKSIZE)
        numpydata = np.fromstring(data, dtype=np.float32)
        numpydata *= volume
        # fft = np.abs(np.fft.fft(numpydata, n=10*CHUNKSIZE))
        # ifft = np.fft.ifft(fft)[:CHUNKSIZE]
        # stream.write(ifft.astype(np.float32).tostring())

        if i % 2 == 0:
            plot(numpydata)
        i += 1


plotter_thread = threading.Thread(target=plotter)
plotter_thread.daemon = True
plotter_thread.start()


def blinker():
    while True:
        time.sleep(6)
        avatar.eyes = "closed"
        time.sleep(.1)
        avatar.eyes = "open"


blinker_thread = threading.Thread(target=blinker)
blinker_thread.daemon = True
blinker_thread.start()

ConceptApp().run()
