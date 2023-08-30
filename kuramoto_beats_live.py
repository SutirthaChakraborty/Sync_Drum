import time
import pyaudio
import aubio
import numpy as np

win_s = 2048
hop_s = win_s // 2
filename = "a.wav"
size_difference = 0.8
samplerate = 44100
out_samplerate = int(samplerate*size_difference)
phase_correction_samples = 4
phase_difference_collection = np.zeros((phase_correction_samples))
coupling_factor = 20
base_frequency = np.pi*4
error_threshold = base_frequency/50
machine_phase_diff = -np.pi/10
const_phase_difference = -np.pi*2*(size_difference) + machine_phase_diff

f = []
stream = None


class Oscillator:
    def __init__(self):
        self.phase = 0
        self.beat = 0
        self.last_beat = 0

    def set_beat(self):
        self.last_beat = self.beat
        self.beat = time.time()

    def get_phase(self):
        phase = 0
        if self.last_beat != 0:
            t = time.time()
            phase = (self.beat - t)/(self.last_beat - self.beat)*(2*np.pi)
        return(phase % (np.pi*2))


class BeatManager:
    def __init__(self):
        self.BPM = -1
        self.input = Oscillator()
        self.oscillator = Oscillator()
        self.last_update = 0
        self.epoch = 0
        self.frequency = base_frequency
        self.phase = 0
        self.phase_correction = 0
        self.play_beat = False

    def set_BPM(self, _BPM):
        self.BPM = _BPM

    def should_play_beat(self):
        if self.play_beat:
            self.play_beat = False
            return True
        return False

    def set_beat(self):
        self.input.set_beat()

    def update_phase(self, rad):
        cp = self.get_corrected_phase()
        if ((cp + rad)) > (np.pi*2) - error_threshold:
            self.play_beat = True
        self.phase += rad
        self.phase %= (np.pi*2)

    def get_phase(self):
        return (self.phase+const_phase_difference)

    def get_corrected_phase(self):
        return (self.phase+self.phase_correction+const_phase_difference) % (2*np.pi)

    def update(self):
        global phase_difference_collection
        t = time.time()
        delta = t - self.last_update
        self.last_update = t
        self.epoch += delta
        beat_phase = self.get_phase()
        input_phase = self.input.get_phase()
        delta_phase = self.frequency

        phase_difference = beat_phase - input_phase

        if np.abs(phase_difference) > error_threshold:
            delta_phase = self.frequency + \
                coupling_factor/2*(np.sin(phase_difference))
            self.BPM = np.abs(delta_phase*60/(2*np.pi))

        self.update_phase(delta_phase*delta)

        phase_difference_collection = np.append(
            phase_difference_collection, phase_difference)[1:]
        self.phase_correction = np.average(phase_difference_collection)


a_tempo = aubio.tempo("default", win_s, hop_s, samplerate)

click = 0.7 * np.sin(2. * np.pi * np.arange(hop_s) /
                     hop_s * samplerate / 3000.)

silence = 0 * np.zeros((hop_s))

beat_manager = BeatManager()


def pyaudio_callback(_in_data, _frame_count, _time_info, _status):

    data = stream.read(1024)
    samples = np.fromstring(data,
                            dtype=aubio.float_type)

    is_beat = a_tempo(samples)
    if is_beat:
        beat_manager.set_beat()
    beat_manager.update()

    samples *= 0

    if beat_manager.should_play_beat():
        samples += click
        pass


    audiobuf = samples.tobytes()
    return (audiobuf, pyaudio.paContinue)


def main():
    global stream
    p = pyaudio.PyAudio()
    pyaudio_format = pyaudio.paFloat32
    frames_per_buffer = hop_s
    n_channels = 1

    stream = p.open(format=pyaudio.paFloat32,
                    channels=1, rate=samplerate, input=True, output=True,
                    frames_per_buffer=1024)

    out_stream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(samplerate*size_difference),
                        output=True, frames_per_buffer=frames_per_buffer,
                        stream_callback=pyaudio_callback)

    stream.start_stream()

    out_stream.start_stream()

    while stream.is_active():
        pass

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    main()
