import numpy as np
import pyaudio
import aubio
import time
import librosa
from pythonosc import udp_client

import matplotlib.animation as animation
import matplotlib.pyplot as plt

win_s = 1024
hop_s = win_s // 2
samplerate = 44100
beat_vol = 0.0005
sender_host = '127.0.0.1'
sender_port = 4560
sender_url = '/live_loop/foo'

click = beat_vol * np.sin(2. * np.pi * np.arange(hop_s) /
                          hop_s * samplerate / 3000.)

graph_update_interval = 500
stabilization_time = 10*1000
delta_mult = 0.04
rad_per_cycle = np.math.pi/8
beats_per_cycle = 0.5

# base class for simulations
class Simulation:
    def __init__(self, initial_conditions=np.zeros([2, 1])):
        self.state = np.copy(np.array(initial_conditions, dtype=float))
        self.setup()
        self.beat_track = None
        self.current_beat = None
        self.live_beats = []
        self.bpm = 0

        # in mode 0 is file
        # in mode 1 is live
        self.in_mode = 0

        # out mode 0 is normal output
        # out mode 1 is sonic Pi bpm
        self.out_mode = 0

    def set_in_mode(self, mode):
        self.in_mode = mode

    def set_out_mode(self, mode):
        self.out_mode = mode
        if mode == 1:
            self.sender = udp_client.SimpleUDPClient(sender_host, sender_port)
    def set_beat_track(self, _beat_track):
        self.beat_track = _beat_track
        self.current_beat = 0

    def get_beat(self):
        processed_data = 0
        i = 0
        while(i < len(self.live_beats)):
            beat = self.live_beats[i]

            processed_data += self.beat_track[beat[0]][beat[1]]

            beat[1] += 1
            if (beat[1] < len(self.beat_track[beat[0]])):
                i += 1
            else:
                self.live_beats.pop(i)
        return processed_data

    def progress_beat(self):
        self.current_beat += 1
        self.current_beat %= len(self.beat_track)
        self.live_beats.append([self.current_beat, 0])

    def run(self, iterations, delta, c):
        t_state = np.copy(self.state)
        w_record = np.array([t_state[0]])

        while iterations > 0:
            self.simulate(t_state, delta, c)
            w_record = np.append(w_record, [t_state[0]], axis=0)
            self.post_processing(t_state)
            iterations -= 1

        self.plot(w_record)
        return(w_record)

    def run_live(self, file_path, c):
        p = pyaudio.PyAudio()
        pyaudio_format = pyaudio.paFloat32
        frames_per_buffer = hop_s
        n_channels = 1
        self.t_state = np.copy(self.state)
        self.history = np.array(
            [[np.copy(self.state[0])], [np.copy(self.state[1])]], dtype=float)
        self.timestamps = [0]
        self.live_w = 0
        self.live_theta = 0
        self.theta = 0
        self.last_beat = 0

        if self.in_mode == 0:
            # self.a_source = librosa.load(file_path, 22050, True, 0, 0.02321)
            self.a_source = aubio.source(file_path, samplerate, hop_s)
        elif self.in_mode == 1:
            self.in_stream = p.open(format=pyaudio.paFloat32,
                                    channels=1, rate=samplerate, input=True, output=True,
                                    frames_per_buffer=hop_s)

        self.a_tempo = aubio.tempo("default", win_s, hop_s, samplerate)
        fig = plt.figure()
        self.ax = fig.add_subplot(1, 1, 1)
        self.start_time = time.time()
        self.last_time_live = None
        self.last_time = time.time()
        self.c = c

        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=samplerate,
                        output=True, frames_per_buffer=frames_per_buffer,
                        stream_callback=self.live_sim)

        stream.start_stream()
        ani = animation.FuncAnimation(
            fig, self.live_graph, interval=graph_update_interval)
        plt.show()

        while stream.is_active():
            time.sleep(0.1)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def live_sim(self, _in_data, _frame_count, _time_info, _status):
        if self.in_mode == 0:
            samples, read = self.a_source()
            is_beat = self.a_tempo(samples)

        elif self.in_mode == 1:
            in_data = self.in_stream.read(hop_s)
            samples = np.array(np.frombuffer(in_data, dtype=aubio.float_type))
            read = hop_s
            is_beat = self.a_tempo(samples)
            samples *= 0

        if self.out_mode == 0:
            if(self.beat_track != None):
                samples += self.get_beat()

        if time.time() < self.start_time + stabilization_time:
            if self.out_mode == 0:
                if self.should_play_beat():
                    if(self.beat_track != None):
                        self.progress_beat()
                    else:
                        samples += click

            elif self.out_mode == 1:
                self.sender.send_message(sender_url, [self.bpm])

            if is_beat:
                if self.last_time_live != None:
                    delta = time.time() - self.last_time_live
                    self.bpm = 60//delta
                    self.live_theta += rad_per_cycle
                    self.live_w = (rad_per_cycle)/delta
                    self.last_time_live += delta
                else:
                    self.last_time_live = time.time()

                delta = time.time() - self.last_time
                self.last_time += delta
                temp_state = np.copy(self.t_state)
                self.simulate(temp_state, delta*delta_mult, self.c)

                self.theta += self.history[0][-1][0]*delta
                self.t_state = temp_state
                self.t_state[0][0] = self.live_w
                self.t_state[1][0] = self.live_theta + self.live_w*(self.last_time_live - time.time())
                self.post_processing(self.t_state)
                self.history = np.append(self.history, [[self.t_state[0]], [self.t_state[1]]], axis=1)
                self.timestamps = np.append(self.timestamps, [self.last_time - self.start_time], axis=0)

        audiobuf = samples.tobytes()
        if read < hop_s:
            return (audiobuf, pyaudio.paComplete)
        return (audiobuf, pyaudio.paContinue)

    def should_play_beat(self):
        result = False
        num_sub_beats = 1
        if(self.beat_track != None):
            num_sub_beats = len(self.beat_track)

        current_beat_no = (self.theta*num_sub_beats) // (rad_per_cycle/beats_per_cycle)

        if current_beat_no > self.last_beat:
            self.last_beat = current_beat_no
            result = True
        return result

    def live_graph(self, i):
        self.ax.clear()
        self.ax.plot(self.timestamps, self.history[0])

    def simulate(self, t_state, delta, c):
        # method should be implemented in derived classes
        pass

    def plot(self, data):
        plt.plot(data)
        plt.show()

    def setup(self):
        # method should be implemented in derived classes if needed
        pass

    def post_processing(self, t_state):
        # method should be implemented in derived classes if needed
        pass

