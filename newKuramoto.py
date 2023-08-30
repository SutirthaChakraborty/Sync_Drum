import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import pyaudio
import aubio

# constants
memory = 8
win_s = 1024
hop_s = win_s // 2
samplerate = 44100
click = 0.7 * np.sin(2. * np.pi * np.arange(hop_s) /
                     hop_s * samplerate / 3000.)
graph_update_interval = 500
stabilization_time = 10*1000
delta_mult = 0.04
rad_per_beat = np.math.pi/8
beats_per_cycle = 0.5
# main driver


def main():
    # First Attempt
    # init_1 = np.array([[0.57, 3.36, 7.23, 3.28, 10.54],
    #                    [-2.25, -0.49, 2.61, 1.84, 2.89]])
    # attempt_1 = FirstAttempt(initial_conditions=init_1)
    # attempt_1.run(10000, 0.01, 2)

    # # Second Attempt
    # init_2 = np.array([[5.62, 2.93, 6.83, 3.20, 9.23],
    #                    [-0.05, -0.33, -0.08, -2.10, -0.87]])
    # attempt_2 = SecondAttempt(initial_conditions=init_2)
    # attempt_2.run(10000, 0.001, 2)

    # # Third Attempt
    # init_3 = np.array([[1.87, 2.99, 6.90, 3.33, 9.40],
    #                    [0.48, 2.50, -0.23, -0.64, -2.48]])
    # attempt_3 = ThirdAttempt(initial_conditions=init_3)
    # attempt_3.run(1000, 0.01, 2)

    # # Fourth Attempt
    # init_4 = np.array([[1.87, 2.99, 6.90, 3.33, 9.40],
    #                    [0.48, 2.50, -0.23, -0.64, -2.48]])
    # attempt_4 = FourthAttempt(initial_conditions=init_4)
    # attempt_4.run(1000, 0.01, 2)

    # # Fifth Attempt
    # init_5 = np.array([[1.87, 2.99, 6.90, 3.33, 9.40],
    #                    [0.48, 2.50, -0.23, -0.64, -2.48]])
    # attempt_5 = FifthAttempt(initial_conditions=init_5)
    # attempt_5.run(1000, 0.01, 2)

    # # Sixth Attempt
    # init_6 = np.array([[1.87, 2.99, 6.90, 3.33, 9.40],
    #                    [0.48, 2.50, -0.23, -0.64, -2.48]])
    # attempt_6 = SixthAttempt(initial_conditions=init_6)
    # attempt_6.run(1000, 0.01, 2)

    # Live Audio
    init = [[0.5, 0.5], [0, 0]]
    for attempt in [
                    SongAttempt(init),
                    FirstAttempt(init),
                    SecondAttempt(init),
                    ThirdAttempt(init),
                    FourthAttempt(init),
                    FifthAttempt(init),
                    SixthAttempt(init), ]:
        attempt.run_live("pattern1/100.wav", 12)

# base class for simulations


class Simulation:
    def __init__(self, initial_conditions=np.zeros([2, 1])):
        self.state = np.copy(np.array(initial_conditions, dtype=np.float))
        self.setup()

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
            [[np.copy(self.state[0])], [np.copy(self.state[1])]], dtype=np.float)
        self.timestamps = [0]
        self.live_w = 0
        self.live_theta = 0
        self.theta = 0
        self.last_beat = 0

        self.a_source = aubio.source(file_path, samplerate, hop_s)
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
        samples, read = self.a_source()
        is_beat = self.a_tempo(samples)

        if time.time() < self.start_time + stabilization_time:
            if self.should_play_beat():
                samples += click

            if is_beat:
                if self.last_time_live != None:
                    delta = time.time() - self.last_time_live
                    self.live_theta += rad_per_beat
                    self.live_w = (rad_per_beat)/delta
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
                self.t_state[1][0] = self.live_theta + \
                    self.live_w*(self.last_time_live - time.time())
                self.post_processing(self.t_state)
                self.history = np.append(
                    self.history, [[self.t_state[0]], [self.t_state[1]]], axis=1)
                self.timestamps = np.append(
                    self.timestamps, [self.last_time - self.start_time], axis=0)

        audiobuf = samples.tobytes()
        if read < hop_s:
            return (audiobuf, pyaudio.paComplete)
        return (audiobuf, pyaudio.paContinue)


    def should_play_beat(self):
        result = False
        current_beat_no = (self.theta) // (rad_per_beat/beats_per_cycle)
        if current_beat_no > self.last_beat:
            self.last_beat = current_beat_no
            result = True
        return result

    def live_graph(self, i):
        self.ax.clear()
        self.ax.plot(self.timestamps ,self.history[0])

    def simulate(self, t_state, delta, c):
        # method should be implemented in derived classes
        pass

    def plot(self, data):
        plt.plot(data)
        plt.show()

    def setup(self):
        pass

    def post_processing(self, t_state):
        pass


# derived classes
# class for first attempt

class SongAttempt(Simulation):
    def live_graph(self, i):
        self.ax.clear()
        data = np.transpose(self.history[0])[0] * 60 / rad_per_beat
        self.ax.plot(self.timestamps ,data)


class FirstAttempt(Simulation):
    def simulate(self, t_state, delta, c):
        num_osc = len(t_state[0])
        del_theta = np.zeros([1, num_osc])
        del_w = np.zeros([1, num_osc])
        group_w = np.average(t_state[0])
        group_phase = np.average(t_state[1])

        r = 0
        for i in range(num_osc):
            r += e_val(t_state[1][i] - group_phase)
        r /= num_osc

        for i in range(num_osc):
            diff_phase = 0
            std_dev_w = 0
            for j in range(num_osc):
                diff_phase += np.math.sin(t_state[0][j] - t_state[0][i])
                std_dev_w += (t_state[0][i] - t_state[0][j])**2
            std_dev_w /= (num_osc-1)
            std_dev_w = np.math.sqrt(std_dev_w)

            del_theta[0][i] = t_state[0][i] + c*diff_phase
            if t_state[0][i] - group_w > 0:
                del_w[0][i] = -(1-r)*(t_state[0][i])
            else:
                del_w[0][i] = (1-r)*(t_state[0][i])

        del_all = np.append(del_w, del_theta, axis=0)
        t_state += del_all*delta


# class for second attempt
class SecondAttempt(Simulation):
    def simulate(self, t_state, delta, c):
        num_osc = len(t_state[0])
        del_theta = np.zeros([1, num_osc])
        del_w = np.zeros([1, num_osc])
        group_w = np.average(t_state[0])
        group_phase = np.average(t_state[1])

        for i in range(num_osc):

            r = 0
            for j in range(num_osc):
                r += e_val(t_state[1][j])
            r += e_val(t_state[1][i])
            r /= (num_osc + 1)

            diff_phase = 0
            std_dev_w = 0
            for j in range(num_osc):
                diff_phase += np.math.sin(t_state[0][j] - t_state[0][i])
                std_dev_w += (t_state[0][i] - t_state[0][j])**2
            std_dev_w /= (num_osc-1)
            std_dev_w = np.math.sqrt(std_dev_w)

            del_theta[0][i] = t_state[0][i] + c*diff_phase
            if t_state[0][i] - group_w > 0:
                del_w[0][i] = -(1-r)*(t_state[0][i])
            else:
                del_w[0][i] = (1-r)*(t_state[0][i])

        del_all = np.append(del_w, del_theta, axis=0)
        t_state += del_all*delta


# class for third attempt
class ThirdAttempt(Simulation):

    def setup(self):
        self.state_record = np.zeros([memory, len(self.state[0])])
        self.state_record += np.copy(self.state[0])

    def simulate(self, t_state, delta, c):
        num_osc = len(t_state[0])
        del_theta = np.zeros([1, num_osc])
        del_w = np.zeros([1, num_osc])
        group_w = np.average(t_state[0])
        group_phase = np.average(t_state[1])

        std_dev_w = np.std(self.state_record, axis=0)
        std_dev_w = normalize2(std_dev_w)
        a = np.zeros([num_osc, num_osc])
        for i in range(num_osc):
            for j in range(num_osc):
                if std_dev_w[j] == 0:
                    a[i][j] = np.inf
                else:
                    a[i][j] = std_dev_w[i]/std_dev_w[j]

        a = normalize3(a)

        for i in range(num_osc):
            r = 0
            for j in range(num_osc):
                r += e_val(t_state[1][j])
            r += e_val(t_state[1][i])
            r /= (num_osc + 1)

            diff_phase = 0
            std_dev_w = 0
            for j in range(num_osc):
                diff_phase += a[i][j] * \
                    np.math.sin(t_state[0][j] - t_state[0][i])
                std_dev_w += (t_state[0][i] - t_state[0][j])**2
            std_dev_w /= (num_osc-1)
            std_dev_w = np.math.sqrt(std_dev_w)

            del_theta[0][i] = t_state[0][i] + c*diff_phase
            if t_state[0][i] - group_w > 0:
                del_w[0][i] = -(1-r)*(t_state[0][i])
            else:
                del_w[0][i] = (1-r)*(t_state[0][i])

        del_all = np.append(del_w, del_theta, axis=0)
        t_state += del_all*delta

    def post_processing(self, t_state):
        self.state_record = np.append(
            self.state_record[1:], [t_state[0]], axis=0)


# class for fourth attempt
class FourthAttempt(Simulation):

    def setup(self):
        self.state_record = np.zeros([memory, len(self.state[0])])
        self.state_record += np.copy(self.state[0])

    def simulate(self, t_state, delta, c):
        num_osc = len(t_state[0])
        del_theta = np.zeros([1, num_osc])
        del_w = np.zeros([1, num_osc])
        group_w = np.average(t_state[0])
        group_phase = np.average(t_state[1])

        std_dev_w = np.std(self.state_record, axis=0)
        std_dev_w = normalize2(std_dev_w)
        a = np.zeros([num_osc, num_osc])
        for i in range(num_osc):
            for j in range(num_osc):
                if std_dev_w[i] == 0:
                    a[i][j] = np.inf
                else:
                    a[i][j] = std_dev_w[j]/std_dev_w[i]

        a = normalize3(a)

        for i in range(num_osc):
            r = 0
            for j in range(num_osc):
                r += e_val(t_state[1][j])
            r += e_val(t_state[1][i])
            r /= (num_osc + 1)

            diff_phase = 0
            std_dev_w = 0
            for j in range(num_osc):
                diff_phase += a[i][j] * \
                    np.math.sin(t_state[0][j] - t_state[0][i])
                std_dev_w += (t_state[0][i] - t_state[0][j])**2
            std_dev_w /= (num_osc-1)
            std_dev_w = np.math.sqrt(std_dev_w)

            del_theta[0][i] = t_state[0][i] + c*diff_phase
            if t_state[0][i] - group_w > 0:
                del_w[0][i] = -(1-r)*(t_state[0][i])
            else:
                del_w[0][i] = (1-r)*(t_state[0][i])

        del_all = np.append(del_w, del_theta, axis=0)
        t_state += del_all*delta

    def post_processing(self, t_state):
        self.state_record = np.append(
            self.state_record[1:], [t_state[0]], axis=0)


# class for fifth attempt
class FifthAttempt(Simulation):

    def setup(self):
        self.state_record = np.zeros([memory, len(self.state[0])])
        self.state_record += np.copy(self.state[0])

    def simulate(self, t_state, delta, c):
        num_osc = len(t_state[0])
        del_theta = np.zeros([1, num_osc])
        del_w = np.zeros([1, num_osc])
        group_w = np.average(t_state[0])
        group_phase = np.average(t_state[1])

        std_dev_w = np.std(self.state_record, axis=0)
        std_dev_w = normalize2(std_dev_w)
        avg_std_dev_w = np.average(std_dev_w)
        a = np.zeros([num_osc, num_osc])
        for i in range(num_osc):
            if std_dev_w[i] > avg_std_dev_w:
                for j in range(num_osc):
                    if std_dev_w[i] == 0:
                        a[i][j] = np.inf
                    else:
                        a[i][j] = std_dev_w[j]/std_dev_w[i]
            else:
                for j in range(num_osc):
                    if std_dev_w[j] == 0:
                        a[i][j] = np.inf
                    else:
                        a[i][j] = std_dev_w[i]/std_dev_w[j]

        a = normalize3(a)

        for i in range(num_osc):
            r = 0
            for j in range(num_osc):
                r += e_val(t_state[1][j])
            r += e_val(t_state[1][i])
            r /= (num_osc + 1)

            diff_phase = 0
            std_dev_w = 0
            for j in range(num_osc):
                diff_phase += a[i][j] * \
                    np.math.sin(t_state[0][j] - t_state[0][i])
                std_dev_w += (t_state[0][i] - t_state[0][j])**2
            std_dev_w /= (num_osc-1)
            std_dev_w = np.math.sqrt(std_dev_w)

            del_theta[0][i] = t_state[0][i] + c*diff_phase
            if t_state[0][i] - group_w > 0:
                del_w[0][i] = -(1-r)*(t_state[0][i])
            else:
                del_w[0][i] = (1-r)*(t_state[0][i])

        del_all = np.append(del_w, del_theta, axis=0)
        t_state += del_all*delta

    def post_processing(self, t_state):
        self.state_record = np.append(
            self.state_record[1:], [t_state[0]], axis=0)


# class for sixth attempt
class SixthAttempt(Simulation):

    def setup(self):
        self.state_record = np.zeros([memory, len(self.state[0])])
        self.state_record += np.copy(self.state[0])

    def simulate(self, t_state, delta, c):
        num_osc = len(t_state[0])
        del_theta = np.zeros([1, num_osc])
        del_w = np.zeros([1, num_osc])
        group_w = np.average(t_state[0])
        group_phase = np.average(t_state[1])

        std_dev_w = np.std(self.state_record, axis=0)
        std_dev_w = normalize2(std_dev_w)
        avg_std_dev_w = np.average(std_dev_w)
        a = np.zeros([num_osc, num_osc])
        for i in range(num_osc):
            if std_dev_w[i] < avg_std_dev_w:
                for j in range(num_osc):
                    if std_dev_w[i] == 0:
                        a[i][j] = np.inf
                    else:
                        a[i][j] = std_dev_w[j]/std_dev_w[i]
            else:
                for j in range(num_osc):
                    if std_dev_w[j] == 0:
                        a[i][j] = np.inf
                    else:
                        a[i][j] = std_dev_w[i]/std_dev_w[j]

        a = normalize3(a)

        for i in range(num_osc):
            r = 0
            for j in range(num_osc):
                r += e_val(t_state[1][j])
            r += e_val(t_state[1][i])
            r /= (num_osc + 1)

            diff_phase = 0
            std_dev_w = 0
            for j in range(num_osc):
                diff_phase += a[i][j] * \
                    np.math.sin(t_state[0][j] - t_state[0][i])
                std_dev_w += (t_state[0][i] - t_state[0][j])**2
            std_dev_w /= (num_osc-1)
            std_dev_w = np.math.sqrt(std_dev_w)

            del_theta[0][i] = t_state[0][i] + c*diff_phase
            if t_state[0][i] - group_w > 0:
                del_w[0][i] = -(1-r)*(t_state[0][i])
            else:
                del_w[0][i] = (1-r)*(t_state[0][i])

        del_all = np.append(del_w, del_theta, axis=0)
        t_state += del_all*delta

    def post_processing(self, t_state):
        self.state_record = np.append(
            self.state_record[1:], [t_state[0]], axis=0)


def e_val(theta):
    return np.abs(np.math.cos(theta))


def normalize2(v):
    result = np.copy(v)
    m = 0
    for i in v:
        abs_i = np.abs(i)
        if abs_i > m:
            m = abs_i

    if m == 0:
        result = np.ones([len(v)])
    else:
        result /= m
    return result


def normalize3(v):
    result = np.copy(v)
    m = 0
    for i in v:
        for j in i:
            abs_j = np.abs(j)
            if abs_j > m:
                m = abs_j

    if m == 0:
        result = np.ones([len(v), len(v[0])])
    elif m == np.inf:
        for i in range(len(result)):
            for j in range(len(result[i])):
                if result[i][j] == np.inf:
                    result[i][j] = 1
                else:
                    result[i][j] = 0
    else:
        result /= m
    return result


if __name__ == "__main__":
    main()
