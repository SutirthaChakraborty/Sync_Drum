import numpy as np
from algo.simulation import Simulation

memory = 8

# class for fifth attempt
class Janus(Simulation):
    def setup(self):
        self.state_record = np.zeros([memory, len(self.state[0])])
        self.state_record += np.copy(self.state[0])

    def simulate(self, t_state, delta, c):
        num_osc = len(t_state[0])
        del_theta = np.zeros([1, num_osc])

        beta = 10
        sigma = 10
        wn = 1
        vn = 1
        delta = 0.01
        for i in range(0, num_osc):
            if i - 1 == 0:
                j = num_osc - 1
            elif i + 1 == num_osc:
                j = 0
            else:
                j = i
            t_state[0][i] += (vn + beta * np.sin(t_state[1][i] - t_state[0][i]) + sigma * np.sin(
                t_state[1][j] - t_state[0][i])) * delta
            t_state[1][i] += (wn + beta * np.sin(t_state[0][i] - t_state[1][i]) + sigma * np.sin(
                t_state[0][j] - t_state[1][i])) * delta

    def post_processing(self, t_state):
        self.state_record = np.append(
            self.state_record[1:], [t_state[0]], axis=0)


