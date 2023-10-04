import numpy as np
import wave

from algo.kuramoto import Kuramoto
from algo.swarmalator import Swarmalator
from algo.janus import Janus
# constants

win_s = 1024
hop_s = win_s // 2
beat_vol = 0.00005

beat_a = wave.open("extras/sounds/drum-a.wav")
beat_b = wave.open("extras/sounds/drum-a.wav")
beat_c = wave.open("extras/sounds/drum-b.wav")
beat_d = wave.open("extras/sounds/drum-a.wav")

beats = [beat_a, beat_b, beat_c, beat_d]
beat_track = []


for i in range(len(beats)):
    beat = beats[i]
    beat_data = []
    beat_buff = np.frombuffer(beat.readframes(hop_s), np.int16)
    shape_size = np.shape(beat_buff)[0]

    while(shape_size > 0):
        if (shape_size < hop_s):
            size_difference = hop_s - shape_size
            beat_buff = np.pad(
                beat_buff, (0, size_difference), "constant")
        beat_data.append(beat_buff)
        beat_buff = np.frombuffer(beat.readframes(hop_s), np.int16) * beat_vol
        shape_size = np.shape(beat_buff)[0]
    beat_track.append(beat_data)

if __name__ == "__main__":
    # Live Audio
    init = [[0.5, 0.5], [0, 0]]
    for attempt in [
        # Swarmalator(init),
        Kuramoto(init),
        # Janus(init),
    ]:
        attempt.set_in_mode(1)  # 0 means input will be audio file, 1 means intput will be microphone
        attempt.set_out_mode(0) # 0 means output will be  wav samples, 1 means sonic pi
        attempt.set_beat_track(beat_track)

        attempt.run_live("extras/sounds/a.wav", 12)
    pass

