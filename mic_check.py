import pyaudio
import numpy as np

def check_pyaudio_installation():
    try:
        import pyaudio
        return True
    except ImportError:
        return False

def list_all_audio_devices():
    p = pyaudio.PyAudio()
    device_count = p.get_device_count()
    devices = []
    for i in range(device_count):
        device_info = p.get_device_info_by_index(i)
        devices.append(device_info)
    p.terminate()
    return devices

def test_audio_input(device_index=None):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=device_index)

    print("Recording audio for 5 seconds...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()
    return frames

def check_audio_data(frames):
    audio_data = b''.join(frames)
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    if np.mean(np.abs(audio_array)) > 500:  # Arbitrary threshold
        return True
    return False

if __name__ == "__main__":
    if not check_pyaudio_installation():
        print("PyAudio is not installed. Please install it.")
    else:
        print("PyAudio is installed.")

        devices = list_all_audio_devices()
        print("\nAudio Devices:")
        for idx, device in enumerate(devices):
            print(f"{idx}. {device['name']} (Channels: {device['maxInputChannels']})")

        default_device_index = None
        for idx, device in enumerate(devices):
            if 'defaultSampleRate' in device:
                default_device_index = idx
                break

        if default_device_index is not None:
            print(f"\nTesting audio input for device: {devices[default_device_index]['name']}...")
            frames = test_audio_input(2)
            if check_audio_data(frames):
                print("Audio data captured successfully.")
            else:
                print("No valid audio data captured. Check microphone connection or settings.")
        else:
            print("No default audio input device found.")
