
import numpy as np
import sounddevice as sd
import mido
import time
from math import pi
from threading import Lock

SR = 44100
BLOCKSIZE = 512

voices = {}
voices_lock = Lock()
ATTACK_TIME = 0.01  # 10ms fade-in
RELEASE_TIME = 0.05  # 50ms fade-out

def midi_note_to_freq(note):
    return 440.0 * 2 ** ((note - 69) / 12.0)

def midi_callback(msg):
    global voices
    if msg.type == 'note_on' and msg.velocity > 0:
        with voices_lock:
            voices[msg.note] = {
                'freq': midi_note_to_freq(msg.note),
                'phase': 0.0,
                'vel': msg.velocity / 127.0,
                'attack': 0,    # start attack counter
                'release': None # None means note is held
            }
    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
        with voices_lock:
            if msg.note in voices:
                voices[msg.note]['release'] = 0  # start release

def audio_callback(outdata, frames, time_info, status):
    buffer = np.zeros(frames, dtype=np.float32)
    with voices_lock:
        for note, v in list(voices.items()):
            t = np.arange(frames)
            inc = 2 * pi * v['freq'] / SR
            phases = v['phase'] + t * inc

            amp = v['vel']

            # Handle attack fade-in
            if v['attack'] is not None:
                attack_samples = int(ATTACK_TIME * SR)
                start = v['attack']
                end = start + frames
                attack_curve = np.clip(np.arange(start, end) / attack_samples, 0, 1)
                amp *= attack_curve
                v['attack'] += frames
                if v['attack'] >= attack_samples:
                    v['attack'] = None  # attack complete

            # Handle release fade-out
            if v['release'] is not None:
                release_samples = int(RELEASE_TIME * SR)
                start = v['release']
                end = start + frames
                release_curve = 1 - np.clip(np.arange(start, end) / release_samples, 0, 1)
                amp *= release_curve
                v['release'] += frames
                if v['release'] >= release_samples:
                    del voices[note]  # remove note after release
                    continue

            buffer += np.sin(phases) * amp * 0.2
            v['phase'] = (phases[-1] + inc) % (2 * pi)

    outdata[:] = np.column_stack([buffer, buffer])

def main():
    ports = mido.get_input_names()
    if not ports:
        print("No MIDI input detected. Connect a keyboard or virtual MIDI device.")
        return
    print(f"Using MIDI input: {ports[0]}")
    with mido.open_input(ports[0], callback=midi_callback):
        with sd.OutputStream(channels=2, callback=audio_callback, samplerate=SR, blocksize=BLOCKSIZE):
            print("Synth running. Press Ctrl+C to quit.")
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nExiting.")

if __name__ == "__main__":
    main()
