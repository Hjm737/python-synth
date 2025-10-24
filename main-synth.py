# midi_sine_3_11.py
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

def midi_note_to_freq(note):
    return 440.0 * 2 ** ((note - 69) / 12.0)

def midi_callback(msg):
    global voices
    if msg.type == 'note_on' and msg.velocity > 0:
        with voices_lock:
            voices[msg.note] = {
                'freq': midi_note_to_freq(msg.note),
                'phase': 0.0,
                'vel': msg.velocity / 127.0
            }
    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
        with voices_lock:
            if msg.note in voices:
                del voices[msg.note]

def audio_callback(outdata, frames, time_info, status):
    buffer = np.zeros(frames, dtype=np.float32)
    with voices_lock:
        for v in voices.values():
            t = np.arange(frames)
            inc = 2 * pi * v['freq'] / SR
            phases = v['phase'] + t * inc
            buffer += np.sin(phases) * v['vel'] * 0.2
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
