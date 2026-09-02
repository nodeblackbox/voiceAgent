"""Print input devices so you can pick --device for the live scripts."""
import sounddevice as sd

apis = sd.query_hostapis()
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        print(f"{i:3d}  {apis[d['hostapi']]['name']:<18} {d['default_samplerate']:>7.0f} Hz  {d['name']}")
print("default input:", sd.default.device[0])
