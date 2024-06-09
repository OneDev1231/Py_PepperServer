import torch

SAMPLING_RATE = 16000

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=False)

(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

vad_iterator = VADIterator(model)


wav = read_audio('data/recording.wav', sampling_rate=SAMPLING_RATE)
print(len(wav))
window_size_samples = 16000 # number of samples in a single audio chunk
for i in range(0, len(wav), window_size_samples):
    chunk = wav[i: i+ window_size_samples]
    if len(chunk) < window_size_samples:
      break
    speech_dict = vad_iterator(chunk, return_seconds=True)
    print(speech_dict)
# vad_iterator.reset_states()