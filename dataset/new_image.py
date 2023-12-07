import os
import torch
import torchaudio
import numpy as np
from PIL import Image
import wave
import contextlib
import torch
import torchaudio


def audio_to_image(full_path, file_path, audio_folder, metadata_folder, times, power_for_image=0.25, sample_rate=44100, step_size_ms=10, window_duration_ms=100, padded_duration_ms=400, num_frequencies=512, min_frequency=0, max_frequency=10000, mel_scale_type="htk", num_griffin_lim_iters=32):
    waveform, sample_rate = torchaudio.load(full_path)
    num_samples_for_6_sec = sample_rate * 6  
    times = int(times)

    #Cut audio into melspectrogram every 6 seconds, up to 20 melspectrograms.
    for segment in range(times):
        start_sample = segment * num_samples_for_6_sec
        end_sample = start_sample + num_samples_for_6_sec
        segment_waveform = waveform[:, start_sample:end_sample]

        # FFT parameters
        n_fft = int(padded_duration_ms / 1000.0 * sample_rate)
        hop_length = int(step_size_ms / 1000.0 * sample_rate)

        # Mel scale transformation
        mel_scaler = torchaudio.transforms.MelScale(
            n_mels=num_frequencies,
            sample_rate=sample_rate,
            f_min=min_frequency,
            f_max=max_frequency,
            n_stft=n_fft // 2 + 1,
            norm=None,
            mel_scale=mel_scale_type,
        )

        # Spectrogram transformation
        spectrogram_func = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=None,
            pad=0,
            window_fn=torch.hann_window,
            power=None,
            normalized=False,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )

        # Generate Spectrogram
        spectrogram_complex = spectrogram_func(segment_waveform)
        amplitudes = torch.abs(spectrogram_complex)
        melspec = mel_scaler(amplitudes)

        # Cut the spectrogram
        melspec = melspec[:, :, :512]  # Cut to 512 units

        def image_from_spectrogram(spectrogram, power):
            # Convert the input spectrogram to a NumPy array
            spectrogram = np.array(spectrogram)
            # Get the maximum value in the spectrogram for subsequent normalization
            max_value = np.max(spectrogram)
            data = spectrogram / max_value
            data = np.power(data, power)
            # Scale data to a range of 0-255 for use in creating images
            data = data * 255
            data = data.astype(np.uint8)
            data = np.array([np.zeros_like(data[0]), data[0], data[1]]).transpose(1, 2, 0)
            # Create RGB images from processed data using the PIL library
            image = Image.fromarray(data, mode="RGB")
            # Flip the image up and down, as spectrograms are usually represented with frequencies from low to high
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            return image, max_value

        # Convert to image
        im, max_val = image_from_spectrogram(melspec, power_for_image)

        # Save image
        image_name = os.path.splitext(file_path)[0] + f"_{segment}.png"  
        image_path = os.path.join(metadata_folder, image_name)
        im.save(image_path)




# get audio's time
def get_wav_duration(file_path):
    with contextlib.closing(wave.open(file_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration

# Resolving Library Conflicts
os.environ['KMP_DUPLICATE_LIB_OK']='True'
audio_folder='data\MajorAndMinor'
metadata_folder='metadata'


for filename in os.listdir(audio_folder):
    full_path = os.path.join(audio_folder, filename)
    duration = get_wav_duration(full_path)
    times=duration//6
    # If the audio is longer than 2 minutes
    if times>=20:        
        audio_to_image(full_path,filename, audio_folder, metadata_folder,20)
    else:
        audio_to_image(full_path,filename, audio_folder, metadata_folder,times)
