import torch
import torchaudio


class UtilsSignal:
    @staticmethod
    def apply_bandpass_filter_torchaudio(data, lowcut, highcut, fs):
        filtered_data = torch.zeros_like(data)

        # Calculate the central frequency and Q factor for the bandpass filter
        central_freq = (lowcut + highcut) / 2
        bandwidth = highcut - lowcut
        Q = central_freq / bandwidth  # Quality factor

        # Apply the bandpass filter on each EEG channel
        for i in range(data.size(0)):
            # Using torchaudio's bandpass biquad filter
            filtered_data[i] = torchaudio.functional.bandpass_biquad(
                waveform=data[i].unsqueeze(0),  # Add batch dimension
                sample_rate=fs,
                central_freq=central_freq,  # Central frequency of the bandpass filter
                Q=Q  # Quality factor determining the bandwidth
            ).squeeze(0)  # Remove batch dimension after filtering

        return filtered_data

    @staticmethod
    def apply_bandpass_batch(x, freqranges, fs):
        n = int(x.shape[0] / 19)
        x_out = torch.zeros_like(x)
        for i in range(n):
            st, en = i * 19, (i + 1) * 19
            x_out[st:en] = UtilsSignal.apply_bandpass_filter_torchaudio(x[st:en], freqranges[i][0], freqranges[i][1], fs)

        return x_out