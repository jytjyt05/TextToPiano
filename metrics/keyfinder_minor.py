import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
import librosa.display
import collections

hashmap=collections.defaultdict(int)
# only minor music
ListMinorOnly=[]
# maybe minor+maybe major
ListMajororMinor=[]
# class that uses the librosa library to analyze the key that an mp3 is in
# arguments:
#     waveform: an mp3 file loaded by librosa, ideally separated out from any percussive sources
#     sr: sampling rate of the mp3, which can be obtained when the file is read with librosa
#     tstart and tend: the range in seconds of the file to be analyzed; default to the beginning and end of file if not specified
class Tonal_Fragment(object):
    def __init__(self, waveform, sr, tstart=None, tend=None):
        self.waveform = waveform
        self.sr = sr
        self.tstart = tstart
        self.tend = tend
        
        if self.tstart is not None:
            self.tstart = librosa.time_to_samples(self.tstart, sr=self.sr)
        if self.tend is not None:
            self.tend = librosa.time_to_samples(self.tend, sr=self.sr)
        self.y_segment = self.waveform[self.tstart:self.tend]
        self.chromograph = librosa.feature.chroma_cqt(y=self.y_segment, sr=self.sr, bins_per_octave=24)
        
        # chroma_vals is the amount of each pitch class present in this time interval
        self.chroma_vals = []
        for i in range(12):
            self.chroma_vals.append(np.sum(self.chromograph[i]))
        pitches = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        # dictionary relating pitch names to the associated intensity in the song
        self.keyfreqs = {pitches[i]: self.chroma_vals[i] for i in range(12)} 
        
        keys = [pitches[i] + ' major' for i in range(12)] + [pitches[i] + ' minor' for i in range(12)]

        # use of the Krumhansl-Schmuckler key-finding algorithm, which compares the chroma
        # data above to typical profiles of major and minor keys:
        maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

        # finds correlations between the amount of each pitch class in the time interval and the above profiles,
        # starting on each of the 12 pitches. then creates dict of the musical keys (major/minor) to the correlation
        self.min_key_corrs = []
        self.maj_key_corrs = []
        for i in range(12):
            key_test = [self.keyfreqs.get(pitches[(i + m)%12]) for m in range(12)]
            # correlation coefficients (strengths of correlation for each key)
            self.maj_key_corrs.append(round(np.corrcoef(maj_profile, key_test)[1,0], 3))
            self.min_key_corrs.append(round(np.corrcoef(min_profile, key_test)[1,0], 3))

        # names of all major and minor keys
        self.key_dict = {**{keys[i]: self.maj_key_corrs[i] for i in range(12)}, 
                         **{keys[i+12]: self.min_key_corrs[i] for i in range(12)}}
        
        # this attribute represents the key determined by the algorithm
        self.key = max(self.key_dict, key=self.key_dict.get)
        self.bestcorr = max(self.key_dict.values())
        
        # this attribute represents the second-best key determined by the algorithm,
        # if the correlation is close to that of the actual key determined
        self.altkey = None
        self.altbestcorr = None

        for key, corr in self.key_dict.items():
            if corr > self.bestcorr*0.9 and corr != self.bestcorr:
                self.altkey = key
                self.altbestcorr = corr
                
    # prints the relative prominence of each pitch class            
    def print_chroma(self):
        self.chroma_max = max(self.chroma_vals)
        for key, chrom in self.keyfreqs.items():
            print(key, '\t', f'{chrom/self.chroma_max:5.3f}')
                
    # prints the correlation coefficients associated with each major/minor key
    def corr_table(self):
        for key, corr in self.key_dict.items():
            print(key, '\t', f'{corr:6.3f}')
    
    # printout of the key determined by the algorithm; if another key is close, that key is mentioned
    def print_key(self,index):
        print("likely key: ", max(self.key_dict, key=self.key_dict.get), ", correlation: ", self.bestcorr, sep='')
        if self.altkey is not None:
            print("also possible: ", self.altkey, ", correlation: ", self.altbestcorr, sep='')
            # if ('minor' in max(self.key_dict, key=self.key_dict.get) and 'major' in self.altkey) or \
            #         ('major' in max(self.key_dict, key=self.key_dict.get) and 'minor' in self.altkey):
            #     ListMajororMinor.append('Minor%d.wav'%index)
            if ('minor' in max(self.key_dict, key=self.key_dict.get) and 'major' in self.altkey):
                ListMajororMinor.append('Minor%d.wav'%index)
        if ('minor' in max(self.key_dict, key=self.key_dict.get) and self.altkey is None) or \
                ('minor' in max(self.key_dict, key=self.key_dict.get) and 'minor' in self.altkey):
            ListMinorOnly.append('Minor%d.wav'%index)
    
    # prints a chromagram of the file, showing the intensity of each pitch class over time
    def chromagram(self, title=None):
        C = librosa.feature.chroma_cqt(y=self.waveform, sr=sr, bins_per_octave=24)
        plt.figure(figsize=(12,4))
        librosa.display.specshow(C, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
        if title is None:
            plt.title('Chromagram')
        else:
            plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        plt.show()
for i in range(1,6):
    # this audio takes a long time to load because it has a very high sampling rate; be patient.
    audio_path = './Musicgen_minor/m%d.wav'%i
    # the load function generates a tuple consisting of an audio object y and its sampling rate sr
    y, sr = librosa.load(audio_path)
    # this function filters out the harmonic part of the sound file from the percussive part, allowing for
    # more accurate harmonic analysis
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    ipd.Audio(audio_path)
    # unebarque_fsharp_min = Tonal_Fragment(y_harmonic, sr, tend=5)
    # unebarque_fsharp_min.print_chroma()

    key_table = Tonal_Fragment(y_harmonic, sr, tend=15)
    key_table.print_key(i)

    # unebarque_fsharp_min.print_key()
    # unebarque_fsharp_min.corr_table()
    # unebarque_fsharp_min.chromagram("Une Barque sur l\'Ocean")
print(hashmap)
print(ListMinorOnly)
print(ListMajororMinor)
print(len(ListMinorOnly)+len(ListMajororMinor))
