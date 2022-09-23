# audio-preprocessing-pipeline

## Preparation

```shell
apt-get install libsox-fmt-all libsox-dev sox
add-apt-repository -y ppa:savoury1/ffmpeg4
apt-get -qq install -y ffmpeg
```

## Usage

### Convert audio format to ogg and sampling to 16k

`python convert_format_sampling.py -s /audio_folder/ -w 30`

### Language Identification (LID) and speech enhancement

```python
from lid_enhancement import AudioLIDEnhancer

ase = AudioLIDEnhancer(enable_enhancement=False)
print(ase('test.ogg'))
```

## References

Denoiser copied
from [fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/speech_synthesis/preprocessing/denoiser)
