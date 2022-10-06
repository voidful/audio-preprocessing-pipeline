import gc
from pathlib import Path

import torch
import torchaudio
from collections import Counter
from speechbrain.pretrained import EncoderClassifier

import denoiser
from denoiser.pretrained import master64
from utility import shuffle_gen, write, collate_fn_padd


class AudioLIDEnhancer:
    def __init__(self, device='cpu', dry_wet=0.01, sampling_rate=16000, chunk_sec=30, max_batch=3,
                 lid_return_n=5,
                 lid_silero_enable=True,
                 lid_voxlingua_enable=True,
                 enable_enhancement=False):
        torchaudio.set_audio_backend("sox_io")  # switch backend
        self.device = device
        self.dry_wet = dry_wet
        self.sampling_rate = sampling_rate
        self.chunk_sec = chunk_sec
        self.chunk_length = sampling_rate * chunk_sec
        self.lid_return_n = lid_return_n
        self.lid_silero_enable = lid_silero_enable
        self.lid_voxlingua_enable = lid_voxlingua_enable
        self.enable_enhancement = enable_enhancement

        # Speech enhancement model
        if enable_enhancement:
            self.enhance_model = master64()
            self.enhance_model = self.enhance_model.to(self.device)
            self.enhance_model.eval()
            self.max_batch = self.get_max_batch()

        # LID model
        if lid_silero_enable:
            self.silero_model, self.silero_lang_dict, self.silero_lang_group_dict, silero_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_lang_detector_95',
                onnx=False)
            self.silero_get_language_and_group, self.silero_read_audio = silero_utils

        # LID model
        if lid_voxlingua_enable:
            self.voxlingua_language_id = EncoderClassifier.from_hparams(source="TalTechNLP/voxlingua107-epaca-tdnn",
                                                                        run_opts={"device": self.device},
                                                                        savedir="tmp")
            self.voxlingua_language_id = self.voxlingua_language_id.to(self.device)
            self.voxlingua_language_id.eval()

    def get_max_batch(self):
        print("calculating max batch size...")
        batch = 1
        with torch.no_grad():
            try:
                while True:
                    self.enhance_model(torch.rand([batch, self.chunk_length]).cuda())
                    batch += 1
                    gc.collect()
                    torch.cuda.empty_cache()
            except:
                pass

        batch = max(batch - 5, 1)
        print("maximum batch size will be", batch)
        return batch

    # performance language identification on input audio,
    # if the language is one of the possible language, perform language enhancement
    # otherwise we just return the original audio
    def __call__(self, filepath='', input_values=[], result_path='', possible_langs=[], max_trial=10,
                 hit_times=5):
        if len(filepath) > 0:
            # loading audio file and generating the enhanced version
            out, sr = torchaudio.load(filepath)
            out = out.mean(0).unsqueeze(0)
        else:
            out = input_values

        # split audio into chunks
        chunks = list(torch.split(out, self.chunk_length, dim=1))
        if chunks[-1].shape[-1] < self.sampling_rate:
            concat_index = -2 if len(chunks) >= 2 else 0
            chunks[concat_index] = torch.cat(chunks[-2:], dim=-1)
            chunks = chunks[:concat_index + 1]

        # total probability
        audio_langs = Counter({})

        # randomly select chunk for language detection
        for s_i in list(shuffle_gen(len(chunks)))[:max_trial]:
            # segment probability
            lid_result = Counter({})
            if self.lid_silero_enable:
                languages, language_groups = self.silero_get_language_and_group(chunks[s_i].squeeze(),
                                                                                self.silero_model,
                                                                                self.silero_lang_dict,
                                                                                self.silero_lang_group_dict,
                                                                                top_n=self.lid_return_n)
                # add the ('2 char lang_code': probability) pair to lid_result
                for l in languages:
                    lang_code = l[0].split(',')[0][:2]
                    if lang_code in lid_result:
                        lid_result[lang_code] += l[-1]
                    else:
                        lid_result[lang_code] = l[-1]

            if self.lid_voxlingua_enable:
                self.voxlingua_language_id = self.voxlingua_language_id.to(self.device)
                prediction = self.voxlingua_language_id.classify_batch(chunks[s_i].squeeze().to(self.device))
                values, indices = torch.topk(prediction[0], self.lid_return_n, dim=-1)
                # add the ('2 char lang_code': probability) pair to lid_result
                for i, l in enumerate(self.voxlingua_language_id.hparams.label_encoder.decode_torch(indices)[0]):
                    lang_code = l[:2]
                    if lang_code in lid_result:
                        lid_result[lang_code] += values[0][i].item()
                    else:
                        lid_result[lang_code] = values[0][i].item()

            # add segment probability to total probability
            if len(possible_langs) == 0:
                audio_langs += lid_result
            else:
                audio_langs += dict(filter(lambda x: x[0] in possible_langs, lid_result.items()))
        
        audio_lang = max(audio_langs, key=audio_langs.get, default='na')

        if self.enable_enhancement and (len(possible_langs) == 0):
            batch_data = []
            cache_batch = []
            for c in chunks:
                if len(cache_batch) >= self.max_batch:
                    batch_data.append(cache_batch)
                    cache_batch = []
                cache_batch.append(c)
            if len(cache_batch) > 0:
                batch_data.append(cache_batch)

            enhance_result = []
            for bd in batch_data:
                batch, lengths, masks = collate_fn_padd([i[0] for i in bd], self.device)
                estimate = (1 - self.dry_wet) * self.enhance_model(batch).squeeze(1) + self.dry_wet * batch
                enhance_result.append(torch.masked_select(estimate, masks).detach().cpu())

            denoise = torch.cat(enhance_result, dim=-1).unsqueeze(0)

            p = Path(filepath)
            write(denoise, str(Path(p.parent, f"{p.stem}_enhanced{p.suffix}")), sr)
            snr = denoiser.utils.cal_snr(out, denoise)
            snr = snr.cpu().detach().numpy()[0]
            return audio_lang, snr
        else:
            return audio_lang, 0
