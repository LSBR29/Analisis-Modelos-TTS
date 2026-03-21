import os
import sys
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

sys.path.append("RUTA_REPOSITORIO_ORIGINAL_MODELO")

# Add here the xtts_config path
CONFIG_PATH = "/kaggle/working/run/training/.../config.json"
# Add here the vocab file that you have used to train the model
TOKENIZER_PATH = "/kaggle/working/run/training/.../vocab.json"
# Add here the checkpoint that you want to do inference with
XTTS_CHECKPOINT = "/kaggle/working/run/training/.../best_model_....pth"
# Add here the speaker reference
SPEAKER_REFERENCE = ["/kaggle/working/dataset/wavs/REFERENCIA.wav"]
# output wav path
OUTPUT_WAV_PATH = "/kaggle/working/out.wav"

print("Loading model...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=XTTS_CHECKPOINT, vocab_path=TOKENIZER_PATH, use_deepspeed=False)
model.cuda() #ACTIVAR PARA GPU

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=SPEAKER_REFERENCE)