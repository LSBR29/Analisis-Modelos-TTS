text = """¡Hola!, esta es una voz artificial con el modelo de texto a voz entrenado para el t c u 748."""
# output wav path
OUTPUT_WAV_PATH = "/kaggle/working/out.wav"

print("Inference...")
out = model.inference(
    text,
    "es",
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.95,
    top_k=50,
    top_p=0.8,
    repetition_penalty=2.0,
)
torchaudio.save(OUTPUT_WAV_PATH, torch.tensor(out["wav"]).unsqueeze(0), 24000)