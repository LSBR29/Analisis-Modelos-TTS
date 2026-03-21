import os
from pathlib import Path
import librosa
import soundfile as sf

def main(input_dir, output_dir, encoding_txt, extension_audio, target_sr=22050):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.is_dir():
        raise ValueError(f"El directorio {input_dir} no existe.")

    # Crear directorio de salida y subcarpeta wavs (si no existen)
    output_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(exist_ok=True)

    wav_files = list(input_dir.glob(f"*.{extension_audio}"))
    if not wav_files:
        print(f"No se encontraron archivos con extensión .wav en {input_dir}")
        return

    with open(output_dir / "metadata.csv", "w", encoding="utf-8") as out_f:
        for wav_path in wav_files:
            txt_path = wav_path.with_suffix(".txt")
            if not txt_path.exists():
                print(f"No se encuentra transcripción para {wav_path.name}. Se omite.")
                continue

            # Leer transcripción
            text = txt_path.read_text(encoding=encoding_txt).strip()

            # Ruta de salida para el audio (mismo nombre)
            target_wav_path = wavs_dir / wav_path.name

            try:
                # Cargar audio (mono, frecuencia original)
                audio, sr = librosa.load(wav_path, sr=None, mono=True)

                # Resamplear si la frecuencia no es la deseada
                if sr != target_sr:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

                # Guardar como WAV (16-bit PCM)
                sf.write(target_wav_path, audio, target_sr, subtype='PCM_16')

                # Escribir metadatos (formato: nombre_sin_ext|texto|texto)
                out_f.write(f"{wav_path.stem}|{text}|{text}\n")

            except Exception as e:
                print(f"Error procesando {wav_path.name}: {e}")
                continue

if __name__ == "__main__":
    input_dir = "RUTA_DATASET"
    output_dir = "/kaggle/working/dataset"
    extension_audio = "wav"
    encoding_txt = "windows-1252"

    main(input_dir, output_dir, encoding_txt, extension_audio)
    print("Dataset organizado correctamente")