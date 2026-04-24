#!/usr/bin/env python
"""
Standalone music continuation test — no service required.

Usage:
    conda run -n cf python scripts/test_musicgen.py \
        --input "/Library/Audio/Music/KAESUL/Schedule I - Original Soundtrack (2025)/KAESUL - Schedule I - Original Soundtrack - 17 - the life i lead (reveal trailer).mp3"

Options:
    --input PATH          Audio file to continue (any ffmpeg-readable format)
    --output PATH         Output WAV path (default: /tmp/continuation_output.wav)
    --model MODEL         MusicGen variant (default: facebook/musicgen-melody)
    --duration SECS       Seconds of new audio to generate (default: 30)
    --prompt-duration SECS  Seconds from end of song to condition on (default: 10)
    --description TEXT    Optional style hint, e.g. "dark ambient electronic"
    --device DEVICE       cuda or cpu (default: cuda)
    --join                Concatenate original prompt segment + continuation in output

The generated file is saved to --output. Open it in any audio player to listen.
Model weights download to /Library/Assets/LLM/musicgen/ on first run (~8 GB for melody).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Redirect HF cache before any audiocraft import
os.environ.setdefault("HF_HOME", "/Library/Assets/LLM/musicgen")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
)
log = logging.getLogger("test_musicgen")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="cf-musicgen standalone test")
    p.add_argument("--input", required=True, help="Input audio file path")
    p.add_argument("--output", default="/tmp/continuation_output.wav")
    p.add_argument("--model", default="facebook/musicgen-melody")
    p.add_argument("--duration", type=float, default=30.0,
                   help="Seconds of new audio to generate")
    p.add_argument("--prompt-duration", type=float, default=10.0,
                   help="Seconds from end of song used as prompt")
    p.add_argument("--description", default=None,
                   help="Optional text description to guide the style")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--join", action="store_true",
                   help="Prepend the prompt segment to the output file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input):
        log.error("Input file not found: %s", args.input)
        sys.exit(1)

    log.info("Input:  %s", args.input)
    log.info("Model:  %s", args.model)
    log.info("Duration: %.1fs  |  Prompt: %.1fs", args.duration, args.prompt_duration)
    if args.description:
        log.info("Style hint: %s", args.description)

    import torch
    import torchaudio

    log.info("Loading model (weights -> /Library/Assets/LLM/musicgen/)")
    from audiocraft.models import MusicGen

    model = MusicGen.get_pretrained(args.model, device=args.device)
    model.set_generation_params(duration=args.duration, top_k=250, temperature=1.0, cfg_coef=3.0)

    # Load input audio
    wav, sr = torchaudio.load(args.input)
    log.info("Loaded audio: %.1fs @ %d Hz (%d ch)", wav.shape[-1] / sr, sr, wav.shape[0])

    # Trim to last prompt_duration_s seconds
    max_prompt_samples = int(args.prompt_duration * sr)
    prompt_wav = wav[..., -max_prompt_samples:] if wav.shape[-1] > max_prompt_samples else wav
    log.info("Using %.1fs prompt from end of track", prompt_wav.shape[-1] / sr)

    # MusicGen expects [batch, channels, time]
    prompt_tensor = prompt_wav.unsqueeze(0).to(args.device)

    log.info("Generating %.1fs of continuation ...", args.duration)
    t0 = time.time()

    with torch.no_grad():
        output = model.generate_continuation(
            prompt=prompt_tensor,
            prompt_sample_rate=sr,
            descriptions=[args.description],
            progress=True,
        )

    elapsed = time.time() - t0
    model_sr = model.sample_rate
    output_wav = output[0].cpu()  # [C, T]
    actual_s = output_wav.shape[-1] / model_sr
    log.info("Done in %.1fs  ->  %.1fs of audio at %d Hz", elapsed, actual_s, model_sr)

    if args.join:
        # Resample prompt to model sample rate so concatenation is seamless
        prompt_resampled = torchaudio.functional.resample(prompt_wav, sr, model_sr)
        # Reconcile channel count: MusicGen outputs 1ch; prompt may be stereo.
        # Convert to mono by averaging if needed so cat doesn't blow up.
        if prompt_resampled.shape[0] != output_wav.shape[0]:
            if output_wav.shape[0] == 1 and prompt_resampled.shape[0] > 1:
                prompt_resampled = prompt_resampled.mean(dim=0, keepdim=True)
            elif prompt_resampled.shape[0] == 1 and output_wav.shape[0] > 1:
                prompt_resampled = prompt_resampled.expand_as(output_wav)
        output_wav = torch.cat([prompt_resampled, output_wav], dim=-1)
        total_s = output_wav.shape[-1] / model_sr
        log.info("Joined prompt + continuation: %.1fs total", total_s)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torchaudio.save(args.output, output_wav, model_sr)
    log.info("Saved: %s", args.output)
    log.info("Play:  ffplay %r  (or open in any audio player)", args.output)


if __name__ == "__main__":
    main()
