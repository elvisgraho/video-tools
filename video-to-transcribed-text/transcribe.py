#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RTX 4070 Ti SUPER Transcriber (Optimized)
- GPU: Force Float16 + Batching (Max Speed)
- Guitar/Music Mode: Prevents looping during instrumental parts.
- VAD: Skips silence processing but KEEPS timestamps perfect for SRT.
- Resume Support: Tracks processed files in logs for instant resume.
"""

import argparse
import os
import subprocess
import sys
import shutil
import logging
import torch
import gc
import time
import signal
from pathlib import Path

from tqdm.auto import tqdm
from typing import Set, List, Optional
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from faster_whisper import WhisperModel, BatchedInferencePipeline

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Transcriber")

def check_gpu():
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        name = torch.cuda.get_device_name(0)
        logger.info(f"🚀 GPU DETECTED: {name} | VRAM: {vram:.1f} GB")
        return True
    else:
        logger.warning("⚠️  GPU NOT FOUND. This will be very slow on CPU.")
        return False

# --- Log File Management ---
def get_log_dir(args) -> Path:
    """Returns the directory where log files should be stored."""
    if args.output_dir:
        return Path(args.output_dir)
    elif args.inplace:
        return Path(args.input).resolve() if Path(args.input).is_dir() else Path(args.input).parent.resolve()
    else:
        return Path("output")

def load_processed_files(log_path: Path) -> Set[str]:
    """Load set of already processed file paths from log."""
    processed = set()
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    processed.add(line)
    return processed

def mark_processed(log_path: Path, file_path: Path):
    """Append a file to the processed log."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(str(file_path.resolve()) + "\n")

def mark_failed(log_path: Path, file_path: Path, error: str):
    """Append a file to the failed log with error message."""
    with open(log_path, "a", encoding="utf-8") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} | {file_path.resolve()} | {error}\n")

# --- Processing Statistics ---
@dataclass
class ProcessingStats:
    """Tracks processing statistics for ETA and throughput calculations."""
    total_files: int
    batch_start_time: float = field(default_factory=time.time)
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    # Rolling window of recent processing times (last 50 files) for adaptive ETA
    recent_times: deque = field(default_factory=lambda: deque(maxlen=50))

    def record_processed(self, elapsed_seconds: float):
        """Record a successfully processed file with its processing time."""
        self.files_processed += 1
        self.recent_times.append(elapsed_seconds)

    def record_skipped(self):
        """Record a skipped file."""
        self.files_skipped += 1

    def record_failed(self):
        """Record a failed file."""
        self.files_failed += 1

    @property
    def files_done(self) -> int:
        """Total files handled (processed + skipped + failed)."""
        return self.files_processed + self.files_skipped + self.files_failed

    @property
    def files_remaining(self) -> int:
        """Files still to process."""
        return self.total_files - self.files_done

    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed time since batch started."""
        return time.time() - self.batch_start_time

    @property
    def avg_time_per_file(self) -> Optional[float]:
        """Average processing time per file (from rolling window)."""
        if not self.recent_times:
            return None
        return sum(self.recent_times) / len(self.recent_times)

    @property
    def files_per_minute(self) -> Optional[float]:
        """Current throughput in files per minute."""
        avg = self.avg_time_per_file
        if avg is None or avg == 0:
            return None
        return 60.0 / avg

    @property
    def files_per_hour(self) -> Optional[float]:
        """Current throughput in files per hour."""
        fpm = self.files_per_minute
        if fpm is None:
            return None
        return fpm * 60

    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimated seconds until completion."""
        avg = self.avg_time_per_file
        if avg is None:
            return None
        # Only count non-skipped remaining files for ETA
        return self.files_remaining * avg

    def format_duration(self, seconds: float) -> str:
        """Format seconds into human readable duration."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            remaining_minutes = (seconds % 3600) / 60
            return f"{hours:.0f}h {remaining_minutes:.0f}m"

    def format_eta(self) -> str:
        """Format ETA as time remaining and estimated completion time."""
        eta = self.eta_seconds
        if eta is None:
            return "calculating..."

        # Time remaining
        remaining = self.format_duration(eta)

        # Estimated completion time
        completion_time = datetime.now() + timedelta(seconds=eta)
        completion_str = completion_time.strftime("%H:%M")

        return f"{remaining} (done ~{completion_str})"

    def get_progress_line(self, current_file_num: int) -> str:
        """Generate a progress status line."""
        parts = [f"[{current_file_num}/{self.total_files}]"]

        # Add throughput if available
        fpm = self.files_per_minute
        if fpm is not None:
            parts.append(f"⚡ {fpm:.1f} files/min")

        # Add ETA
        parts.append(f"ETA: {self.format_eta()}")

        return " | ".join(parts)

    def get_summary(self) -> List[str]:
        """Generate summary lines for end of batch."""
        elapsed = self.format_duration(self.elapsed_seconds)
        lines = [
            f"   Total files:     {self.total_files}",
            f"   Processed:       {self.files_processed}",
            f"   Skipped:         {self.files_skipped}",
            f"   Failed:          {self.files_failed}",
            f"   Total time:      {elapsed}",
        ]

        # Add throughput stats if we processed files
        if self.files_processed > 0:
            avg = self.avg_time_per_file
            if avg:
                lines.append(f"   Avg time/file:   {avg:.1f}s")
            fpm = self.files_per_minute
            if fpm:
                lines.append(f"   Throughput:      {fpm:.1f} files/min ({fpm * 60:.0f}/hour)")

        return lines

def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        logger.critical("FFmpeg not found! Please install FFmpeg and add it to System PATH.")
        sys.exit(1)

def format_timestamp(seconds: float):
    """Formats time for SRT (00:00:00,000)"""
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def extract_audio(video_path, temp_wav_path):
    """Extracts audio to 16kHz WAV safely using FFmpeg."""
    # Handle Windows long paths
    safe_video = str(video_path.resolve())
    if os.name == 'nt' and not safe_video.startswith("\\\\?\\"):
        safe_video = f"\\\\?\\{safe_video}"
    
    command = [
        "ffmpeg", "-y", "-v", "error",
        "-i", safe_video,
        "-ac", "1", "-ar", "16000", "-vn", 
        "-f", "wav", str(temp_wav_path)
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        logger.error(f"FFmpeg failed to extract audio from: {video_path.name}")
        return False

def write_srt(segments, srt_path, max_chars: int = 42, max_words: int = 8,
              pause_threshold: float = 0.4, orphan_threshold: int = 3):
    """
    Writes synchronized subtitles with readable chunk sizes.
    Uses word-level timestamps for accurate splits.
    Splits on natural pauses in speech for better readability.
    Avoids orphaned words by extending subtitles when close to end.

    Args:
        segments: Whisper segments with word timestamps
        srt_path: Output file path
        max_chars: Max characters per line (~42 is standard for subtitles)
        max_words: Max words per subtitle block
        pause_threshold: Seconds of silence that triggers a split (default: 0.4s)
        orphan_threshold: If <= this many words remain, include them in current subtitle (default: 3)
    """
    subtitles = []

    for segment in segments:
        # Check if we have word-level timestamps
        if hasattr(segment, 'words') and segment.words:
            words_list = list(segment.words)
            total_words = len(words_list)
            current_words = []
            current_start = None
            last_word_end = None

            for word_idx, word in enumerate(words_list):
                words_remaining_after_this = total_words - word_idx - 1

                # Detect pause: gap between last word end and current word start
                has_pause = False
                if last_word_end is not None:
                    gap = word.start - last_word_end
                    if gap >= pause_threshold:
                        has_pause = True

                # If there's a pause and we have words, split here
                # But don't split if it would leave orphan words
                if has_pause and current_words and words_remaining_after_this > orphan_threshold:
                    current_text = ''.join(current_words).strip()
                    if current_text:
                        subtitles.append({
                            'start': current_start,
                            'end': last_word_end,
                            'text': current_text
                        })
                    current_words = []
                    current_start = None

                # Start new subtitle if needed
                if current_start is None:
                    current_start = word.start

                current_words.append(word.word)
                current_text = ''.join(current_words).strip()
                last_word_end = word.end

                # Check if we should split (max words/chars reached)
                should_split = (
                    len(current_words) >= max_words or
                    len(current_text) >= max_chars * 2  # Allow 2 lines
                )

                # Don't split if it would leave orphan words at the end
                if should_split and words_remaining_after_this <= orphan_threshold:
                    should_split = False  # Extend to include remaining words

                if should_split:
                    subtitles.append({
                        'start': current_start,
                        'end': word.end,
                        'text': current_text
                    })
                    current_words = []
                    current_start = None

            # Don't forget remaining words
            if current_words:
                current_text = ''.join(current_words).strip()
                if current_text:
                    subtitles.append({
                        'start': current_start,
                        'end': segment.words[-1].end,
                        'text': current_text
                    })
        else:
            # Fallback: split by character count if no word timestamps
            text = segment.text.strip()
            words = text.split()
            words_total = len(words)

            if len(text) <= max_chars * 2 and words_total <= max_words:
                # Short enough, use as-is
                subtitles.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': text
                })
            else:
                # Split into chunks
                chunk_words = []
                chunk_start = segment.start
                segment_duration = segment.end - segment.start

                for idx, word in enumerate(words):
                    words_remaining = words_total - idx - 1
                    chunk_words.append(word)
                    chunk_text = ' '.join(chunk_words)

                    should_split = (
                        len(chunk_words) >= max_words or
                        len(chunk_text) >= max_chars * 2
                    )

                    # Don't split if it would leave orphan words
                    if should_split and words_remaining <= orphan_threshold:
                        should_split = False

                    if should_split or idx == words_total - 1:
                        # Estimate timing based on word position
                        progress = (idx + 1) / words_total
                        chunk_end = segment.start + (segment_duration * progress)

                        subtitles.append({
                            'start': chunk_start,
                            'end': chunk_end,
                            'text': chunk_text
                        })
                        chunk_words = []
                        chunk_start = chunk_end

    # Write to file
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, sub in enumerate(subtitles, start=1):
            start = format_timestamp(sub['start'])
            end = format_timestamp(sub['end'])
            text = sub['text']

            # Optional: wrap long lines into 2 lines
            if len(text) > max_chars:
                words = text.split()
                mid = len(words) // 2
                line1 = ' '.join(words[:mid])
                line2 = ' '.join(words[mid:])
                text = f"{line1}\n{line2}"

            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

def write_txt(segments, txt_path):
    """Writes continuous text (silence effectively removed)."""
    with open(txt_path, "w", encoding="utf-8") as f:
        for segment in segments:
            f.write(segment.text.strip() + " ")

def process_file(model, file_path, args, processed_log: Path, failed_log: Path,
                  processed_set: Set[str], stats: ProcessingStats) -> tuple[str, Optional[float]]:
    """
    Process a single file.
    Returns: (status, elapsed_time) where status is 'processed', 'skipped', or 'failed'
    """
    start_time = time.time()
    file_resolved = str(file_path.resolve())
    progress = stats.get_progress_line(stats.files_done + 1)

    # -- SKIP IF ALREADY PROCESSED (from log) --
    if file_resolved in processed_set:
        logger.info(f"{progress} Skipping {file_path.name} (already in processed.log)")
        return ('skipped', None)

    # -- PATH SETUP --
    # pathlib.with_suffix replaces the LAST extension only.
    # "Movie.Part1.mp4" -> "Movie.Part1.srt" (Preserves dots)
    if args.inplace:
        srt_path = file_path.with_suffix(".srt")
        txt_path = file_path.with_suffix(".txt")
        # Temp file next to video
        temp_wav = file_path.parent / f".tmp_{file_path.stem}_{int(time.time())}.wav"
    else:
        # Custom output folder
        out_root = Path(args.output_dir) if args.output_dir else Path("output")
        out_root.mkdir(parents=True, exist_ok=True)
        srt_path = out_root / file_path.with_suffix(".srt").name
        txt_path = out_root / file_path.with_suffix(".txt").name
        temp_wav = out_root / f".tmp_{file_path.stem}_{int(time.time())}.wav"

    # -- SKIP IF OUTPUT FILES ALREADY EXIST --
    srt_exists = srt_path.exists() if args.srt else True
    txt_exists = txt_path.exists() if args.txt else True

    if srt_exists and txt_exists:
        logger.info(f"{progress} Skipping {file_path.name} (output files exist)")
        # Mark as processed so future runs skip faster
        mark_processed(processed_log, file_path)
        processed_set.add(file_resolved)
        return ('skipped', None)

    logger.info(f"{progress} Processing: {file_path.name}")

    if not extract_audio(file_path, temp_wav):
        mark_failed(failed_log, file_path, "FFmpeg audio extraction failed")
        return ('failed', None)

    try:
        # -- TRANSCRIPTION --
        # condition_on_previous_text=False is CRITICAL for guitar/music.
        # It stops the AI from repeating "Thank you" or "Okay" during guitar solos.
        segments_generator, info = model.transcribe(
            str(temp_wav),
            beam_size=args.beam_size,
            language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            batch_size=args.batch_size,
            condition_on_previous_text=False,
            word_timestamps=True if args.srt else False
        )

        segments = []
        with tqdm(total=int(info.duration), unit="sec", desc="Transcribing", leave=False) as pbar:
            last_pos = 0
            for segment in segments_generator:
                segments.append(segment)
                current_pos = segment.end
                if current_pos > last_pos:
                    pbar.update(int(current_pos - last_pos))
                    last_pos = current_pos

        # -- OUTPUT GENERATION --
        if args.srt:
            write_srt(segments, srt_path, max_chars=args.max_chars, max_words=args.max_words,
                      pause_threshold=args.pause_split)
            logger.info(f"✅ Created SRT: {srt_path.name}")

        if args.txt:
            write_txt(segments, txt_path)
            logger.info(f"✅ Created TXT: {txt_path.name}")

        # Mark as successfully processed
        elapsed = time.time() - start_time
        logger.info(f"✅ Completed in {elapsed:.1f}s")
        mark_processed(processed_log, file_path)
        processed_set.add(file_resolved)
        return ('processed', elapsed)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Error on {file_path.name}: {error_msg}")
        mark_failed(failed_log, file_path, error_msg)
        return ('failed', None)
    finally:
        # Cleanup
        if temp_wav.exists():
            try: os.remove(temp_wav)
            except: pass

        # Free VRAM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="RTX 4070 Ti Super Transcriber")

    parser.add_argument("--input", "-i", required=True, help="Input file or folder.")

    # FLAGS to choose what you want
    parser.add_argument("--srt", action="store_true", help="Create Subtitles (.srt)")
    parser.add_argument("--txt", action="store_true", help="Create Transcript (.txt)")

    parser.add_argument("--inplace", action="store_true", help="Save files next to video.")
    parser.add_argument("--output-dir", "-o", help="Custom output folder.")
    parser.add_argument("--model", default="large-v3-turbo", help="Model size.")
    parser.add_argument("--reset-logs", action="store_true", help="Clear processed/failed logs and start fresh.")

    # Subtitle formatting options
    parser.add_argument("--max-words", type=int, default=12, help="Max words per subtitle (default: 12)")
    parser.add_argument("--max-chars", type=int, default=52, help="Max chars per line (default: 52)")
    parser.add_argument("--pause-split", type=float, default=1, help="Split subtitle on pauses longer than X seconds (default: 0.4)")

    # Performance tuning options
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size for decoding (1=fastest, 5=most accurate, default: 1)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for GPU inference (default: 32, reduce if OOM)")
    parser.add_argument("--cpu-threads", type=int, default=8, help="CPU threads for feature extraction (default: 8)")

    args = parser.parse_args()

    # Safety Check: Did the user ask for anything?
    if not args.srt and not args.txt:
        print("\n⚠️  WARNING: You did not select an output format!")
        print("Please add --srt (for subtitles) or --txt (for text) or both.\n")
        print(f"Example: python {sys.argv[0]} --input \"C:\\Videos\" --inplace --srt")
        sys.exit(1)

    check_ffmpeg()
    check_gpu()

    # --- Setup Log Files ---
    log_dir = get_log_dir(args)
    log_dir.mkdir(parents=True, exist_ok=True)
    processed_log = log_dir / "processed.log"
    failed_log = log_dir / "failed.log"

    # Reset logs if requested
    if args.reset_logs:
        if processed_log.exists():
            processed_log.unlink()
            logger.info("Cleared processed.log")
        if failed_log.exists():
            failed_log.unlink()
            logger.info("Cleared failed.log")

    # Load previously processed files for instant skip
    processed_set = load_processed_files(processed_log)
    if processed_set:
        logger.info(f"📋 Loaded {len(processed_set)} previously processed files from log")

    # Load Model
    logger.info("Loading Model (Float16) with Batched Inference...")
    use_gpu = check_gpu()

    try:
        if use_gpu:
            # Optimized for RTX 4070 Ti SUPER (16GB VRAM)
            # - int8_float16: Weights in int8, compute in float16 (faster, minimal quality loss)
            # - cpu_threads: Parallel feature extraction on CPU
            # - num_workers: Parallel audio preprocessing
            base_model = WhisperModel(
                args.model,
                device="cuda",
                compute_type="int8_float16",
                cpu_threads=args.cpu_threads,
                num_workers=4
            )
        else:
            logger.info("Using CPU mode (int8 for speed)")
            base_model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=args.cpu_threads)
        model = BatchedInferencePipeline(model=base_model)
    except Exception as e:
        if "cublas" in str(e).lower() or "cudnn" in str(e).lower() or "cuda" in str(e).lower():
            logger.warning(f"CUDA error: {e}")
            logger.info("Falling back to CPU mode...")
            try:
                base_model = WhisperModel(args.model, device="cpu", compute_type="int8")
                model = BatchedInferencePipeline(model=base_model)
            except Exception as e2:
                logger.critical(f"Model Load Failed: {e2}")
                sys.exit(1)
        else:
            logger.critical(f"Model Load Failed: {e}")
            sys.exit(1)

    # Find Files
    input_path = Path(args.input)
    files = []
    exts = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.mp3', '.wav', '.m4a'}

    if input_path.is_file():
        files.append(input_path)
    elif input_path.is_dir():
        for f in input_path.rglob('*'):
            if f.suffix.lower() in exts:
                files.append(f)

    # Sort files for predictable processing order (alphabetically by full path)
    files.sort(key=lambda f: str(f).lower())

    total_files = len(files)
    logger.info(f"Found {total_files} media files.")

    # Initialize processing statistics
    stats = ProcessingStats(total_files=total_files)

    if processed_set:
        remaining = sum(1 for f in files if str(f.resolve()) not in processed_set)
        logger.info(f"📊 {remaining} files remaining to process")

    # --- Graceful shutdown handling ---
    shutdown_requested = False

    def signal_handler(_signum, _frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            logger.warning("Force quit requested. Exiting immediately.")
            sys.exit(1)
        shutdown_requested = True
        logger.warning("\n⚠️  Shutdown requested. Finishing current file... (Press Ctrl+C again to force quit)")

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    # --- Process Files ---
    for file in files:
        if shutdown_requested:
            logger.info("🛑 Stopping gracefully. Progress saved to processed.log")
            break

        status, elapsed = process_file(model, file, args, processed_log, failed_log,
                                        processed_set, stats)

        # Update statistics
        if status == 'processed':
            stats.record_processed(elapsed)
        elif status == 'skipped':
            stats.record_skipped()
        else:  # failed
            stats.record_failed()

    # --- Summary ---
    logger.info("=" * 50)
    logger.info("📊 BATCH SUMMARY")
    for line in stats.get_summary():
        logger.info(line)
    logger.info(f"📁 Logs saved to: {log_dir}")
    if stats.files_failed > 0:
        logger.info(f"⚠️  Check {failed_log.name} for failed files")
    logger.info("🎉 Batch Complete.")

if __name__ == "__main__":
    main()