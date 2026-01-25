import subprocess
import json
import os
import sys
import time
import argparse
from pathlib import Path
import multiprocessing
from datetime import datetime

# ================= CONFIGURATION =================
FFMPEG_AUDIO_ARGS = [
    "-c:v", "copy",
    "-c:a", "aac",
    "-b:a", "96k",
    "-ac", "2",
    "-ar", "44100",
    "-af", "loudnorm=I=-16:TP=-1.5:LRA=11"
]

HISTORY_FILE = "processing_history.json"
# =================================================

class ProcessManager:
    def __init__(self, input_root, output_root, replace_mode):
        self.input_root = Path(input_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.replace_mode = replace_mode
        self.history = self._load_history()
        
    def _load_history(self):
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_history(self):
        try:
            with open(HISTORY_FILE, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")

    def is_processed(self, file_path):
        path_str = str(file_path)
        try:
            mtime = os.path.getmtime(file_path)
        except OSError:
            return False
            
        if path_str in self.history:
            if self.history[path_str]['mtime'] == mtime:
                return True
        return False

    def mark_processed(self, file_path):
        self.history[str(file_path)] = {
            'mtime': os.path.getmtime(file_path),
            'processed_at': datetime.now().isoformat()
        }

def process_file_worker(file_data):
    """
    Worker function running in a separate process.
    """
    # IGNORE KeyboardInterrupt in workers so the main process handles the exit logic
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    input_path, output_path, is_replace = file_data
    
    # Temp file to ensure atomic write
    temp_path = output_path.with_suffix(f".temp_{os.getpid()}.mp4")
    
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(input_path),
        *FFMPEG_AUDIO_ARGS,
        "-movflags", "+faststart",
        str(temp_path)
    ]

    start_t = time.time()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True, capture_output=True)
        
        if is_replace:
            os.replace(temp_path, input_path)
        else:
            os.replace(temp_path, output_path)
            
        return ("OK", str(input_path), time.time() - start_t)

    except subprocess.CalledProcessError:
        if temp_path.exists(): os.remove(temp_path)
        return ("ERROR", str(input_path), 0)
    except Exception as e:
        if temp_path.exists(): os.remove(temp_path)
        return ("FAIL", f"{str(input_path)} ({str(e)})", 0)

def main():
    parser = argparse.ArgumentParser(description="Super-Optimized Audio Normalizer")
    parser.add_argument("--input", "-i", default=".", help="Input folder")
    parser.add_argument("--output", "-o", default="./processed", help="Output folder")
    parser.add_argument("--workers", "-w", type=int, default=(os.cpu_count()-1), help="Number of CPU threads")
    parser.add_argument("--replace", action="store_true", help="Overwrite original files")
    args = parser.parse_args()

    print(f"=== Audio Normalization Engine ===")
    print(f"Detected {os.cpu_count()} CPU cores.")
    print(f"Workers: {args.workers}")
    
    manager = ProcessManager(args.input, args.output, args.replace)
    
    print("Scanning directory tree...")
    all_files = sorted(Path(args.input).rglob("*.mp4"))
    
    tasks = []
    skipped_count = 0
    
    for f in all_files:
        if "processed" in str(f) and not args.replace: continue
        if f.name.startswith(".temp_"): continue
            
        if manager.is_processed(f):
            skipped_count += 1
            continue

        if args.replace:
            out_f = f
        else:
            try:
                rel_path = f.relative_to(manager.input_root)
            except ValueError:
                # Fallback if file is not inside input_root (e.g. symlinks)
                rel_path = f.name
            out_f = manager.output_root / rel_path

        tasks.append((f, out_f, args.replace))

    print(f"Found {len(all_files)} files.")
    print(f"Skipping {skipped_count} previously processed.")
    print(f"Queued {len(tasks)} files.")
    
    if not tasks:
        print("No work to do.")
        return

    start_time = time.time()
    completed = 0
    errors = 0
    save_counter = 0 
    
    print("\nStarting Workers... (Press Ctrl+C to stop safely)")
    
    # ================= CHANGED: Using multiprocessing.Pool =================
    pool = multiprocessing.Pool(processes=args.workers)
    
    try:
        # imap_unordered yields results as soon as they finish, like as_completed
        for result in pool.imap_unordered(process_file_worker, tasks):
            status, fname, duration = result
            
            if status == "OK":
                completed += 1
                manager.mark_processed(fname)
                save_counter += 1
                
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = len(tasks) - completed
                eta = remaining / rate if rate > 0 else 0
                
                print(f"\r[OK] {completed}/{len(tasks)} | {rate*60:.1f} f/m | ETA: {int(eta//60)}m {int(eta%60)}s | {Path(fname).name}", end="")
                
                if save_counter >= 50:
                    manager.save_history()
                    save_counter = 0
            else:
                errors += 1
                print(f"\n[{status}] {fname}")
                
        # Normal exit
        pool.close()
        pool.join()

    except KeyboardInterrupt:
        print("\n\n!!! Keyboard Interrupt Detected !!!")
        print("Terminating worker processes immediately...")
        pool.terminate() # Kills all workers instantly
        pool.join()      # Cleans up the zombies
        print("Saving progress before exit...")
        manager.save_history()
        sys.exit(0)
    # =======================================================================

    manager.save_history()
    
    total_time = time.time() - start_time
    print(f"\n\n=== DONE ===")
    print(f"Processed: {completed}")
    print(f"Errors:    {errors}")
    print(f"Total Time: {int(total_time//60)}m {int(total_time%60)}s")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()