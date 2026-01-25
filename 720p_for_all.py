import os
import subprocess
import json
import shutil
import signal
import logging
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= USER CONFIGURATION =================
INPUT_FOLDER = r"C:\..."

# --- HYBRID SPEED SETTINGS ---
# Since we are moving decoding to CPU, we can likely run MORE threads 
# if your CPU has many cores (e.g., Ryzen 9, i7, i9).
# If you have a weaker CPU, set this to 4. If strong, set to 6 or 8.
GPU_LIMIT = 6 

SCAN_THREADS = 16 

# Target Resolution
TARGET_HEIGHT = 720
EXTENSIONS = ('.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg', '.mts', '.m2ts')

# --- BITRATE SETTINGS ---
BITRATE_TARGET = '600k'
MAX_BITRATE = '900k'
BUF_SIZE = '1200k' 

CODEC = 'h264_nvenc'
PRESET = 'p4'        # Fast/Medium (Best balance)
TUNE = 'hq'          

AUDIO_CODEC = 'aac'
AUDIO_BITRATE = '96k'

LOG_FILE = 'conversion_log.txt'
# =================================================

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')
stop_requested = False
gpu_semaphore = threading.Semaphore(GPU_LIMIT)

def signal_handler(sig, frame):
    global stop_requested
    stop_requested = True
signal.signal(signal.SIGINT, signal_handler)

def get_file_info(file_path):
    cmd = [
        'ffprobe', '-v', 'error', 
        '-show_entries', 'stream=index,codec_name,width,height,bit_rate,r_frame_rate', 
        '-analyzeduration', '5M', '-probesize', '5M',
        '-of', 'json', file_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', timeout=5)
        data = json.loads(res.stdout)
        video = next((s for s in data.get('streams', []) if s.get('codec_name') not in ['aac', 'mp3', 'ac3']), None)
        audio = next((s for s in data.get('streams', []) if s.get('codec_name') in ['aac', 'mp3', 'ac3']), None)
        return video, audio
    except:
        return None, None

def get_fps_float(fps_string):
    try:
        num, den = fps_string.split('/')
        return float(num) / float(den)
    except:
        return 30.0

def process_video(file_path):
    global stop_requested
    if stop_requested: return "STOPPED"

    v_info, a_info = get_file_info(file_path)
    if not v_info: return "ERROR_READ"

    current_h = int(v_info.get('height', 0))
    current_fps = get_fps_float(v_info.get('r_frame_rate', '30/1'))
    
    try: current_bitrate = int(v_info.get('bit_rate')) // 1000
    except: current_bitrate = 5000 

    is_h264 = v_info.get('codec_name') == 'h264'
    is_mp4 = file_path.lower().endswith('.mp4')

    if is_mp4 and is_h264 and (current_h <= TARGET_HEIGHT) and (current_bitrate < 750) and (current_fps <= 31):
         return "SKIP_ALREADY_SMALL"

    directory, filename = os.path.split(file_path)
    name_no_ext = os.path.splitext(filename)[0]
    final_output = os.path.join(directory, name_no_ext + ".mp4")
    temp_output = os.path.join(directory, name_no_ext + f".temp.mp4")

    if not is_mp4 and os.path.exists(final_output): return "SKIP_EXISTS"

    audio_params = ['-c:a', 'copy'] if a_info and a_info.get('codec_name') == 'aac' else ['-c:a', AUDIO_CODEC, '-b:a', AUDIO_BITRATE]
    
    # --- MAJOR CHANGE HERE ---
    # We init a hardware device for the FILTERS, but we do NOT set global -hwaccel.
    # This forces CPU decode -> Upload to GPU Memory -> GPU Scale -> GPU Encode
    vf_chain = f'hwupload,scale_cuda=-2:{TARGET_HEIGHT}:interp_algo=lanczos'

    fps_cmd_part = ['-r', '30'] if current_fps > 31 else ['-vsync', '0']

    cmd = [
        'ffmpeg', '-y',
        # Initialize CUDA device for the filters (scale_cuda)
        '-init_hw_device', 'cuda=cuda:0',
        '-filter_hw_device', 'cuda',
        
        # NOTE: -hwaccel cuda REMOVED. CPU will decode.
        '-i', file_path,
        *fps_cmd_part,
        '-vf', vf_chain, # Upload to GPU -> Scale
        '-c:v', CODEC,
        '-preset', PRESET,
        '-tune', TUNE,
        '-rc', 'vbr',
        '-b:v', BITRATE_TARGET,
        '-maxrate', MAX_BITRATE,
        '-bufsize', BUF_SIZE,
        '-bf', '2',
        '-temporal-aq', '1',
        '-movflags', '+faststart',
        *audio_params,
        '-loglevel', 'error', 
        temp_output
    ]

    try:
        with gpu_semaphore: 
            if stop_requested: return "STOPPED"
            subprocess.run(cmd, check=True)

        if is_mp4:
            os.replace(temp_output, file_path)
            return "UPDATED"
        else:
            os.replace(temp_output, final_output)
            if os.path.exists(final_output) and os.path.getsize(final_output) > 1024:
                try: os.remove(file_path)
                except: pass
            return "CONVERTED"
    except:
        if os.path.exists(temp_output): os.remove(temp_output)
        return "ERROR"

def main():
    if not shutil.which("ffmpeg"): return

    print(f"--- HYBRID MODE: CPU DECODE / GPU ENCODE ---")
    print(f"Workers: {SCAN_THREADS} | GPU Slots: {GPU_LIMIT}")
    
    files = []
    print("Scanning...")
    for r, d, f in os.walk(INPUT_FOLDER):
        for file in f:
            if file.lower().endswith(EXTENSIONS):
                files.append(os.path.join(r, file))

    print(f"Processing {len(files)} videos...")

    with ThreadPoolExecutor(max_workers=SCAN_THREADS) as executor:
        future_to_file = {executor.submit(process_video, f): f for f in files}
        with tqdm(total=len(files), unit="vid", dynamic_ncols=True) as pbar:
            for future in as_completed(future_to_file):
                try: status = future.result()
                except: status = "ERROR"
                
                short_name = os.path.basename(future_to_file[future])[:15]
                pbar.set_postfix(file=short_name, res=status if len(status)<5 else status[:4])
                pbar.update(1)
                
                if stop_requested:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

if __name__ == "__main__":
    main()