import os
import subprocess
import shutil
import tempfile

def create_video_from_results(results_path, img_format):
    output_file_name = 'cool_output.mp4'
    fps = 2 #you aint flash buddy, sooo vid gon be slow, face it

    ffmpeg = shutil.which("ffmpeg") #this like allows you to manipulate files and folers
    if ffmpeg is None:
        print("ffmpeg wasnt found in path")
        return

    video_save_path = os.path.join(results_path, output_file_name)

    frames = sorted(
        f for f in os.listdir(results_path)
        if f.endswith(img_format[1]))
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_file = f.name
        for frame in frames:
            f.write(f"file '{os.path.join(results_path, frame)}'\n")

    comm = [ffmpeg,"-y","-r", str(fps),"-f", "concat","-safe", "0","-i", concat_file,"-c:v", "libx264","-crf", "18","-pix_fmt", "yuv420p",video_save_path]
     #libx264 is a video codedc format, crf is the quality level(lower is better) and yuv420p is pixel format supported by most devices
    result = subprocess.run(comm, capture_output=True, text=True)

    os.remove(concat_file)

    if result.returncode != 0:
        print("ffmpeg failed")
