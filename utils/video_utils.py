import os
import subprocess

def create_video_from_results(results_path,img_format):
    import shutil #this is to manipulate like folders and files
    output_file_name='merged_output.mp4'
    fps=24
    first_frame=0
    number_of_frames_to_process=len(os.listdir(results_path))
    
    ffmpeg='ffmpeg'
    if shutil.which(ffmpeg):
        img_name_format='%' + str(img_format[0])+'d'+(img_format[1])
        pattern=os.path.join(results_path,img_name_format)
        video_save_path=os.path.join(results_path,output_file_name)
        
        trim_video_comm=['-start_number',str(first_frame),'-vframes',str(number_of_frames_to_process)] #this is to start from the first frame and the go until the last
        input_options=['-r',str(fps),'-i',pattern] #read all the images in the pattern path at 24 fps(so if 300 images-> 300/24, you get 12.5s vid)
        encoding_options=['-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p']
        #libx264 is a video codedc format, crf is the quality level(lower is better) and yuv420p is pixel format supported by most devices
        subprocess.call([ffmpeg,*input_options,*trim_video_comm,*encoding_options,video_save_path])
    else:
        print("ffmpeg was not installed, install it and try again later")