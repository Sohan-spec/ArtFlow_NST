import os
import sys
import subprocess
import time
import re
from flask import Flask, render_template, request, jsonify, send_from_directory, Response

app = Flask(__name__, template_folder='static')

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
CONTENT_IMAGES_DIR = os.path.join(DATA_DIR, 'content-images')
STYLE_IMAGES_DIR = os.path.join(DATA_DIR, 'style-images')
OUTPUT_IMAGES_DIR = os.path.join(DATA_DIR, 'output-images')

os.makedirs(CONTENT_IMAGES_DIR, exist_ok=True)
os.makedirs(STYLE_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

# Temp folder for custom uploaded styles (session-based, not permanent)
TEMP_STYLES_DIR = os.path.join(DATA_DIR, 'temp-styles')
os.makedirs(TEMP_STYLES_DIR, exist_ok=True)

# Clean temp styles on startup
for f in os.listdir(TEMP_STYLES_DIR):
    try:
        os.remove(os.path.join(TEMP_STYLES_DIR, f))
    except:
        pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/styles')
def list_styles():
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    styles = [
        f for f in os.listdir(STYLE_IMAGES_DIR)
        if os.path.splitext(f)[1].lower() in valid_extensions
        and 'resized' not in f.lower()
        and 'crop' not in f.lower()
    ]
    return jsonify(styles)

@app.route('/data/style-images/<filename>')
def serve_style_image(filename):
    return send_from_directory(STYLE_IMAGES_DIR, filename)

@app.route('/data/output-images/<path:filename>')
def serve_output_image(filename):
    return send_from_directory(OUTPUT_IMAGES_DIR, filename)

@app.route('/upload_content', methods=['POST'])
def upload_content():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    filename = file.filename # In prod, secure_filename needed
    save_path = os.path.join(CONTENT_IMAGES_DIR, filename)
    file.save(save_path)
    
    return jsonify({'filename': filename})

@app.route('/upload_style', methods=['POST'])
def upload_style():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save to temp folder instead of permanent style-images
    import uuid
    unique_id = uuid.uuid4().hex[:8]
    filename = f'temp_{unique_id}_{file.filename}'
    save_path = os.path.join(TEMP_STYLES_DIR, filename)
    file.save(save_path)
    
    return jsonify({'filename': filename, 'url': f'/data/temp-styles/{filename}'})

@app.route('/data/temp-styles/<filename>')
def serve_temp_style(filename):
    return send_from_directory(TEMP_STYLES_DIR, filename)

@app.route('/predict')
def predict():
    content_img_name = request.args.get('content_img_name')
    style_img_name = request.args.get('style_img_name')
    
    if not content_img_name or not style_img_name:
        return "Missing parameters", 400
    
    # Handle temp style images - copy to style folder for NST to find
    import shutil
    temp_style_copied = False
    if style_img_name.startswith('temp_'):
        temp_path = os.path.join(TEMP_STYLES_DIR, style_img_name)
        if os.path.exists(temp_path):
            shutil.copy(temp_path, os.path.join(STYLE_IMAGES_DIR, style_img_name))
            temp_style_copied = True

    # Build command to run neural_style_transfer.py
    # We use subprocess to capture stdout for progress bars
    
    # We need to construct the output directory/filename that NST will create
    # logic from neural_style_transfer.py:
    # output_dir_name= 'All_' +os.path.split(content_image_path)[1].split('.')[0]+'_'+os.path.split(style_image_path)[1].split('.')[0]
    # dump_path=os.path.join(config['output_img_dir'],output_dir_name)
    # The final image depends on config['saving_freq']. Default is 100.
    # If we set saving_freq to -1, it saves final result. 
    # But usually it saves with iteration numbers.
    # Let's ensure strict params to predict where it goes.
    
    c_name = os.path.splitext(content_img_name)[0]
    s_name = os.path.splitext(style_img_name)[0]
    dir_name = f"All_{c_name}_{s_name}"
    full_out_dir = os.path.join(OUTPUT_IMAGES_DIR, dir_name)
    
    # 1. Clean output directory to ensure fresh results
    if os.path.exists(full_out_dir):
        for f in os.listdir(full_out_dir):
            try:
                os.remove(os.path.join(full_out_dir, f))
            except:
                pass

    cmd = [
        sys.executable, 'neural_style_transfer.py',
        '--content_img_name', content_img_name,
        '--style_img_name', style_img_name,
        '--optimizer', 'lbfgs',
        '--model', 'vgg19',
        '--saving_freq', '50',  # Save progress images
        '--height', '400',
        '--content_weight', '1e5',
        '--style_weight', '3e2'
    ]
    
    def generate():
        yield f"data: {{\"log\": \"Starting neural style transfer...\"}}\\n\\n"
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'}  # Force unbuffered output
            )
        except Exception as e:
            yield f"data: {{\"error\": \"Failed to start process: {str(e)}\"}}\\n\\n"
            return
        
        # Regex to catch: "using LBFGS, iteration : 050, total_loss : ..."
        # LBFGS default max_iter is 600 in NST code
        max_iter = 600
        
        for line in process.stdout:
            match = re.search(r'iteration\s*:\s*(\d+)', line)
            if match:
                iteration = int(match.group(1))
                progress = min(int((iteration / max_iter) * 100), 100)
                # Estimate time? Simple linear extrapolation might be jumpy but better than nothing.
                # Just streaming progress % for now.
                yield f"data: {{\"progress\": {progress}, \"status\": \"processing\", \"log\": \"{line.strip()}\"}}\n\n"
            else:
                 # Forward other logs
                 cleaned_line = line.strip().replace('"', '\\"')
                 yield f"data: {{\"log\": \"{cleaned_line}\"}}\n\n"
        
        process.wait()
        
        if process.returncode == 0:
            # Construct expected output path to return to frontend
            # Logic from NST:
            # c_name and s_name are already computed above
            
            try:
                # Find the generated image - look for numeric filenames (0050.jpg etc)
                files = os.listdir(full_out_dir)
                # Filter strictly for numeric images generated by saving_freq > 0
                images = [f for f in files if f.endswith('.jpg') and f.replace('.jpg', '').isdigit()]
                
                if images:
                    # Sort by integer value of filename to get the latest iteration
                    images.sort(key=lambda x: int(x.split('.')[0]), reverse=True)
                    final_image = images[0]
                    image_url = f"/data/output-images/{dir_name}/{final_image}"
                    yield f"data: {{\"progress\": 100, \"status\": \"done\", \"url\": \"{image_url}\"}}\n\n"
                else:
                     # Fallback check for non-numeric if something weird happened
                     all_images = [f for f in files if f.endswith('.jpg')]
                     if all_images:
                        final_image = all_images[0]
                        image_url = f"/data/output-images/{dir_name}/{final_image}"
                        yield f"data: {{\"progress\": 100, \"status\": \"done\", \"url\": \"{image_url}\"}}\n\n"
                     else:   
                        yield f"data: {{\"error\": \"Output image not found\"}}\n\n"
            except FileNotFoundError:
                yield f"data: {{\"error\": \"Output directory not found\"}}\n\n"
        else:
             yield f"data: {{\"error\": \"Process failed with code {process.returncode}\"}}\n\n"
        
        # Cleanup: remove temp style copy from style-images folder
        if temp_style_copied:
            try:
                os.remove(os.path.join(STYLE_IMAGES_DIR, style_img_name))
            except:
                pass

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    # Cloud Run/Hugging Face compatibility
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)
