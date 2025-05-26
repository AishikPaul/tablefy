import os
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, flash
from werkzeug.utils import secure_filename
from .processing import process_table_image

[os.makedirs(d, exist_ok=True) for d in ("files", "result_files")]

app = Flask(__name__)
app.secret_key = "supersecretkey"  # For flashing messages

# Set folder paths relative to your project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "files")
RESULT_FOLDER = os.path.join(BASE_DIR, "result_files")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB upload

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)
            
            try:
                csv_file_path = process_table_image(input_path)
            except Exception as e:
                flash(f"Processing failed: {str(e)}")
                return redirect(request.url)

            csv_filename = os.path.basename(csv_file_path)
            return render_template('result.html', csv_file=csv_filename)
        else:
            flash('Allowed file types are png, jpg, jpeg, bmp')
            return redirect(request.url)

    return render_template('upload.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
