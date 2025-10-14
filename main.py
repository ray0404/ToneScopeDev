import os
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import librosa
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from pydub import AudioSegment
from scipy import signal

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'dev-secret-key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'flac', 'aiff', 'ogg', 'm4a'}

CUSTOM_PRESETS_FILE = 'custom_presets.json'
custom_presets = {}

def load_custom_presets():
    global custom_presets
    if os.path.exists(CUSTOM_PRESETS_FILE):
        with open(CUSTOM_PRESETS_FILE, 'r') as f:
            custom_presets = json.load(f)


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('files[]')
        uploaded_files = []
        
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_files.append(filename)
        
        if not uploaded_files:
            return jsonify({'error': 'No valid audio files were uploaded'}), 400
        
        return jsonify({'files': uploaded_files})
    except Exception as e:
        app.logger.error(f'Upload error: {str(e)}')
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_tracks():
    try:
        analysis_results = []
        upload_folder = app.config['UPLOAD_FOLDER']
        
        for filename in os.listdir(upload_folder):
            if allowed_file(filename):
                filepath = os.path.join(upload_folder, filename)
                
                y, sr = librosa.load(filepath, sr=None, mono=False)
                
                if y.ndim == 1:
                    y_mono = y
                    y_stereo = np.stack([y, y])
                else:
                    y_mono = librosa.to_mono(y)
                    y_stereo = y
                
                frequency_spectrum = calculate_frequency_spectrum(y_mono, sr)
                dynamic_range = calculate_dynamic_range(y_stereo, sr)
                stereo_image = calculate_stereo_image(y_stereo, sr)
                
                analysis_results.append({
                    'filename': filename,
                    'frequency_spectrum': frequency_spectrum,
                    'dynamic_range': dynamic_range,
                    'stereo_image': stereo_image
                })
        
        return jsonify({'analysis': analysis_results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def calculate_frequency_spectrum(y, sr):
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    
    max_freq = min(20000, sr / 2)
    freq_bins = np.logspace(np.log10(20), np.log10(max_freq), 100)
    spectrum = []
    valid_freqs = []
    
    for i in range(len(freq_bins) - 1):
        mask = (freqs >= freq_bins[i]) & (freqs < freq_bins[i + 1])
        if np.any(mask):
            spectrum.append(float(np.mean(20 * np.log10(S[mask] + 1e-10))))
            valid_freqs.append(float(freq_bins[i]))
    
    return {
        'frequencies': valid_freqs,
        'magnitudes': spectrum
    }

def calculate_dynamic_range(y, sr):
    if y.ndim == 1:
        y = np.stack([y, y])
    
    y_mono = np.mean(y, axis=0)
    
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y_mono)
    
    true_peak = 20 * np.log10(np.max(np.abs(y)) + 1e-10)
    
    rms = np.sqrt(np.mean(y_mono ** 2))
    peak = np.max(np.abs(y_mono))
    crest_factor = 20 * np.log10((peak / (rms + 1e-10)) + 1e-10)
    
    return {
        'lufs': float(loudness),
        'true_peak': float(true_peak),
        'crest_factor': float(crest_factor)
    }

def calculate_stereo_image(y, sr):
    if y.ndim == 1:
        return {'stereo_width': 0.0, 'correlation': 1.0}
    
    left = y[0]
    right = y[1]
    
    mid = (left + right) / 2
    side = (left - right) / 2
    
    mid_energy = np.sum(mid ** 2)
    side_energy = np.sum(side ** 2)
    
    stereo_width = float(side_energy / (mid_energy + side_energy + 1e-10))
    
    left_variance = np.var(left)
    right_variance = np.var(right)
    
    if left_variance < 1e-10 or right_variance < 1e-10:
        correlation = 1.0
    else:
        correlation = float(np.corrcoef(left, right)[0, 1])
    
    return {
        'stereo_width': stereo_width,
        'correlation': correlation
    }

@app.route('/apply-preset', methods=['POST'])
def apply_preset():
    try:
        data = request.json
        if not data or 'filename' not in data or 'preset' not in data:
            return jsonify({'error': 'Missing filename or preset'}), 400
        filename = secure_filename(data['filename'])
        preset = data['preset']
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        y, sr = librosa.load(filepath, sr=None, mono=False)
        
        if y.ndim == 1:
            y = np.stack([y, y])
        
        if preset == 'custom' and 'settings' in data:
            eq_curve = data['settings']
            y_processed = apply_eq_curve(y, sr, eq_curve)
        else:
            y_processed = apply_tonal_preset(y, sr, preset)
        
        ceiling = data.get('ceiling', -0.1)
        y_processed = peak_normalize(y_processed, target_dbfs=ceiling)
        
        output_filename = f"{os.path.splitext(filename)[0]}_{preset}.wav"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        
        sf.write(output_path, y_processed.T, sr)
        
        return jsonify({'processed_file': output_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def analyze_tonal_character(y, sr):
    curve = extract_eq_curve(y, sr)
    
    low_freq_limit = 500
    high_freq_limit = 4000
    
    low_mags = [p['magnitude'] for p in curve if p['freq'] < low_freq_limit]
    high_mags = [p['magnitude'] for p in curve if p['freq'] > high_freq_limit]
    
    avg_low = np.mean(low_mags) if low_mags else 0
    avg_high = np.mean(high_mags) if high_mags else 0
    
    if avg_low > avg_high * 1.5:
        return 'dark'
    elif avg_high > avg_low * 1.5:
        return 'bright'
    else:
        return 'balanced'

def apply_tonal_preset(y, sr, preset):
    y_processed = y.copy()
    
    # Analyze the tonal character of the input audio
    if y.ndim > 1:
        y_mono = librosa.to_mono(y)
    else:
        y_mono = y
    character = analyze_tonal_character(y_mono, sr)

    presets = {
        'brighter': {
            'high_shelf': {'freq': 8000, 'gain': 3.0, 'q': 0.7},
            'presence': {'freq': 3000, 'gain': 2.0, 'q': 1.0}
        },
        'darker': {
            'high_shelf': {'freq': 6000, 'gain': -4.0, 'q': 0.7},
            'low_shelf': {'freq': 200, 'gain': 2.0, 'q': 0.7}
        },
        'vintage_warmth': {
            'low_mid': {'freq': 400, 'gain': 2.5, 'q': 1.2},
            'high_shelf': {'freq': 10000, 'gain': -2.0, 'q': 0.5}
        },
        'modern_punch': {
            'low_shelf': {'freq': 100, 'gain': 3.0, 'q': 0.7},
            'presence': {'freq': 5000, 'gain': 2.5, 'q': 1.0}
        }
    }
    
    if preset.lower() in presets:
        eq_settings = presets[preset.lower()]

        # Adapt the preset based on the character
        if preset.lower() == 'brighter' and character == 'dark':
            eq_settings['high_shelf']['gain'] *= 1.5
        elif preset.lower() == 'darker' and character == 'bright':
            eq_settings['high_shelf']['gain'] *= 1.5

        for channel_idx in range(y.shape[0]):
            y_channel = y[channel_idx]
            
            for eq_type, params in eq_settings.items():
                freq = params['freq']
                gain = params['gain']
                q = params['q']
                
                w0 = 2 * np.pi * freq / sr
                alpha = np.sin(w0) / (2 * q)
                A = 10 ** (gain / 40)
                
                if 'shelf' in eq_type:
                    if gain > 0:
                        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
                        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
                        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
                        a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
                        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
                        a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
                    else:
                        b0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
                        b1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
                        b2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
                        a0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
                        a1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
                        a2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
                else:
                    b0 = 1 + alpha * A
                    b1 = -2 * np.cos(w0)
                    b2 = 1 - alpha * A
                    a0 = 1 + alpha / A
                    a1 = -2 * np.cos(w0)
                    a2 = 1 - alpha / A
                
                b = [b0 / a0, b1 / a0, b2 / a0]
                a = [1, a1 / a0, a2 / a0]
                
                y_channel = signal.lfilter(b, a, y_channel)
            
            y_processed[channel_idx] = y_channel
    elif preset in custom_presets:
        eq_curve = custom_presets[preset]
        y_processed = apply_eq_curve(y, sr, eq_curve)
    
    return y_processed

def peak_normalize(y, target_dbfs=-0.1):
    peak = np.max(np.abs(y))
    if peak > 1e-10: # Avoid division by zero
        gain = (10.0 ** (target_dbfs / 20.0)) / peak
        y = y * gain
    return y

def normalize_loudness(y, sr, target_lufs=-23.0):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    
    gain_db = target_lufs - loudness
    gain_linear = 10.0 ** (gain_db / 20.0)
    
    return y * gain_linear

@app.route('/get-custom-presets', methods=['GET'])
def get_custom_presets():
    return jsonify(custom_presets)

@app.route('/save-custom-preset', methods=['POST'])
def save_custom_preset():
    global custom_presets
    data = request.json
    if not data or 'name' not in data or 'settings' not in data:
        return jsonify({'error': 'Missing preset name or settings'}), 400
    
    preset_name = data['name']
    settings = data['settings']
    
    custom_presets[preset_name] = settings
    
    with open(CUSTOM_PRESETS_FILE, 'w') as f:
        json.dump(custom_presets, f, indent=4)
        
    return jsonify({'status': 'success'})

@app.route('/apply-multiband-compressor', methods=['POST'])
def apply_multiband_compressor_route():
    try:
        data = request.json
        if not data or 'filename' not in data or 'settings' not in data:
            return jsonify({'error': 'Missing filename or settings'}), 400
        
        filename = secure_filename(data['filename'])
        settings = data['settings']
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        y, sr = librosa.load(filepath, sr=None, mono=False)
        
        if y.ndim == 1:
            y = np.stack([y, y])
            
        y_processed = apply_multiband_compressor(y, sr, settings)
        
        ceiling = data.get('ceiling', -0.1)
        y_processed = peak_normalize(y_processed, target_dbfs=ceiling)
        
        output_filename = f"{os.path.splitext(filename)[0]}_compressed.wav"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        
        sf.write(output_path, y_processed.T, sr)
        
        return jsonify({'processed_file': output_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def apply_multiband_compressor(y, sr, settings):
    # Crossover frequencies
    low_mid_crossover = settings.get('low_mid_crossover', 300)
    mid_high_crossover = settings.get('mid_high_crossover', 3000)

    # --- Create crossover filters ---
    # Low-pass for the low band
    b_low, a_low = signal.butter(2, low_mid_crossover / (sr / 2), btype='low')
    # High-pass for the high band
    b_high, a_high = signal.butter(2, mid_high_crossover / (sr / 2), btype='high')
    
    # Band-pass for the mid band
    b_mid_low, a_mid_low = signal.butter(2, low_mid_crossover / (sr / 2), btype='high')
    b_mid_high, a_mid_high = signal.butter(2, mid_high_crossover / (sr / 2), btype='low')

    y_processed = np.zeros_like(y)

    for channel_idx in range(y.shape[0]):
        y_channel = y[channel_idx]

        # --- Split into bands ---
        low_band = signal.lfilter(b_low, a_low, y_channel)
        
        mid_band_tmp = signal.lfilter(b_mid_low, a_mid_low, y_channel)
        mid_band = signal.lfilter(b_mid_high, a_mid_high, mid_band_tmp)
        
        high_band = signal.lfilter(b_high, a_high, y_channel)

        # --- Process each band ---
        bands = {'low': low_band, 'mid': mid_band, 'high': high_band}
        processed_bands = {}

        for band_name, band_signal in bands.items():
            band_settings = settings.get(band_name, {})
            threshold = band_settings.get('threshold', 0.0)
            ratio = band_settings.get('ratio', 1.0)
            attack = band_settings.get('attack', 0.01)
            release = band_settings.get('release', 0.1)

            # Simple compressor logic
            if ratio > 1:
                alpha_attack = np.exp(-1.0 / (sr * attack))
                alpha_release = np.exp(-1.0 / (sr * release))

                envelope = np.zeros_like(band_signal)
                gain = np.ones_like(band_signal)

                for i in range(1, len(band_signal)):
                    # Peak detection
                    envelope[i] = max(abs(band_signal[i]), envelope[i-1] * alpha_release)
                
                # Gain computation
                for i in range(len(envelope)):
                    if envelope[i] > threshold:
                        gain[i] = (threshold / envelope[i]) ** (1 - 1/ratio)

                # Apply gain
                processed_bands[band_name] = band_signal * gain
            else:
                processed_bands[band_name] = band_signal
        
        # --- Combine bands ---
        y_processed[channel_idx] = processed_bands['low'] + processed_bands['mid'] + processed_bands['high']

    return y_processed

@app.route('/apply-stereo-imaging', methods=['POST'])
def apply_stereo_imaging_route():
    try:
        data = request.json
        if not data or 'filename' not in data or 'width' not in data:
            return jsonify({'error': 'Missing filename or width'}), 400
        
        filename = secure_filename(data['filename'])
        width = data['width']
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        y, sr = librosa.load(filepath, sr=None, mono=False)
        
        if y.ndim == 1:
            return jsonify({'error': 'Stereo imaging can only be applied to stereo files'}), 400
            
        y_processed = apply_stereo_imaging(y, sr, width)
        
        ceiling = data.get('ceiling', -0.1)
        y_processed = peak_normalize(y_processed, target_dbfs=ceiling)
        
        output_filename = f"{os.path.splitext(filename)[0]}_stereo.wav"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        
        sf.write(output_path, y_processed.T, sr)
        
        return jsonify({'processed_file': output_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def apply_stereo_imaging(y, sr, width):
    if y.ndim < 2:
        return y

    left = y[0]
    right = y[1]

    mid = (left + right) / 2
    side = (left - right) / 2

    # Adjust the width of the side channel
    side = side * width

    # Convert back to left/right
    new_left = mid + side
    new_right = mid - side

    return np.stack([new_left, new_right])

@app.route('/match-reference', methods=['POST'])
def match_reference():
    try:
        if 'reference' not in request.files or 'target' not in request.form:
            return jsonify({'error': 'Missing reference or target file'}), 400
        
        reference_file = request.files['reference']
        target_filename = secure_filename(request.form['target'])
        
        if not reference_file.filename:
            return jsonify({'error': 'Invalid reference file'}), 400
        ref_filename = secure_filename(reference_file.filename)
        ref_path = os.path.join(app.config['UPLOAD_FOLDER'], 'ref_' + ref_filename)
        reference_file.save(ref_path)
        
        target_path = os.path.join(app.config['UPLOAD_FOLDER'], target_filename)
        
        y_ref, sr_ref = librosa.load(ref_path, sr=None, mono=True)
        y_target, sr_target = librosa.load(target_path, sr=None, mono=False)
        
        if sr_ref != sr_target:
            y_ref = librosa.resample(y_ref, orig_sr=sr_ref, target_sr=sr_target)
            sr_ref = sr_target
        
        if y_target.ndim == 1:
            y_target_mono = y_target
            y_target = np.stack([y_target, y_target])
        else:
            y_target_mono = librosa.to_mono(y_target)
        
        y_ref = normalize_loudness(y_ref, sr_ref)
        y_target_mono = normalize_loudness(y_target_mono, sr_target)
        
        ref_curve = extract_eq_curve(y_ref, sr_ref)
        target_curve = extract_eq_curve(y_target_mono, sr_target)
        
        corrective_curve = compute_corrective_eq(ref_curve, target_curve)
        corrective_curve = smooth_curve(corrective_curve)
        
        y_matched = apply_eq_curve(y_target, sr_target, corrective_curve)
        
        ceiling = float(request.form.get('ceiling', -0.1))
        y_matched = peak_normalize(y_matched, target_dbfs=ceiling)
        
        output_filename = f"{os.path.splitext(target_filename)[0]}_matched.wav"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        
        sf.write(output_path, y_matched.T, sr_target)
        
        os.remove(ref_path)
        
        return jsonify({'processed_file': output_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def extract_eq_curve(y, sr):
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    
    freq_bins = np.logspace(np.log10(20), np.log10(20000), 100)
    eq_curve = []
    
    for i in range(len(freq_bins) - 1):
        mask = (freqs >= freq_bins[i]) & (freqs < freq_bins[i + 1])
        if np.any(mask):
            magnitude = np.mean(S[mask])
            eq_curve.append({
                'freq': float(freq_bins[i]),
                'magnitude': float(magnitude)
            })
    
    return eq_curve

def smooth_curve(curve, window_size=5):
    smoothed_curve = []
    gains = [p['gain_db'] for p in curve]
    
    # Pad the gains to handle edges
    padded_gains = np.pad(gains, (window_size // 2, window_size // 2), mode='edge')
    
    # Apply moving average
    smoothed_gains = np.convolve(padded_gains, np.ones(window_size) / window_size, mode='valid')
    
    for i in range(len(curve)):
        smoothed_curve.append({
            'freq': curve[i]['freq'],
            'gain_db': smoothed_gains[i]
        })
        
    return smoothed_curve

def compute_corrective_eq(ref_curve, target_curve):
    corrective_curve = []
    
    for i in range(min(len(ref_curve), len(target_curve))):
        ref_mag = ref_curve[i]['magnitude']
        target_mag = target_curve[i]['magnitude']
        
        gain_db = 20 * np.log10((ref_mag / (target_mag + 1e-10)) + 1e-10)
        gain_db = np.clip(gain_db, -12, 12)
        
        corrective_curve.append({
            'freq': ref_curve[i]['freq'],
            'gain_db': float(gain_db)
        })
    
    return corrective_curve

def apply_eq_curve(y, sr, eq_curve, num_taps=1025):
    y_processed = y.copy()
    
    if len(eq_curve) == 0:
        return y_processed

    # Create the frequency and gain arrays for firwin2
    freqs = [p['freq'] for p in eq_curve]
    gains_db = [p['gain_db'] for p in eq_curve]
    
    # Add 0 Hz and Nyquist to the arrays
    freqs = [0] + freqs + [sr / 2]
    gains_db = [gains_db[0]] + gains_db + [gains_db[-1]] # Pad with edge values

    # Convert gains from dB to linear
    gains = 10.0 ** (np.array(gains_db) / 20.0)

    # Normalize frequencies to [0, 1]
    freqs = np.array(freqs) / (sr / 2)

    # Design the FIR filter
    taps = signal.firwin2(num_taps, freqs, gains)

    # Apply the filter to each channel
    for channel_idx in range(y.shape[0]):
        y_processed[channel_idx] = signal.lfilter(taps, 1.0, y[channel_idx])

    return y_processed

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def serve_processed(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/clear', methods=['POST'])
def clear_files():
    try:
        for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER']]:
            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

load_custom_presets()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
