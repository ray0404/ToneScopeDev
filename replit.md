# ToneScope - Audio Analysis & Mastering Tool

## Project Overview
ToneScope is a professional web-based audio analysis and mastering tool built for music producers. It enables users to analyze album tracks for tonal balance, dynamic range, and stereo imaging, while also providing non-destructive audio processing capabilities for auditioning different mastering styles.

## Last Updated
October 15, 2025

## Recent Changes
- **October 15, 2025**: Unified Processing Workflow & Auto-Tune Features
  - Implemented unified /process-audio endpoint for sequential effect application
  - Fixed apply_eq_curve to use biquad peaking filters instead of FIR filters
  - Replaced individual "Apply" buttons with toggle switches for cleaner UX
  - Added global "Process Audio" button that builds effects array from active toggles
  - Integrated Auto-Tune Settings button for multiband compressor optimization
  - Effects now applied in optimal order: tonal → compression → stereo imaging → normalization
  
- **October 14, 2025**: Production deployment fixes
  - Added Flask-CORS for cross-origin request support in production
  - Enhanced error handling for file uploads with detailed logging
  - Configured Gunicorn with 300s timeout for large file processing
  - Added 2 workers for improved concurrent request handling
  - Fixed critical bugs in reference matching, frequency spectrum, and stereo correlation
  
- **October 14, 2025**: Initial project setup complete
  - Created Flask backend with audio analysis capabilities
  - Implemented Module 1: Analyzer (Tonal Balance, Dynamic Range, Stereo Image)
  - Implemented Module 2: Tonal Palette (Presets and Reference Matching)
  - Built responsive frontend with Chart.js and Wavesurfer.js
  - Set up file upload system with drag-and-drop support

## Tech Stack

### Backend
- **Python 3.11** with Flask framework
- **Flask-CORS**: Cross-origin request support for production deployment
- **Gunicorn**: Production WSGI server with worker processes
- **librosa**: Audio analysis and feature extraction
- **pyloudnorm**: LUFS loudness measurement
- **soundfile**: Audio file I/O
- **numpy**: Numerical computations
- **scipy**: Signal processing and filtering
- **pydub**: Audio format handling

### Frontend
- **HTML5/CSS3/JavaScript**
- **Chart.js**: Frequency spectrum visualization
- **Wavesurfer.js**: Waveform display and audio playback
- Modern gradient UI design optimized for music production

## Project Structure

```
/
├── main.py                 # Flask application with all routes and audio processing
├── templates/
│   └── index.html         # Main UI template
├── static/
│   ├── css/
│   │   └── style.css      # Responsive styling
│   └── js/
│       └── app.js         # Frontend JavaScript logic
├── uploads/               # Uploaded audio files (gitignored)
├── processed/             # Processed audio files (gitignored)
└── replit.md             # This file
```

## Features

### Module 1: Analyzer
1. **Tonal Balance**: Overlaid frequency spectrum visualization of all tracks showing album-wide tonal consistency
2. **Dynamic Range**: Table displaying LUFS, True Peak, and Crest Factor for each track
3. **Stereo Image**: Visual analysis of stereo width and L/R correlation for each track

### Module 2: Tonal Palette
1. **One-Click Presets**: Four mastering presets with toggle-based workflow
   - ✨ Brighter: High-frequency boost for airy, modern sound
   - 🌙 Darker: High-frequency cut with low-end emphasis
   - 🔥 Vintage Warmth: Mid-range warmth with gentle top-end rolloff
   - ⚡ Modern Punch: Enhanced low-end and presence
2. **Custom EQ**: Interactive drag-and-drop EQ curve with biquad filter implementation
3. **Reference Matching**: Upload reference track and apply its EQ curve to selected track
4. **Waveform Playback**: Visual waveform with play/pause controls for A/B testing

### Module 3: Advanced Processing
1. **Multi-band Compressor**: Dynamic compression across low, mid, and high frequency bands
   - Auto-Tune Settings: Automatically optimize compressor parameters based on audio analysis
2. **Stereo Imaging**: Adjust stereo width from mono to ultra-wide
3. **Peak Limiter**: Final ceiling control for optimal loudness
4. **Unified Processing**: Global "Process Audio" button applies all enabled effects sequentially

## How It Works

### Audio Analysis Pipeline
1. Files uploaded via drag-and-drop or file selector
2. Audio loaded using librosa with support for stereo/mono files
3. Frequency spectrum calculated using STFT with logarithmic binning
4. Dynamic range metrics computed using pyloudnorm and numpy
5. Stereo image analyzed using mid/side processing
6. Results visualized with Chart.js and rendered in responsive tables

### Unified Processing Workflow
1. User selects track and enables desired effects via toggle switches
2. Effects configuration:
   - **Tonal Preset/Custom EQ**: Radio buttons for preset selection or draggable EQ curve
   - **Multi-band Compressor**: Individual controls per band with auto-tune capability
   - **Stereo Imaging**: Width slider for stereo field adjustment
3. "Process Audio" button triggers sequential effect application:
   - Tonal balance (preset or custom EQ with biquad peaking filters)
   - Multi-band compression (dynamic processing per frequency band)
   - Stereo imaging (mid/side manipulation)
   - Peak normalization (final ceiling limiter)
4. Processed files saved as WAV with descriptive naming
5. Waveform visualization updates for immediate playback and A/B comparison

## API Endpoints

- `GET /`: Main application page
- `POST /upload`: Upload audio files
- `POST /analyze`: Analyze all uploaded tracks
- `POST /apply-preset`: Apply tonal preset to selected track
- `POST /match-reference`: Apply reference track EQ to target
- `POST /clear`: Clear all uploaded and processed files
- `GET /uploads/<filename>`: Serve uploaded files
- `GET /processed/<filename>`: Serve processed files

## Supported Audio Formats
- WAV, MP3, FLAC, AIFF, OGG, M4A

## Environment Variables
- `SESSION_SECRET`: Flask session secret (automatically configured)

## User Preferences
- None specified yet

## Architecture Notes
- Non-destructive processing: Original files remain untouched
- Biquad IIR filters for precise EQ control
- Logarithmic frequency display for musical perception
- Session-based file management with cleanup endpoint
- Responsive design for various screen sizes
- Real-time waveform visualization with Wavesurfer.js
- CORS enabled for production deployment compatibility
- Gunicorn with 300s timeout and 2 workers for scalability
- Comprehensive error handling and logging for debugging
