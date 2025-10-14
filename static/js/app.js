let uploadedFiles = [];
let analysisData = [];
let currentTrack = null;
let wavesurfer = null;
let referenceFile = null;
let frequencyChart = null;

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const analysisResults = document.getElementById('analysisResults');
const trackSelector = document.getElementById('trackSelector');
const referenceInput = document.getElementById('referenceInput');
const matchBtn = document.getElementById('matchBtn');
const referenceInfo = document.getElementById('referenceInfo');

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files);
    handleFiles(files);
});

uploadArea.addEventListener('click', (e) => {
    if (e.target.tagName !== 'BUTTON') {
        fileInput.click();
    }
});

fileInput.addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    handleFiles(files);
});

analyzeBtn.addEventListener('click', analyzeFiles);
clearBtn.addEventListener('click', clearAllFiles);

document.querySelectorAll('.btn-preset').forEach(btn => {
    btn.addEventListener('click', () => {
        const preset = btn.dataset.preset;
        applyPreset(preset);
    });
});

trackSelector.addEventListener('change', (e) => {
    currentTrack = e.target.value;
    if (currentTrack) {
        loadWaveform(currentTrack);
    }
});

referenceInput.addEventListener('change', (e) => {
    referenceFile = e.target.files[0];
    if (referenceFile) {
        referenceInfo.innerHTML = `<strong>Reference Track:</strong> ${referenceFile.name}`;
        matchBtn.disabled = !currentTrack;
    }
});

matchBtn.addEventListener('click', applyReferenceMatching);

async function handleFiles(files) {
    const formData = new FormData();
    files.forEach(file => {
        formData.append('files[]', file);
    });
    
    showLoading(true);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: 'Upload failed' }));
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.files && data.files.length > 0) {
            uploadedFiles = [...uploadedFiles, ...data.files];
            updateFileList();
            analyzeBtn.disabled = false;
        } else if (data.error) {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert('Error uploading files: ' + error.message);
    } finally {
        showLoading(false);
    }
}

function updateFileList() {
    fileList.innerHTML = '';
    uploadedFiles.forEach(filename => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <span>🎵 ${filename}</span>
            <span>${getFileSize(filename)}</span>
        `;
        fileList.appendChild(fileItem);
    });
}

function getFileSize(filename) {
    return 'Audio File';
}

async function analyzeFiles() {
    if (uploadedFiles.length === 0) return;
    
    showLoading(true);
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.analysis) {
            analysisData = data.analysis;
            displayAnalysis();
            populateTrackSelector();
            analysisResults.style.display = 'block';
        }
    } catch (error) {
        alert('Error analyzing files: ' + error.message);
    } finally {
        showLoading(false);
    }
}

function displayAnalysis() {
    displayFrequencyChart();
    displayDynamicRangeTable();
    displayStereoImage();
}

function displayFrequencyChart() {
    const ctx = document.getElementById('frequencyChart').getContext('2d');
    
    if (frequencyChart) {
        frequencyChart.destroy();
    }
    
    const datasets = analysisData.map((track, index) => {
        const hue = (index * 360 / analysisData.length) % 360;
        return {
            label: track.filename,
            data: track.frequency_spectrum.frequencies.map((freq, i) => ({
                x: freq,
                y: track.frequency_spectrum.magnitudes[i]
            })),
            borderColor: `hsl(${hue}, 70%, 50%)`,
            backgroundColor: `hsla(${hue}, 70%, 50%, 0.1)`,
            tension: 0.4,
            borderWidth: 2,
            pointRadius: 0
        };
    });
    
    frequencyChart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Frequency (Hz)',
                        font: { size: 14 }
                    },
                    min: 20,
                    max: 20000
                },
                y: {
                    title: {
                        display: true,
                        text: 'Magnitude (dB)',
                        font: { size: 14 }
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
    
    document.getElementById('frequencyChart').style.height = '400px';
}

function displayDynamicRangeTable() {
    const tbody = document.querySelector('#dynamicRangeTable tbody');
    tbody.innerHTML = '';
    
    analysisData.forEach(track => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${track.filename}</strong></td>
            <td>${track.dynamic_range.lufs.toFixed(2)} LUFS</td>
            <td>${track.dynamic_range.true_peak.toFixed(2)} dB</td>
            <td>${track.dynamic_range.crest_factor.toFixed(2)} dB</td>
        `;
        tbody.appendChild(row);
    });
}

function displayStereoImage() {
    const stereoGrid = document.getElementById('stereoGrid');
    stereoGrid.innerHTML = '';
    
    analysisData.forEach(track => {
        const card = document.createElement('div');
        card.className = 'stereo-card';
        
        const widthPercent = (track.stereo_image.stereo_width * 100).toFixed(1);
        const correlation = (track.stereo_image.correlation * 100).toFixed(1);
        
        card.innerHTML = `
            <h4>${track.filename}</h4>
            <div>
                <div class="stereo-value">${widthPercent}%</div>
                <div class="stereo-label">Stereo Width</div>
            </div>
            <div style="margin-top: 15px;">
                <div class="stereo-value">${correlation}%</div>
                <div class="stereo-label">L/R Correlation</div>
            </div>
        `;
        stereoGrid.appendChild(card);
    });
}

function populateTrackSelector() {
    trackSelector.innerHTML = '<option value="">Choose a track...</option>';
    uploadedFiles.forEach(filename => {
        const option = document.createElement('option');
        option.value = filename;
        option.textContent = filename;
        trackSelector.appendChild(option);
    });
}

async function applyPreset(preset) {
    if (!currentTrack) {
        alert('Please select a track first');
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch('/apply-preset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filename: currentTrack,
                preset: preset
            })
        });
        
        const data = await response.json();
        
        if (data.processed_file) {
            displayProcessedFile(data.processed_file, `${preset} preset applied`);
        } else if (data.error) {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error applying preset: ' + error.message);
    } finally {
        showLoading(false);
    }
}

async function applyReferenceMatching() {
    if (!currentTrack || !referenceFile) {
        alert('Please select a track and upload a reference file');
        return;
    }
    
    showLoading(true);
    
    try {
        const formData = new FormData();
        formData.append('reference', referenceFile);
        formData.append('target', currentTrack);
        
        const response = await fetch('/match-reference', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.processed_file) {
            displayProcessedFile(data.processed_file, 'Reference EQ applied');
        } else if (data.error) {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error applying reference matching: ' + error.message);
    } finally {
        showLoading(false);
    }
}

function displayProcessedFile(filename, description) {
    const processedFiles = document.getElementById('processedFiles');
    
    const item = document.createElement('div');
    item.className = 'processed-item';
    item.innerHTML = `
        <div>
            <strong>✅ ${description}</strong>
            <div>${filename}</div>
        </div>
        <a href="/processed/${filename}" download class="btn btn-primary">Download</a>
    `;
    processedFiles.appendChild(item);
    
    loadWaveform('/processed/' + filename);
}

function loadWaveform(audioUrl) {
    const waveformPanel = document.getElementById('waveformPanel');
    waveformPanel.style.display = 'block';
    
    if (wavesurfer) {
        wavesurfer.destroy();
    }
    
    wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: '#667eea',
        progressColor: '#764ba2',
        cursorColor: '#333',
        barWidth: 2,
        barRadius: 3,
        cursorWidth: 1,
        height: 150,
        barGap: 2
    });
    
    const url = audioUrl.startsWith('/') ? audioUrl : `/uploads/${audioUrl}`;
    wavesurfer.load(url);
    
    wavesurfer.on('ready', () => {
        const duration = wavesurfer.getDuration();
        document.getElementById('duration').textContent = formatTime(duration);
    });
    
    wavesurfer.on('audioprocess', () => {
        const currentTime = wavesurfer.getCurrentTime();
        document.getElementById('currentTime').textContent = formatTime(currentTime);
    });
    
    document.getElementById('playBtn').onclick = () => wavesurfer.play();
    document.getElementById('pauseBtn').onclick = () => wavesurfer.pause();
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

async function clearAllFiles() {
    if (!confirm('Are you sure you want to clear all files?')) {
        return;
    }
    
    showLoading(true);
    
    try {
        await fetch('/clear', { method: 'POST' });
        
        uploadedFiles = [];
        analysisData = [];
        currentTrack = null;
        referenceFile = null;
        
        fileList.innerHTML = '';
        analysisResults.style.display = 'none';
        document.getElementById('processedFiles').innerHTML = '';
        document.getElementById('waveformPanel').style.display = 'none';
        referenceInfo.innerHTML = '';
        
        analyzeBtn.disabled = true;
        matchBtn.disabled = true;
        
        if (wavesurfer) {
            wavesurfer.destroy();
            wavesurfer = null;
        }
        
        if (frequencyChart) {
            frequencyChart.destroy();
            frequencyChart = null;
        }
    } catch (error) {
        alert('Error clearing files: ' + error.message);
    } finally {
        showLoading(false);
    }
}

function showLoading(show) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
}
