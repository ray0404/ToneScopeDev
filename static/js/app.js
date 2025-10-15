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

// Process Audio button and toggle switches
const processAudioBtn = document.getElementById('processAudioBtn');
const tonalPresetToggle = document.getElementById('tonalPresetToggle');
const customEqToggle = document.getElementById('customEqToggle');
const multibandCompressorToggle = document.getElementById('multibandCompressorToggle');
const stereoImagingToggle = document.getElementById('stereoImagingToggle');
const autoTuneCompressorBtn = document.getElementById('autoTuneCompressorBtn');

// Enable Process Audio button when track is selected
trackSelector.addEventListener('change', () => {
    processAudioBtn.disabled = !currentTrack;
});

// Process Audio button click handler
processAudioBtn.addEventListener('click', processAudioWithEffects);

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
                preset: preset,
                ceiling: getLimiterCeiling()
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
        formData.append('ceiling', getLimiterCeiling());
        
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

let customEqChart = null;

const customPresetName = document.getElementById('customPresetName');
const saveCustomPresetBtn = document.getElementById('saveCustomPresetBtn');

saveCustomPresetBtn.addEventListener('click', saveCustomPreset);

function initializeCustomEqChart() {
    const ctx = document.getElementById('customEqChart').getContext('2d');
    
    const initialEqCurve = [
        { x: 20, y: 0 },
        { x: 100, y: 0 },
        { x: 500, y: 0 },
        { x: 2000, y: 0 },
        { x: 8000, y: 0 },
        { x: 20000, y: 0 }
    ];

    customEqChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Custom EQ',
                data: initialEqCurve,
                borderColor: '#764ba2',
                backgroundColor: 'rgba(118, 75, 162, 0.1)',
                tension: 0.4,
                pointRadius: 10,
                pointHoverRadius: 12,
                pointBackgroundColor: '#fff',
                pointBorderColor: '#764ba2'
            }]
        },
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
                        text: 'Gain (dB)',
                        font: { size: 14 }
                    },
                    min: -12,
                    max: 12
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    enabled: false
                },
                dragData: {
                    round: 1,
                    showTooltip: true,
                    onDragEnd: function(e, datasetIndex, index, value) {
                        // console.log(e, datasetIndex, index, value);
                    }
                }
            }
        }
    });
    document.getElementById('customEqChart').style.height = '300px';
}

// Unified Process Audio function - builds effects array from active toggles
async function processAudioWithEffects() {
    if (!currentTrack) {
        alert('Please select a track first');
        return;
    }

    const effects = [];

    // Check tonal preset toggle
    if (tonalPresetToggle.checked) {
        const selectedPreset = document.querySelector('input[name="preset"]:checked');
        if (selectedPreset) {
            effects.push({
                type: 'tonal_balance',
                settings: { preset: selectedPreset.value }
            });
        }
    }

    // Check custom EQ toggle
    if (customEqToggle.checked) {
        const eqCurve = customEqChart.data.datasets[0].data.map(p => ({ freq: p.x, gain_db: p.y }));
        effects.push({
            type: 'tonal_balance',
            settings: { preset: 'custom', eq_curve: eqCurve }
        });
    }

    // Check multiband compressor toggle
    if (multibandCompressorToggle.checked) {
        const settings = getMultibandCompressorSettings();
        effects.push({
            type: 'multiband_compressor',
            settings: settings
        });
    }

    // Check stereo imaging toggle
    if (stereoImagingToggle.checked) {
        const width = parseFloat(stereoWidthSlider.value);
        effects.push({
            type: 'stereo_imaging',
            settings: { width: width }
        });
    }

    if (effects.length === 0) {
        alert('Please enable at least one effect before processing');
        return;
    }

    showLoading(true);

    try {
        const response = await fetch('/process-audio', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filename: currentTrack,
                effects: effects,
                ceiling: getLimiterCeiling()
            })
        });

        const data = await response.json();

        if (data.processed_file) {
            displayProcessedFile(data.processed_file, 'Effects chain applied');
        } else if (data.error) {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error processing audio: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// Helper function to get multiband compressor settings
function getMultibandCompressorSettings() {
    const settings = {
        low: {
            threshold: parseFloat(document.getElementById('low-threshold').value),
            ratio: parseFloat(document.getElementById('low-ratio').value),
            attack: parseFloat(document.getElementById('low-attack').value) / 1000,
            release: parseFloat(document.getElementById('low-release').value) / 1000
        },
        mid: {
            threshold: parseFloat(document.getElementById('mid-threshold').value),
            ratio: parseFloat(document.getElementById('mid-ratio').value),
            attack: parseFloat(document.getElementById('mid-attack').value) / 1000,
            release: parseFloat(document.getElementById('mid-release').value) / 1000
        },
        high: {
            threshold: parseFloat(document.getElementById('high-threshold').value),
            ratio: parseFloat(document.getElementById('high-ratio').value),
            attack: parseFloat(document.getElementById('high-attack').value) / 1000,
            release: parseFloat(document.getElementById('high-release').value) / 1000
        }
    };

    // Convert threshold from dB to linear
    settings.low.threshold = 10 ** (settings.low.threshold / 20);
    settings.mid.threshold = 10 ** (settings.mid.threshold / 20);
    settings.high.threshold = 10 ** (settings.high.threshold / 20);

    return settings;
}

async function saveCustomPreset() {
    const presetName = customPresetName.value;
    if (!presetName) {
        alert('Please enter a name for the preset');
        return;
    }

    const eqCurve = customEqChart.data.datasets[0].data.map(p => ({ freq: p.x, gain_db: p.y }));

    try {
        const response = await fetch('/save-custom-preset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                name: presetName,
                settings: eqCurve
            })
        });

        const data = await response.json();

        if (data.status === 'success') {
            alert('Preset saved successfully');
            loadCustomPresets(); // Reload the presets
        } else if (data.error) {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error saving preset: ' + error.message);
    }
}

async function loadCustomPresets() {
    try {
        const response = await fetch('/get-custom-presets');
        const presets = await response.json();

        const presetButtons = document.querySelector('.preset-buttons');
        // Remove existing custom presets
        presetButtons.querySelectorAll('.btn-custom-preset').forEach(btn => btn.remove());

        for (const presetName in presets) {
            const btn = document.createElement('button');
            btn.className = 'btn btn-preset btn-custom-preset';
            btn.dataset.preset = presetName;
            btn.textContent = presetName;
            btn.addEventListener('click', () => {
                applyPreset(presetName);
            });
            presetButtons.appendChild(btn);
        }
    } catch (error)
    {
        console.error('Error loading custom presets:', error);
    }
}

// Initialize
loadCustomPresets();

// Auto-Tune Settings button for multiband compressor
autoTuneCompressorBtn.addEventListener('click', autoTuneCompressor);

// Update slider values
document.querySelectorAll('.multiband-compressor-controls input[type="range"]').forEach(slider => {
    const valueSpan = document.getElementById(`${slider.id}-value`);
    if (valueSpan) {
        slider.addEventListener('input', () => {
            if (slider.id.includes('threshold')) {
                valueSpan.textContent = `${slider.value} dB`;
            } else if (slider.id.includes('ratio')) {
                valueSpan.textContent = `${slider.value}:1`;
            } else {
                valueSpan.textContent = `${slider.value} ms`;
            }
        });
    }
});

async function applyMultibandCompressor() {
    if (!currentTrack) {
        alert('Please select a track first');
        return;
    }

    const settings = {
        low: {
            threshold: parseFloat(document.getElementById('low-threshold').value),
            ratio: parseFloat(document.getElementById('low-ratio').value),
            attack: parseFloat(document.getElementById('low-attack').value) / 1000,
            release: parseFloat(document.getElementById('low-release').value) / 1000
        },
        mid: {
            threshold: parseFloat(document.getElementById('mid-threshold').value),
            ratio: parseFloat(document.getElementById('mid-ratio').value),
            attack: parseFloat(document.getElementById('mid-attack').value) / 1000,
            release: parseFloat(document.getElementById('mid-release').value) / 1000
        },
        high: {
            threshold: parseFloat(document.getElementById('high-threshold').value),
            ratio: parseFloat(document.getElementById('high-ratio').value),
            attack: parseFloat(document.getElementById('high-attack').value) / 1000,
            release: parseFloat(document.getElementById('high-release').value) / 1000
        }
    };

    // Convert threshold from dB to linear
    settings.low.threshold = 10 ** (settings.low.threshold / 20);
    settings.mid.threshold = 10 ** (settings.mid.threshold / 20);
    settings.high.threshold = 10 ** (settings.high.threshold / 20);

    showLoading(true);

    try {
        const response = await fetch('/apply-multiband-compressor', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filename: currentTrack,
                settings: settings,
                ceiling: getLimiterCeiling()
            })
        });

        const data = await response.json();

        if (data.processed_file) {
            displayProcessedFile(data.processed_file, 'Multi-band compression applied');
        } else if (data.error) {
            alert('Error: ' + data.error);
        }
    } finally {
        showLoading(false);
    }
}

const stereoWidthSlider = document.getElementById('stereo-width');
const stereoWidthValue = document.getElementById('stereo-width-value');

stereoWidthSlider.addEventListener('input', () => {
    stereoWidthValue.textContent = parseFloat(stereoWidthSlider.value).toFixed(1);
});

async function applyStereoImaging() {
    if (!currentTrack) {
        alert('Please select a track first');
        return;
    }

    const width = parseFloat(stereoWidthSlider.value);

    showLoading(true);

    try {
        const response = await fetch('/apply-stereo-imaging', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filename: currentTrack,
                width: width,
                ceiling: getLimiterCeiling()
            })
        });

        const data = await response.json();

        if (data.processed_file) {
            displayProcessedFile(data.processed_file, 'Stereo imaging applied');
        } else if (data.error) {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error applying stereo imaging: ' + error.message);
    } finally {
        showLoading(false);
    }
}

const limiterCeilingSlider = document.getElementById('limiter-ceiling');
const limiterCeilingValue = document.getElementById('limiter-ceiling-value');

limiterCeilingSlider.addEventListener('input', () => {
    limiterCeilingValue.textContent = `${parseFloat(limiterCeilingSlider.value).toFixed(1)} dBFS`;
});

// Helper function to get limiter ceiling
function getLimiterCeiling() {
    return parseFloat(limiterCeilingSlider.value);
}

// Auto-Tune Compressor Settings
async function autoTuneCompressor() {
    if (!currentTrack) {
        alert("Please select a track first");
        return;
    }

    showLoading(true);

    try {
        const response = await fetch("/auto-optimize-compressor", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                filename: currentTrack
            })
        });

        const data = await response.json();

        if (data.error) {
            alert("Error: " + data.error);
        } else {
            // Update low band settings
            document.getElementById("low-threshold").value = data.low.threshold;
            document.getElementById("low-threshold-value").textContent = `${data.low.threshold.toFixed(1)} dB`;
            document.getElementById("low-ratio").value = data.low.ratio;
            document.getElementById("low-ratio-value").textContent = `${data.low.ratio.toFixed(1)}:1`;
            document.getElementById("low-attack").value = (data.low.attack * 1000).toFixed(0);
            document.getElementById("low-attack-value").textContent = `${(data.low.attack * 1000).toFixed(0)} ms`;
            document.getElementById("low-release").value = (data.low.release * 1000).toFixed(0);
            document.getElementById("low-release-value").textContent = `${(data.low.release * 1000).toFixed(0)} ms`;

            // Update mid band settings
            document.getElementById("mid-threshold").value = data.mid.threshold;
            document.getElementById("mid-threshold-value").textContent = `${data.mid.threshold.toFixed(1)} dB`;
            document.getElementById("mid-ratio").value = data.mid.ratio;
            document.getElementById("mid-ratio-value").textContent = `${data.mid.ratio.toFixed(1)}:1`;
            document.getElementById("mid-attack").value = (data.mid.attack * 1000).toFixed(0);
            document.getElementById("mid-attack-value").textContent = `${(data.mid.attack * 1000).toFixed(0)} ms`;
            document.getElementById("mid-release").value = (data.mid.release * 1000).toFixed(0);
            document.getElementById("mid-release-value").textContent = `${(data.mid.release * 1000).toFixed(0)} ms`;

            // Update high band settings
            document.getElementById("high-threshold").value = data.high.threshold;
            document.getElementById("high-threshold-value").textContent = `${data.high.threshold.toFixed(1)} dB`;
            document.getElementById("high-ratio").value = data.high.ratio;
            document.getElementById("high-ratio-value").textContent = `${data.high.ratio.toFixed(1)}:1`;
            document.getElementById("high-attack").value = (data.high.attack * 1000).toFixed(0);
            document.getElementById("high-attack-value").textContent = `${(data.high.attack * 1000).toFixed(0)} ms`;
            document.getElementById("high-release").value = (data.high.release * 1000).toFixed(0);
            document.getElementById("high-release-value").textContent = `${(data.high.release * 1000).toFixed(0)} ms`;

            alert("Compressor settings auto-tuned successfully!");
        }
    } catch (error) {
        alert("Error auto-tuning compressor: " + error.message);
    } finally {
        showLoading(false);
    }
}

