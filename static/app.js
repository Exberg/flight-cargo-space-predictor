// Global configuration
const API_BASE = 'http://localhost:5001/api';
let currentData = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadFileList();
    setupFileUpload();
});

// 1. DATA SOURCE
async function loadFileList() {
    const listDiv = document.getElementById('fileList');
    const container = document.getElementById('fileListContainer');
    
    try {
        const response = await fetch(`${API_BASE}/data`);
        const data = await response.json();
        
        if (data.success && data.files.length > 0) {
            container.style.display = 'block';
            let html = '';
            data.files.forEach(file => {
                const size = (file.size / 1024).toFixed(1) + ' KB';
                const date = new Date(file.modified).toLocaleDateString();
                
                html += `
                    <div class="file-item" onclick="selectDataFile('${file.name}', this)">
                        <div class="file-info">
                            <span class="file-name">${file.name}</span>
                            <span class="file-meta">${size} • ${date}</span>
                        </div>
                    </div>
                `;
            });
            listDiv.innerHTML = html;
        }
    } catch (error) {
        console.error('Error loading files:', error);
    }
}

function setupFileUpload() {
    const fileInput = document.getElementById('fileInput');
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (file) {
            const formData = new FormData();
            formData.append('file', file);
            
            showStatus('Uploading...', 'normal');
            try {
                const response = await fetch(`${API_BASE}/upload`, {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                if (data.success) {
                    handleDataLoaded(data.info);
                    showStatus('File uploaded successfully!', 'success');
                } else {
                    showStatus(data.error || 'Upload failed', 'error');
                }
            } catch (error) {
                showStatus('Error uploading file', 'error');
            }
        }
    });
}

async function generateSampleData() {
    showStatus('Generating random data...', 'normal');
    try {
        const response = await fetch(`${API_BASE}/generate-sample`, {
            method: 'POST'
        });
        const result = await response.json();
        
        if (result.success) {
            showStatus('Random data generated!', 'success');
            loadFileList(); // Refresh list
            // Auto-select the generated file would be nice, but for now just list it
        } else {
            showStatus(result.error, 'error');
        }
    } catch (error) {
        showStatus('Error generating data', 'error');
    }
}

async function selectDataFile(filename, element) {
    // UI Update
    document.querySelectorAll('.file-item').forEach(el => el.classList.remove('selected'));
    element.classList.add('selected');
    
    showStatus(`Loading ${filename}...`, 'normal');
    
    try {
        const response = await fetch(`${API_BASE}/data/select`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename })
        });
        const data = await response.json();
        
        if (data.success) {
            handleDataLoaded(data.info);
            showStatus(`Loaded ${filename}`, 'success');
        } else {
            showStatus(data.error, 'error');
        }
    } catch (error) {
        showStatus('Error loading file', 'error');
    }
}

// 2. DATA PREVIEW
function handleDataLoaded(info) {
    currentData = info;
    
    // Show sections
    document.getElementById('dataPreviewSection').style.display = 'block';
    document.getElementById('analysisControlSection').style.display = 'block';
    
    // Hide results if visible (new data requires new analysis)
    document.getElementById('resultsSection').style.display = 'none';
    
    // Render Table
    const tableDiv = document.getElementById('dataTable');
    if (info.sample && info.sample.length > 0) {
        let tableHTML = '<table><thead><tr>';
        info.columns.forEach(col => tableHTML += `<th>${col}</th>`);
        tableHTML += '</tr></thead><tbody>';
        
        info.sample.forEach(row => {
            tableHTML += '<tr>';
            info.columns.forEach(col => tableHTML += `<td>${row[col]}</td>`);
            tableHTML += '</tr>';
        });
        tableHTML += '</tbody></table>';
        tableDiv.innerHTML = tableHTML;
    }
    
    // Stats
    document.getElementById('dataStats').innerHTML = 
        `<span><strong>Rows:</strong> ${info.rows}</span>
         <span><strong>Columns:</strong> ${info.columns.length}</span>`;
         
    // Scroll to preview
    document.getElementById('dataPreviewSection').scrollIntoView({ behavior: 'smooth' });
}

// 3. ANALYSIS
async function runAnalysis() {
    const btn = document.getElementById('analyzeBtn');
    const progress = document.getElementById('trainingProgress');
    
    btn.disabled = true;
    btn.classList.add('disabled');
    progress.style.display = 'block';
    
    try {
        // 1. Train Models
        const trainResponse = await fetch(`${API_BASE}/train`, { method: 'POST' });
        const trainResult = await trainResponse.json();
        
        if (!trainResult.success) throw new Error(trainResult.error);
        
        // 2. Get Insights
        const insightsResponse = await fetch(`${API_BASE}/insights`);
        const insightsResult = await insightsResponse.json();
        
        if (!insightsResult.success) throw new Error(insightsResult.error);
        
        displayResults(insightsResult.insights);
        
    } catch (error) {
        showStatus('Analysis failed: ' + error.message, 'error');
    } finally {
        btn.disabled = false;
        btn.classList.remove('disabled');
        progress.style.display = 'none';
    }
}

// 4. RESULTS
function displayResults(insights) {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'grid';
    
    // A. Feature Weightage (Bar Chart)
    const importance = insights.feature_importance;
    const features = importance.map(i => i.feature);
    const values = importance.map(i => i.importance);
    
    const trace = {
        x: values,
        y: features,
        type: 'bar',
        orientation: 'h',
        marker: { color: '#4f46e5', line: { width: 0 } }
    };
    
    const layout = {
        margin: { l: 150, r: 20, t: 20, b: 40 },
        xaxis: { title: 'Relative Importance' },
        font: { family: 'Inter, sans-serif' },
        height: 300
    };
    
    Plotly.newPlot('featureImportanceChart', [trace], layout, {displayModeBar: false});
    
    // B. Insights (Sentences)
    const rulesList = document.getElementById('insightsList');
    rulesList.innerHTML = '';
    
    // Convert decision tree rules to human sentences
    insights.rules.slice(0, 5).forEach(rule => {
        const item = document.createElement('div');
        item.className = 'insight-item';
        
        // Format conditions
        const conditions = rule.conditions.map(c => {
            return c.replace('<=', ' is under ').replace('>', ' is over ');
        }).join(' AND ');
        
        item.innerHTML = `<strong>When</strong> ${conditions}, <strong>predicted luggage size</strong> is approx <strong>${rule.predicted_value.toFixed(1)}</strong>. <br><small class="text-gray-500">Applies to ${(rule.percentage).toFixed(1)}% of cases.</small>`;
        rulesList.appendChild(item);
    });
    
    // C. Interactions
    const interactionsList = document.getElementById('featureInteractions');
    interactionsList.innerHTML = '';
    
    insights.feature_interactions.slice(0, 5).forEach(interaction => {
        const item = document.createElement('div');
        item.className = 'interaction-item';
        item.innerHTML = `
            <span>${interaction.feature1} + ${interaction.feature2}</span>
            <span class="interaction-strength">High Impact</span>
        `;
        interactionsList.appendChild(item);
    });
    
    // D. Conditional Probabilities
    const probsContainer = document.getElementById('conditionalProbs');
    probsContainer.innerHTML = '';
    
    insights.conditional_probabilities.slice(0, 4).forEach(prob => {
        const card = document.createElement('div');
        card.className = 'prob-card';
        
        // Conditions text
        const conditions = Object.entries(prob.conditions)
            .map(([feat, [min, max]]) => `${feat} between ${min.toFixed(0)} and ${max.toFixed(0)}`)
            .join(' & ');
            
        let html = `<div class="prob-conditions">If ${conditions}:</div>`;
        
        // Bars
        prob.probabilities.forEach(p => {
            if (p.probability > 5) { // Only show significant probs
                html += `
                    <div class="prob-bar-container">
                        <div class="prob-label">
                            <span>Luggage Size: ${p.range}</span>
                            <span>${p.probability.toFixed(1)}%</span>
                        </div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width: ${p.probability}%"></div>
                        </div>
                    </div>
                `;
            }
        });
        
        card.innerHTML = html;
        probsContainer.appendChild(card);
    });
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function showStatus(message, type) {
    const statusDiv = document.getElementById('statusMessage');
    statusDiv.textContent = message;
    statusDiv.className = 'status-message status-' + type;
    
    if (type === 'success' || type === 'error') {
        setTimeout(() => {
            statusDiv.textContent = '';
        }, 5000);
    }
}
