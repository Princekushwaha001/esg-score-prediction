// ESG Score Prediction - Frontend JavaScript

const API_BASE = '/api';

// DOM Elements
const form = document.getElementById('predictionForm');
const predictBtn = document.getElementById('predictBtn');
const loadExampleBtn = document.getElementById('loadExampleBtn');
const resultsCard = document.getElementById('resultsCard');
const loadingOverlay = document.getElementById('loadingOverlay');

// Form inputs
const modelSelect = document.getElementById('model');
const co2Input = document.getElementById('co2');
const energyInput = document.getElementById('energy');
const diversityInput = document.getElementById('diversity');
const governanceInput = document.getElementById('governance');

// Result elements
const scoreValue = document.getElementById('scoreValue');
const scoreEmoji = document.getElementById('scoreEmoji');
const ratingBadge = document.getElementById('ratingBadge');
const ratingText = document.getElementById('ratingText');
const ratingStars = document.getElementById('ratingStars');
const ratingDescription = document.getElementById('ratingDescription');

// Component scores
const envScore = document.getElementById('envScore');
const envProgress = document.getElementById('envProgress');
const socialScore = document.getElementById('socialScore');
const socialProgress = document.getElementById('socialProgress');
const govScore = document.getElementById('govScore');
const govProgress = document.getElementById('govProgress');

// Summary
const summaryCO2 = document.getElementById('summaryCO2');
const summaryEnergy = document.getElementById('summaryEnergy');
const summaryDiversity = document.getElementById('summaryDiversity');
const summaryGovernance = document.getElementById('summaryGovernance');

// Example scenarios
const examples = [
    {
        name: 'Green Tech Startup',
        co2: 15000,
        energy: 8000,
        diversity: 90,
        governance: 10
    },
    {
        name: 'Average Manufacturing',
        co2: 50000,
        energy: 30000,
        diversity: 65,
        governance: 7
    },
    {
        name: 'Heavy Industrial',
        co2: 200000,
        energy: 100000,
        diversity: 30,
        governance: 3
    }
];

let currentExampleIndex = 0;

// Event Listeners
form.addEventListener('submit', handleSubmit);
loadExampleBtn.addEventListener('click', loadExample);

// Handle form submission
async function handleSubmit(e) {
    e.preventDefault();

    // Get form values
    const data = {
        model: modelSelect.value,
        co2: parseFloat(co2Input.value),
        energy: parseFloat(energyInput.value),
        diversity: parseFloat(diversityInput.value),
        governance: parseInt(governanceInput.value)
    };

    // Validate
    if (!validateInputs(data)) {
        return;
    }

    // Show loading
    showLoading();


    // Replace the entire try-catch block with this:


    try {
        const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    });

        if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

        const result = await response.json();
        displayResults(result);
    } catch (error) {
        console.error('Prediction error:', error);
        alert('Error making prediction. Please check the console for details.');
    } finally {
        hideLoading();
    }
}

// Validate inputs
function validateInputs(data) {
    if (data.co2 < 0 || data.energy < 0) {
        showError('CO2 and Energy must be positive values');
        return false;
    }

    if (data.diversity < 0 || data.diversity > 100) {
        showError('Diversity Index must be between 0-100');
        return false;
    }

    if (data.governance < 1 || data.governance > 10) {
        showError('Governance Rating must be between 1-10');
        return false;
    }

    return true;
}

// Display results
function displayResults(result) {
    const { prediction, components, inputs } = result;

    // Show results card
    resultsCard.style.display = 'block';
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Display score
    scoreValue.textContent = prediction.score.toFixed(2);
    scoreEmoji.textContent = prediction.emoji;

    // Display rating
    ratingText.textContent = prediction.rating;
    ratingDescription.textContent = prediction.description;
    ratingBadge.className = `rating-badge ${prediction.rating.toLowerCase().replace(' ', '-')}`;

    // Display stars
    const stars = '⭐'.repeat(prediction.stars);
    ratingStars.textContent = stars;

    // Display component scores
    displayComponent(envScore, envProgress, components.environmental);
    displayComponent(socialScore, socialProgress, components.social);
    displayComponent(govScore, govProgress, components.governance);

    // Display input summary
    summaryCO2.textContent = inputs.co2.toLocaleString() + ' tons';
    summaryEnergy.textContent = inputs.energy.toLocaleString() + ' MWh';
    summaryDiversity.textContent = inputs.diversity.toFixed(1);
    summaryGovernance.textContent = inputs.governance + '/10';
}

// Display component score with animation
function displayComponent(valueEl, progressEl, score) {
    valueEl.textContent = score.toFixed(2);

    // Animate progress bar
    setTimeout(() => {
        progressEl.style.width = `${score}%`;

        // Color based on score
        if (score >= 70) {
            progressEl.style.background = '#10b981';
        } else if (score >= 50) {
            progressEl.style.background = '#f59e0b';
        } else {
            progressEl.style.background = '#ef4444';
        }
    }, 100);
}

// Load example data
function loadExample() {
    const example = examples[currentExampleIndex];

    co2Input.value = example.co2;
    energyInput.value = example.energy;
    diversityInput.value = example.diversity;
    governanceInput.value = example.governance;

    // Show notification
    showNotification(`Loaded: ${example.name}`);

    // Cycle through examples
    currentExampleIndex = (currentExampleIndex + 1) % examples.length;
}

// Show loading overlay
function showLoading() {
    loadingOverlay.style.display = 'flex';
}

// Hide loading overlay
function hideLoading() {
    loadingOverlay.style.display = 'none';
}

// Show error message
function showError(message) {
    alert('Error: ' + message);
}

// Show notification
function showNotification(message) {
    // Simple notification (can be enhanced)
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('ESG Prediction App Initialized');
});