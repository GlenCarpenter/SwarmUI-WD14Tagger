/**
 * wd14tagger.js
 * WD14 Tagger extension for SwarmUI.
 * Adds a "Generate Tags" button and gear settings button to the prompt area.
 * Analyzes the current generated image and inserts WD14 tags into the prompt box.
 */

'use strict';

/** Available tagger models. */
let wd14TaggerModels = [
    { id: 'SmilingWolf/wd-eva02-large-tagger-v3',    label: 'WD EVA02 Large v3 (default)' },
    { id: 'SmilingWolf/wd-vit-large-tagger-v3',      label: 'WD ViT Large v3' },
    { id: 'SmilingWolf/wd-vit-tagger-v3',            label: 'WD ViT v3' },
    { id: 'SmilingWolf/wd-swinv2-tagger-v3',         label: 'WD SwinV2 v3' },
    { id: 'SmilingWolf/wd-convnext-tagger-v3',       label: 'WD ConvNext v3' },
    { id: 'SmilingWolf/wd-v1-4-swinv2-tagger-v2',   label: 'WD SwinV2 v2' },
    { id: 'SmilingWolf/wd-v1-4-vit-tagger-v2',      label: 'WD ViT v2' },
    { id: 'SmilingWolf/wd-v1-4-convnext-tagger-v2', label: 'WD ConvNext v2' },
    { id: 'deepghs/pixai-tagger-v0.9-onnx',         label: 'PixAI Tagger v0.9' },
];

/**
 * Handles the "Generate Tags" button click.
 * Reads the current image, sends it to the server, and inserts tags into the prompt.
 */
async function handleWD14GenerateTags() {
    let currentImage = document.querySelector('#current_image img.current-image-img');
    if (!currentImage || !currentImage.src) {
        showError('WD14 Tagger: No image selected. Generate an image first.');
        return;
    }

    let button = document.getElementById('wd14tagger_generate_btn');
    let modelSelect = document.getElementById('wd14tagger_model_select');
    let btnLabel = button ? button.querySelector('.wd14tagger-btn-label') : null;

    if (button) {
        button.disabled = true;
    }
    if (btnLabel) {
        btnLabel.innerHTML = ' Generating<span class="wd14tagger-ellipsis"><span>.</span><span>.</span><span>.</span></span>';
    }

    try {
        let fetchResponse = await fetch(currentImage.src);
        let blob = await fetchResponse.blob();
        let base64Data = await new Promise((resolve) => {
            let reader = new FileReader();
            reader.onloadend = () => {
                resolve(reader.result.split(',')[1]);
            };
            reader.readAsDataURL(blob);
        });

        let selectedModel = modelSelect ? modelSelect.value : 'SmilingWolf/wd-eva02-large-tagger-v3';
        let filterInput = document.getElementById('wd14tagger_filter_input');
        let filterTags = filterInput ? filterInput.value : '';
        let thresholdInput = document.getElementById('wd14tagger_threshold_input');
        let threshold = thresholdInput ? (parseFloat(thresholdInput.value) / 100) : 0.35;

        let result = await new Promise((resolve, reject) => {
            genericRequest(
                'WD14TaggerGenerateTags',
                { imageBase64: base64Data, modelId: selectedModel, threshold: threshold, filterTags: filterTags },
                (data) => {
                    if (data.success) {
                        resolve(data);
                    }
                    else {
                        reject(new Error(data.error || 'Tagging failed.'));
                    }
                }
            );
        });

        let promptBox = document.getElementById('alt_prompt_textbox');
        if (promptBox) {
            if (result.tags) {
                let insertModeEl = document.querySelector('input[name="wd14tagger_insert_mode"]:checked');
                let insertMode = insertModeEl ? insertModeEl.value : 'replace';
                if (insertMode === 'prepend') {
                    let existing = promptBox.value;
                    promptBox.value = existing.trim() ? result.tags + ', ' + existing : result.tags;
                }
                else if (insertMode === 'append') {
                    let existing = promptBox.value;
                    promptBox.value = existing.trim() ? existing + ', ' + result.tags : result.tags;
                }
                else {
                    promptBox.value = result.tags;
                }
                triggerChangeFor(promptBox);
                promptBox.focus();
                promptBox.setSelectionRange(0, promptBox.value.length);
            }
            else {
                showError('WD14 Tagger: No tags found above the current threshold.');
            }
        }
    }
    catch (err) {
        console.error('WD14 Tagger error:', err);
        showError('WD14 Tagger: ' + err.message);
    }
    finally {
        if (button) {
            button.disabled = false;
        }
        if (btnLabel) {
            btnLabel.innerHTML = ' Generate Tags';
        }
    }
}

/**
 * Adds the "Generate Tags" button and gear settings button into the prompt region,
 * matching the style and position of the MagicPrompt buttons.
 */
function addWD14TaggerButtons() {
    let altPromptRegion = document.querySelector('.alt_prompt_region');
    if (!altPromptRegion) {
        return;
    }

    // Avoid double-adding
    if (document.getElementById('wd14tagger_container')) {
        return;
    }

    // Container — same pattern as .magicprompt.prompt-buttons-container
    let container = document.createElement('div');
    container.id = 'wd14tagger_container';
    container.className = 'wd14tagger prompt-buttons-container';

    // Generate Tags button
    let generateBtn = document.createElement('button');
    generateBtn.id = 'wd14tagger_generate_btn';
    generateBtn.className = 'wd14tagger prompt-button';
    generateBtn.innerHTML = '<span class="wd14tagger-btn-icon">\uD83C\uDFF7\uFE0F</span><span class="wd14tagger-btn-label"> Generate Tags</span>';
    generateBtn.title = 'Analyze current image and generate WD14 tags';
    generateBtn.addEventListener('click', handleWD14GenerateTags);

    // Gear settings button
    let settingsBtn = document.createElement('button');
    settingsBtn.className = 'wd14tagger prompt-settings-button';
    settingsBtn.innerHTML = '\u2699\uFE0F';
    settingsBtn.title = 'Select tagger model';

    // Settings panel (appended to body, positioned fixed near button)
    let settingsPanel = document.createElement('div');
    settingsPanel.id = 'wd14tagger_settings_panel';
    settingsPanel.className = 'wd14tagger prompt-settings-panel';
    settingsPanel.style.display = 'none';
    let optionsHtml = wd14TaggerModels.map(m => `<option value="${m.id}">${m.label}</option>`).join('');
    settingsPanel.innerHTML = `
        <div class="wd14tagger settings-panel-header">
            <h3>Tagger Settings</h3>
            <button class="wd14tagger panel-close-btn">&times;</button>
        </div>
        <div class="wd14tagger settings-panel-body">
            <div class="wd14tagger feature-setting">
                <label for="wd14tagger_model_select">Model:</label>
                <select id="wd14tagger_model_select" class="wd14tagger feature-select">${optionsHtml}</select>
            </div>
            <div class="wd14tagger feature-setting">
                <label for="wd14tagger_threshold_number">Confidence threshold percentage:</label>
                <input id="wd14tagger_threshold_number" type="number" min="0" max="100" step="1" value="35" class="auto-slider-number" autocomplete="off"><span class="wd14tagger threshold-pct-label">%</span>
                <br>
                <div class="auto-slider-range-wrapper" style="--range-value: 35%">
                    <input id="wd14tagger_threshold_input" type="range" min="0" max="100" step="1" value="35" class="auto-slider-range" oninput="updateRangeStyle(this)" onchange="updateRangeStyle(this)">
                </div>
                <div class="wd14tagger setting-description">Tags below this confidence score are excluded (0–100%).</div>
            </div>
            <div class="wd14tagger feature-setting">
                <label for="wd14tagger_filter_input">Filter tags (comma-separated):</label>
                <textarea id="wd14tagger_filter_input" class="wd14tagger feature-select wd14tagger-filter-input" rows="3" placeholder="e.g. realistic, watermark, signature"></textarea>
                <div class="wd14tagger setting-description">Tags in this list will be removed from the output.</div>
            </div>
            <div class="wd14tagger feature-setting">
                <label>Insert Tags:</label>
                <div class="wd14tagger insert-mode-group">
                    <label><input type="radio" name="wd14tagger_insert_mode" value="replace" checked> Replace</label>
                    <label><input type="radio" name="wd14tagger_insert_mode" value="prepend"> Prepend</label>
                    <label><input type="radio" name="wd14tagger_insert_mode" value="append"> Append</label>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(settingsPanel);

    // Restore saved settings from localStorage
    let savedModel = localStorage.getItem('wd14tagger_model');
    let savedFilters = localStorage.getItem('wd14tagger_filters');
    let savedThreshold = localStorage.getItem('wd14tagger_threshold');
    if (savedModel) {
        let modelSel = settingsPanel.querySelector('#wd14tagger_model_select');
        if (modelSel) {
            modelSel.value = savedModel;
        }
    }
    if (savedFilters !== null) {
        let filterIn = settingsPanel.querySelector('#wd14tagger_filter_input');
        if (filterIn) {
            filterIn.value = savedFilters;
        }
    }
    let savedInsertMode = localStorage.getItem('wd14tagger_insert_mode') || 'replace';
    let insertRadio = settingsPanel.querySelector(`input[name="wd14tagger_insert_mode"][value="${savedInsertMode}"]`);
    if (insertRadio) {
        insertRadio.checked = true;
    }
    if (savedThreshold !== null) {
        // Migrate old decimal values (0.0–1.0) stored before the percentage change
        let pct = parseFloat(savedThreshold);
        if (pct <= 1) { pct = Math.round(pct * 100); }
        let slider = settingsPanel.querySelector('#wd14tagger_threshold_input');
        let numInput = settingsPanel.querySelector('#wd14tagger_threshold_number');
        if (slider) { slider.value = pct; updateRangeStyle(slider); }
        if (numInput) { numInput.value = pct; }
    }

    // Persist changes to localStorage
    settingsPanel.addEventListener('change', function() {
        let modelSel = settingsPanel.querySelector('#wd14tagger_model_select');
        if (modelSel) {
            localStorage.setItem('wd14tagger_model', modelSel.value);
        }
        let checkedMode = settingsPanel.querySelector('input[name="wd14tagger_insert_mode"]:checked');
        if (checkedMode) {
            localStorage.setItem('wd14tagger_insert_mode', checkedMode.value);
        }
    });
    settingsPanel.addEventListener('input', function(e) {
        if (e.target.id === 'wd14tagger_filter_input') {
            localStorage.setItem('wd14tagger_filters', e.target.value);
        }
        if (e.target.id === 'wd14tagger_threshold_input') {
            let val = Math.round(parseFloat(e.target.value));
            localStorage.setItem('wd14tagger_threshold', val);
            let numInput = settingsPanel.querySelector('#wd14tagger_threshold_number');
            if (numInput) { numInput.value = val; }
        }
        if (e.target.id === 'wd14tagger_threshold_number') {
            let val = Math.min(100, Math.max(0, Math.round(parseFloat(e.target.value) || 0)));
            localStorage.setItem('wd14tagger_threshold', val);
            let slider = settingsPanel.querySelector('#wd14tagger_threshold_input');
            if (slider) { slider.value = val; updateRangeStyle(slider); }
        }
    });

    function positionSettingsPanel() {
        let rect = settingsBtn.getBoundingClientRect();
        let panelWidth = 300; // matches CSS width
        settingsPanel.style.position = 'fixed';
        settingsPanel.style.zIndex = '10000';
        // Measure panel height after making it briefly visible but off-screen
        settingsPanel.style.visibility = 'hidden';
        settingsPanel.style.display = 'block';
        let panelHeight = settingsPanel.offsetHeight;
        settingsPanel.style.display = 'none';
        settingsPanel.style.visibility = '';
        // Position to the top-right of the gear button so it stays visible when the sidebar is collapsed
        let leftPos = rect.right + 5;
        // Clamp so panel doesn't overflow off the right edge of the viewport
        if (leftPos + panelWidth > window.innerWidth - 8) {
            leftPos = window.innerWidth - panelWidth - 8;
        }
        let topPos = rect.top;
        // Clamp so panel doesn't overflow off the bottom of the viewport
        if (topPos + panelHeight > window.innerHeight - 8) {
            topPos = window.innerHeight - panelHeight - 8;
            if (topPos < 8) {
                topPos = 8;
            }
        }
        settingsPanel.style.top = topPos + 'px';
        settingsPanel.style.left = leftPos + 'px';
        settingsPanel.style.right = '';
    }

    settingsBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        if (settingsPanel.style.display === 'block') {
            settingsPanel.style.display = 'none';
            return;
        }
        positionSettingsPanel();
        settingsPanel.style.display = 'block';
    });

    let closeBtn = settingsPanel.querySelector('.wd14tagger.panel-close-btn');
    if (closeBtn) {
        closeBtn.addEventListener('click', function() {
            settingsPanel.style.display = 'none';
        });
    }

    document.addEventListener('mousedown', function(e) {
        if (settingsPanel.style.display === 'block' &&
            !settingsPanel.contains(e.target) &&
            !settingsBtn.contains(e.target)) {
            settingsPanel.style.display = 'none';
        }
    });

    container.appendChild(generateBtn);
    container.appendChild(settingsBtn);

    // Insert at the beginning of the prompt region, same as MagicPrompt
    altPromptRegion.insertBefore(container, altPromptRegion.firstChild);
}

document.addEventListener('DOMContentLoaded', function() {
    addWD14TaggerButtons();
});
