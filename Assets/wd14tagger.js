/**
 * wd14tagger.js
 * WD14 Tagger extension for SwarmUI.
 * Registers a "Generate WD14 Tags" button in the image viewer via registerMediaButton.
 * Analyzes the viewed image using WD14 and inserts tags into the prompt box.
 * Settings (model, threshold, filter tags, insert mode) are managed as T2I parameters
 * in the "WD14 Tagger" parameter group.
 */

'use strict';

/**
 * Fetches an image from a src URL (or data-URL) as a base64-encoded string.
 * Returns null if no image is available.
 * @param {string} src - Image URL or data-URL passed by registerMediaButton.
 */
async function wd14TaggerGetImageBase64(src) {
    if (src.startsWith('data:')) {
        let b64 = src.split(',')[1];
        return b64 || null;
    }
    let fetchResponse = await fetch(src);
    let blob = await fetchResponse.blob();
    return new Promise((resolve, reject) => {
        let reader = new FileReader();
        reader.onloadend = () => {
            let b64 = reader.result.split(',')[1];
            resolve(b64 || null);
        };
        reader.onerror = () => reject(new Error('Failed to read image data.'));
        reader.readAsDataURL(blob);
    });
}

/**
 * Handles the "Generate Tags" media button click.
 * Reads T2I parameter values from the DOM, calls the WD14TaggerGenerateTags API,
 * and inserts the returned tags into the main prompt box.
 * @param {string} src - Image URL or data-URL provided by registerMediaButton.
 */
async function handleWD14GenerateTags(src) {
    let base64Data;
    let didStartGeneration = false;
    try {
        base64Data = await wd14TaggerGetImageBase64(src);
    }
    catch (err) {
        showError('WD14 Tagger: Failed to read the current image. ' + err.message);
        return;
    }
    if (!base64Data) {
        showError('WD14 Tagger: No image available.');
        return;
    }

    let modelElem = document.getElementById('input_wdtaggermodel');
    let generalThresholdElem = document.getElementById('input_wdtaggergeneralthreshold');
    let generalToggleElem = document.getElementById('input_wdtaggergeneralthreshold_toggle');
    let characterThresholdElem = document.getElementById('input_wdtaggercharacterthreshold');
    let characterToggleElem = document.getElementById('input_wdtaggercharacterthreshold_toggle');
    let filterTagsElem = document.getElementById('input_wdtaggerfiltertags');
    let insertModeElem = document.getElementById('input_wdtaggerinsertmode');

    let modelId = modelElem ? modelElem.value : 'SmilingWolf/wd-eva02-large-tagger-v3';
    let generalEnabled = !generalToggleElem || generalToggleElem.checked;
    let generalThreshold = generalEnabled ? (parseFloat(generalThresholdElem ? generalThresholdElem.value : '0.35') || 0.35) : -1;
    let characterEnabled = !characterToggleElem || characterToggleElem.checked;
    let characterThreshold = characterEnabled ? (parseFloat(characterThresholdElem ? characterThresholdElem.value : '0.85') || 0.85) : -1;
    let filterTags = filterTagsElem ? filterTagsElem.value : '';
    let insertMode = insertModeElem ? insertModeElem.value : 'replace';
    let promptBox = document.getElementById('alt_prompt_textbox');
    let existingPrompt = promptBox ? promptBox.value : '';
    let promptTagMatch = existingPrompt.match(/<wd14tagger(?::([^>]+))?>/i);
    if (promptTagMatch && promptTagMatch[1] && promptTagMatch[1].trim()) {
        let promptTagSettings = wd14TaggerParsePromptTagArgs(promptTagMatch[1]);
        if (promptTagSettings.modelId) {
            modelId = promptTagSettings.modelId;
        }
        if (promptTagSettings.generalThreshold !== null) {
            generalThreshold = promptTagSettings.generalThreshold;
        }
        if (promptTagSettings.characterThreshold !== null) {
            characterThreshold = promptTagSettings.characterThreshold;
        }
    }

    try {
        let request = new Promise((resolve, reject) => {
            genericRequest(
                'WD14TaggerGenerateTags',
                { imageBase64: base64Data, modelId: modelId, generalThreshold: generalThreshold, characterThreshold: characterThreshold, filterTags: filterTags },
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
        if (typeof updateGenCount === 'function') {
            didStartGeneration = true;
            updateGenCount();
        }
        let result = await request;

        if (!promptBox) {
            return;
        }
        if (!result.tags) {
            showError('WD14 Tagger: No tags found above the current threshold.');
            return;
        }
        let existing = promptBox.value;
        if (promptTagMatch) {
            promptBox.value = existing.replace(promptTagMatch[0], result.tags);
        }
        else if (insertMode === 'prepend') {
            promptBox.value = existing.trim() ? result.tags + ', ' + existing : result.tags;
        }
        else if (insertMode === 'append') {
            promptBox.value = existing.trim() ? existing + ', ' + result.tags : result.tags;
        }
        else {
            promptBox.value = result.tags;
        }
        triggerChangeFor(promptBox);
        promptBox.focus();
        promptBox.setSelectionRange(0, promptBox.value.length);
    }
    catch (err) {
        showError('WD14 Tagger: ' + err.message);
    }
    finally {
        if (didStartGeneration && typeof updateGenCount === 'function') {
            updateGenCount();
        }
    }
}

/** Parses optional prompt-tag positional arguments: model, general threshold, character threshold. */
function wd14TaggerParsePromptTagArgs(spec) {
    let parts = (spec || '').split(',').map(part => part.trim());
    let parseThreshold = (value) => {
        if (!value) {
            return null;
        }
        let parsed = parseFloat(value);
        if (isNaN(parsed)) {
            return null;
        }
        return parsed;
    };
    return {
        modelId: parts.length > 0 && parts[0] ? parts[0] : null,
        generalThreshold: parts.length > 1 ? parseThreshold(parts[1]) : null,
        characterThreshold: parts.length > 2 ? parseThreshold(parts[2]) : null
    };
}

/** Returns the list of WD14 tagger model IDs exposed by the extension parameter dropdown. */
function wd14TaggerGetAvailableModels() {
    if (typeof rawGenParamTypesFromServer !== 'undefined' && rawGenParamTypesFromServer) {
        let modelParam = rawGenParamTypesFromServer.find(p => p.id == 'wdtaggermodel');
        if (modelParam && modelParam.values && modelParam.values.length > 0) {
            return modelParam.values;
        }
    }
    let modelElem = document.getElementById('input_wdtaggermodel');
    if (modelElem && modelElem.options) {
        return [...modelElem.options].map(o => o.value).filter(v => v);
    }
    return [];
}

/** Defaults both threshold toggles to enabled on first load (no user cookie). */
function wd14TaggerDefaultToggles() {
    for (let id of ['input_wdtaggergeneralthreshold', 'input_wdtaggercharacterthreshold']) {
        let toggler = document.getElementById(id + '_toggle');
        if (toggler && !getCookie(`lastparam_${id}_toggle`)) {
            toggler.checked = true;
            doToggleEnable(id);
        }
    }
}

postParamBuildSteps.push(wd14TaggerDefaultToggles);

// Wire up the media button and prompt tab completion once the page is ready.
setTimeout(() => {
    if (typeof promptTabComplete !== 'undefined') {
        promptTabComplete.registerPrefix('wd14tagger', 'Automatically generate tags from the init image and insert them into the prompt. Type ":" after the tag name to pick a model override.', (prefix) => {
            let hasThresholdArgs = prefix.includes(',');
            let helpText = [
                '\nAdd "<wd14tagger>" anywhere in your prompt to auto-tag your init image using the selected tagger model.',
                '\nUse "<wd14tagger:model,general_threshold,character_threshold>" for per-tag overrides. Spaces around commas are ignored.',
                '\nExample: "<wd14tagger:deepghs/pixai-tagger-v0.9-onnx, 0.5, 0.85>".'
            ];
            let models = wd14TaggerGetAvailableModels();
            if (!hasThresholdArgs && models.length > 0) {
                return helpText.concat(promptTabComplete.getOrderedMatches(models, prefix.toLowerCase()));
            }
            return helpText;
        }, true);
    }
    if (typeof registerMediaButton !== 'function') {
        console.warn('WD14 Tagger: registerMediaButton is not available — SwarmUI version may be too old.');
        return;
    }
    registerMediaButton(
        'Generate WD14 Tags',
        (src) => handleWD14GenerateTags(src),
        'Analyze this image with WD14 and insert tags into the prompt',
        ['image'],
        true,
        true
    );
}, 0);
