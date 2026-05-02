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
    let thresholdElem = document.getElementById('input_wdtaggerthreshold');
    let filterTagsElem = document.getElementById('input_wdtaggerfiltertags');
    let insertModeElem = document.getElementById('input_wdtaggerinsertmode');

    let modelId = modelElem ? modelElem.value : 'SmilingWolf/wd-eva02-large-tagger-v3';
    let threshold = parseFloat(thresholdElem ? thresholdElem.value : '0.35') || 0.35;
    let filterTags = filterTagsElem ? filterTagsElem.value : '';
    let insertMode = insertModeElem ? insertModeElem.value : 'replace';

    let result = await new Promise((resolve, reject) => {
        genericRequest(
            'WD14TaggerGenerateTags',
            { imageBase64: base64Data, modelId: modelId, threshold: threshold, filterTags: filterTags },
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
    if (!promptBox) {
        return;
    }
    if (!result.tags) {
        showError('WD14 Tagger: No tags found above the current threshold.');
        return;
    }
    let existing = promptBox.value;
    if (existing.includes('<wd14tagger>')) {
        promptBox.value = existing.replace('<wd14tagger>', result.tags);
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

// Wire up the media button and prompt tab completion once the page is ready.
setTimeout(() => {
    if (typeof promptTabComplete !== 'undefined') {
        promptTabComplete.registerPrefix('wd14tagger', 'Automatically generate WD14 tags from the init image and insert them into the prompt.', () => [
            '\nAdd "<wd14tagger>" anywhere in your prompt to auto-tag your init image using WD14.'
        ], true);
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
