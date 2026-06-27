using System.IO;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;
using SwarmUI.Accounts;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Utils;
using SwarmUI.WebAPI;

namespace GlenCarpenter.Extensions.WD14TaggerExtension;

/// <summary>Permission definitions for the WD14Tagger extension.</summary>
public static class WD14TaggerPermissions
{
    /// <summary>Permission group for WD14Tagger.</summary>
    public static readonly PermInfoGroup WD14TaggerPermGroup = new("WD14Tagger", "Permissions related to WD14 Tagger functionality.");

    /// <summary>Permission to call the tag-generation API.</summary>
    public static readonly PermInfo PermGenerateTags = Permissions.Register(new(
        "wd14tagger_generate_tags",
        "Generate Tags",
        "Allows the user to run WD14 tag generation on images.",
        PermissionDefault.USER,
        WD14TaggerPermGroup));
}

/// <summary>API routes for the WD14Tagger extension.</summary>
[API.APIClass("API routes related to WD14Tagger extension")]
public static class WD14TaggerAPI
{
    /// <summary>Default WD14 tagger model ID.</summary>
    public const string DefaultModelId = "SmilingWolf/wd-eva02-large-tagger-v3";

    /// <summary>Default confidence threshold for general tag inclusion.</summary>
    public const float DefaultGeneralThreshold = 0.35f;

    /// <summary>Default confidence threshold for character tag inclusion.</summary>
    public const float DefaultCharacterThreshold = 0.85f;

    /// <summary>Registers all API calls for this extension.</summary>
    public static void Register()
    {
        API.RegisterAPICall(WD14TaggerGenerateTags, true, WD14TaggerPermissions.PermGenerateTags);
        API.RegisterAPICall(WD14TaggerApplyFilters, true, WD14TaggerPermissions.PermGenerateTags);
    }

    /// <summary>Allowed characters in a HuggingFace repo ID (namespace/repo-name).</summary>
    private static readonly Regex SafeRepoIdPattern = new(@"^[A-Za-z0-9_\-/\.]+$", RegexOptions.Compiled);

    /// <summary>Matches a trailing prompt-weight suffix like ":1.3" on a tag's core text.</summary>
    private static readonly Regex TrailingWeightPattern = new(@":\s*\d+(?:\.\d+)?\s*$", RegexOptions.Compiled);

    /// <summary>Maximum allowed byte length for the filterTags string.</summary>
    private const int MaxFilterTagsLength = 4096;

    /// <summary>Supported matching styles for filter rule source tags.</summary>
    private enum FilterTagMatchMode
    {
        Exact,
        StartsWithPhrase,
        EndsWithPhrase,
        ContainsPhrase
    }

    /// <summary>A single parsed wildcard rule.</summary>
    private record FilterTagRule(FilterTagMatchMode MatchMode, string SourceTag, string TargetTag);

    /// <summary>Parsed filter settings grouped by precedence.</summary>
    private record FilterTagRules(
        HashSet<string> ExactExcludedTags,
        Dictionary<string, string> ExactReplacementTags,
        List<FilterTagRule> WildcardExclusionRules,
        List<FilterTagRule> WildcardReplacementRules);

    /// <summary>
    /// Normalizes a raw filterTags string: truncates to max length, strips any character
    /// that isn't a printable ASCII letter/digit/punctuation/space (no control chars).
    /// </summary>
    private static string SanitizeFilterTags(string raw)
    {
        if (string.IsNullOrEmpty(raw)) return "";
        if (raw.Length > MaxFilterTagsLength) raw = raw[..MaxFilterTagsLength];
        // Allow printable ASCII only (0x20–0x7E)
        return new string(raw.Where(c => c >= 0x20 && c <= 0x7E).ToArray()).Trim();
    }

    /// <summary>Normalizes whitespace so boundary-aware matching treats repeated spaces consistently.</summary>
    private static string NormalizeTagText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return "";
        }
        return string.Join(' ', text.Split(' ', StringSplitOptions.RemoveEmptyEntries));
    }

    /// <summary>Parses a rule source token into an exact or wildcard match mode.</summary>
    private static FilterTagMatchMode ParseFilterTagMatchMode(string rawSourceTag, out string normalizedSourceTag)
    {
        string sourceTag = NormalizeTagText(rawSourceTag);
        bool hasLeadingWildcard = sourceTag.StartsWith('*');
        bool hasTrailingWildcard = sourceTag.EndsWith('*');
        string coreTag = sourceTag.Trim('*');
        if (string.IsNullOrWhiteSpace(coreTag) || coreTag.Contains('*'))
        {
            normalizedSourceTag = sourceTag;
            return FilterTagMatchMode.Exact;
        }
        normalizedSourceTag = NormalizeTagText(coreTag);
        if (hasLeadingWildcard && hasTrailingWildcard)
        {
            return FilterTagMatchMode.ContainsPhrase;
        }
        if (hasLeadingWildcard)
        {
            return FilterTagMatchMode.EndsWithPhrase;
        }
        if (hasTrailingWildcard)
        {
            return FilterTagMatchMode.StartsWithPhrase;
        }
        return FilterTagMatchMode.Exact;
    }

    /// <summary>Returns whether the character should count as part of a word for boundary checks.</summary>
    private static bool IsWordCharacter(char character)
    {
        return char.IsLetterOrDigit(character);
    }

    /// <summary>Checks whether a candidate phrase match starts and ends on non-word boundaries.</summary>
    private static bool IsPhraseBoundaryMatch(string candidateTag, int startIndex, int length)
    {
        int endIndex = startIndex + length;
        bool hasLeadingBoundary = startIndex <= 0 || !IsWordCharacter(candidateTag[startIndex - 1]);
        bool hasTrailingBoundary = endIndex >= candidateTag.Length || !IsWordCharacter(candidateTag[endIndex]);
        return hasLeadingBoundary && hasTrailingBoundary;
    }

    /// <summary>Finds a boundary-aware phrase match anywhere within a candidate tag.</summary>
    private static bool ContainsPhraseBoundaryMatch(string candidateTag, string sourceTag)
    {
        int searchIndex = 0;
        while (searchIndex < candidateTag.Length)
        {
            int matchIndex = candidateTag.IndexOf(sourceTag, searchIndex, StringComparison.OrdinalIgnoreCase);
            if (matchIndex < 0)
            {
                return false;
            }
            if (IsPhraseBoundaryMatch(candidateTag, matchIndex, sourceTag.Length))
            {
                return true;
            }
            searchIndex = matchIndex + 1;
        }
        return false;
    }

    /// <summary>Replaces boundary-aware phrase matches in a tag according to the wildcard rule mode.</summary>
    private static string ApplyWildcardReplacement(string candidateTag, FilterTagRule rule)
    {
        string normalizedCandidate = NormalizeTagText(candidateTag);
        if (string.IsNullOrWhiteSpace(normalizedCandidate) || string.IsNullOrWhiteSpace(rule?.SourceTag) || rule.TargetTag is null)
        {
            return normalizedCandidate;
        }
        if (rule.MatchMode == FilterTagMatchMode.StartsWithPhrase)
        {
            if (normalizedCandidate.StartsWith(rule.SourceTag, StringComparison.OrdinalIgnoreCase)
                && IsPhraseBoundaryMatch(normalizedCandidate, 0, rule.SourceTag.Length))
            {
                return rule.TargetTag + normalizedCandidate[rule.SourceTag.Length..];
            }
            return normalizedCandidate;
        }
        if (rule.MatchMode == FilterTagMatchMode.EndsWithPhrase)
        {
            int startIndex = normalizedCandidate.Length - rule.SourceTag.Length;
            if (startIndex >= 0
                && normalizedCandidate.EndsWith(rule.SourceTag, StringComparison.OrdinalIgnoreCase)
                && IsPhraseBoundaryMatch(normalizedCandidate, startIndex, rule.SourceTag.Length))
            {
                return normalizedCandidate[..startIndex] + rule.TargetTag;
            }
            return normalizedCandidate;
        }
        if (rule.MatchMode == FilterTagMatchMode.ContainsPhrase)
        {
            int searchIndex = 0;
            StringBuilder builder = new();
            while (searchIndex < normalizedCandidate.Length)
            {
                int matchIndex = normalizedCandidate.IndexOf(rule.SourceTag, searchIndex, StringComparison.OrdinalIgnoreCase);
                if (matchIndex < 0)
                {
                    builder.Append(normalizedCandidate[searchIndex..]);
                    break;
                }
                if (!IsPhraseBoundaryMatch(normalizedCandidate, matchIndex, rule.SourceTag.Length))
                {
                    builder.Append(normalizedCandidate[searchIndex..(matchIndex + 1)]);
                    searchIndex = matchIndex + 1;
                    continue;
                }
                builder.Append(normalizedCandidate[searchIndex..matchIndex]);
                builder.Append(rule.TargetTag);
                searchIndex = matchIndex + rule.SourceTag.Length;
            }
            return builder.ToString();
        }
        return normalizedCandidate;
    }

    /// <summary>Checks whether a tag matches a boundary-aware wildcard phrase pattern.</summary>
    private static bool MatchesFilterRule(string candidateTag, FilterTagRule rule)
    {
        return MatchesFilterRule(candidateTag, rule.SourceTag, rule.MatchMode);
    }

    /// <summary>Checks whether a tag matches an exact or boundary-aware phrase pattern.</summary>
    private static bool MatchesFilterRule(string candidateTag, string sourceTag, FilterTagMatchMode matchMode)
    {
        string normalizedCandidate = NormalizeTagText(candidateTag);
        if (string.IsNullOrWhiteSpace(normalizedCandidate) || string.IsNullOrWhiteSpace(sourceTag))
        {
            return false;
        }
        if (matchMode == FilterTagMatchMode.Exact)
        {
            return normalizedCandidate.Equals(sourceTag, StringComparison.OrdinalIgnoreCase);
        }
        if (matchMode == FilterTagMatchMode.StartsWithPhrase)
        {
            return normalizedCandidate.StartsWith(sourceTag, StringComparison.OrdinalIgnoreCase)
                && IsPhraseBoundaryMatch(normalizedCandidate, 0, sourceTag.Length);
        }
        if (matchMode == FilterTagMatchMode.EndsWithPhrase)
        {
            int startIndex = normalizedCandidate.Length - sourceTag.Length;
            return startIndex >= 0
                && normalizedCandidate.EndsWith(sourceTag, StringComparison.OrdinalIgnoreCase)
                && IsPhraseBoundaryMatch(normalizedCandidate, startIndex, sourceTag.Length);
        }
        return ContainsPhraseBoundaryMatch(normalizedCandidate, sourceTag);
    }

    /// <summary>Splits a string on a delimiter while treating double-quoted spans as literal, so the delimiter can appear inside a quoted rule value.</summary>
    private static List<string> SplitRespectingQuotes(string text, char delimiter)
    {
        List<string> parts = [];
        if (string.IsNullOrEmpty(text))
        {
            return parts;
        }
        StringBuilder current = new();
        bool inQuotes = false;
        foreach (char character in text)
        {
            if (character == '"')
            {
                inQuotes = !inQuotes;
                current.Append(character);
            }
            else if (character == delimiter && !inQuotes)
            {
                parts.Add(current.ToString());
                current.Clear();
            }
            else
            {
                current.Append(character);
            }
        }
        parts.Add(current.ToString());
        return parts;
    }

    /// <summary>Finds the index of the first occurrence of a character that is not inside a double-quoted span, or -1 if none.</summary>
    private static int IndexOfOutsideQuotes(string text, char target)
    {
        bool inQuotes = false;
        for (int index = 0; index < text.Length; index++)
        {
            char character = text[index];
            if (character == '"')
            {
                inQuotes = !inQuotes;
            }
            else if (character == target && !inQuotes)
            {
                return index;
            }
        }
        return -1;
    }

    /// <summary>Removes a single pair of wrapping double quotes from a token if present, allowing commas, colons, and other special characters inside a rule value.</summary>
    private static string UnquoteToken(string token)
    {
        string trimmed = (token ?? "").Trim();
        if (trimmed.Length >= 2 && trimmed[0] == '"' && trimmed[^1] == '"')
        {
            return trimmed[1..^1];
        }
        return trimmed;
    }

    /// <summary>
    /// Parses a comma-separated filter string into exclusion and substitution rules.
    /// Entries of the form <c>source:target</c> replace a matching source tag; all other
    /// entries exclude a matching tag from the output. A source token can use
    /// <c>tag*</c>, <c>*tag</c>, or <c>*tag*</c> for boundary-aware phrase matching.
    /// Either side may be wrapped in double quotes (e.g. <c>smile:":)"</c> or
    /// <c>":)":"smiley face, emoji"</c>) so the value can contain commas, colons, or parentheses.
    /// </summary>
    private static FilterTagRules ParseFilterTagRules(string filterTags)
    {
        HashSet<string> exactExcludedTags = new(StringComparer.OrdinalIgnoreCase);
        Dictionary<string, string> exactReplacementTags = new(StringComparer.OrdinalIgnoreCase);
        List<FilterTagRule> wildcardExclusionRules = [];
        List<FilterTagRule> wildcardReplacementRules = [];
        if (string.IsNullOrWhiteSpace(filterTags))
        {
            return new FilterTagRules(exactExcludedTags, exactReplacementTags, wildcardExclusionRules, wildcardReplacementRules);
        }
        foreach (string rawEntry in SplitRespectingQuotes(filterTags, ','))
        {
            string entry = rawEntry.Trim();
            if (string.IsNullOrWhiteSpace(entry))
            {
                continue;
            }
            int separatorIndex = IndexOfOutsideQuotes(entry, ':');
            if (separatorIndex > 0 && separatorIndex < entry.Length - 1)
            {
                string sourceTag = UnquoteToken(entry[..separatorIndex]);
                string targetTag = NormalizeTagText(UnquoteToken(entry[(separatorIndex + 1)..]));
                if (!string.IsNullOrWhiteSpace(sourceTag) && !string.IsNullOrWhiteSpace(targetTag))
                {
                    FilterTagMatchMode matchMode = ParseFilterTagMatchMode(sourceTag, out string normalizedSourceTag);
                    if (matchMode == FilterTagMatchMode.Exact)
                    {
                        exactReplacementTags[normalizedSourceTag] = targetTag;
                    }
                    else
                    {
                        wildcardReplacementRules.Add(new FilterTagRule(matchMode, normalizedSourceTag, targetTag));
                    }
                    continue;
                }
            }
            FilterTagMatchMode exclusionMode = ParseFilterTagMatchMode(UnquoteToken(entry), out string normalizedExcludedTag);
            if (exclusionMode == FilterTagMatchMode.Exact)
            {
                exactExcludedTags.Add(normalizedExcludedTag);
            }
            else
            {
                wildcardExclusionRules.Add(new FilterTagRule(exclusionMode, normalizedExcludedTag, null));
            }
        }
        return new FilterTagRules(exactExcludedTags, exactReplacementTags, wildcardExclusionRules, wildcardReplacementRules);
    }

    /// <summary>
    /// Splits a candidate tag into prompt-weighting decoration and its core text so filter rules
    /// match the underlying tag. Surrounding parentheses and a trailing <c>:number</c> weight are
    /// peeled off (e.g. <c>((swimsuit))</c> and <c>realistic:1.3</c> both match as their core tag),
    /// and the decoration is preserved so kept or replaced tags retain their original weighting.
    /// </summary>
    private static (string Prefix, string Core, string Suffix) SplitPromptWeighting(string rawTag)
    {
        string working = (rawTag ?? "").Trim();
        int leadingParens = 0;
        int startIndex = 0;
        while (startIndex < working.Length && (working[startIndex] == '(' || char.IsWhiteSpace(working[startIndex])))
        {
            if (working[startIndex] == '(')
            {
                leadingParens++;
            }
            startIndex++;
        }
        int endIndex = working.Length;
        int trailingParens = 0;
        while (endIndex > startIndex && (working[endIndex - 1] == ')' || char.IsWhiteSpace(working[endIndex - 1])))
        {
            if (working[endIndex - 1] == ')')
            {
                trailingParens++;
            }
            endIndex--;
        }
        // Only treat parentheses as weighting decoration when balanced, so an unmatched paren in a
        // tag like the emoji ":)" is kept intact rather than mistaken for prompt weighting.
        if (leadingParens != trailingParens)
        {
            leadingParens = 0;
            trailingParens = 0;
            startIndex = 0;
            endIndex = working.Length;
        }
        string core = working[startIndex..endIndex];
        string weight = "";
        Match weightMatch = TrailingWeightPattern.Match(core);
        if (weightMatch.Success)
        {
            weight = core[weightMatch.Index..].Trim();
            core = core[..weightMatch.Index];
        }
        string prefix = new('(', leadingParens);
        string suffix = weight + new string(')', trailingParens);
        return (prefix, core, suffix);
    }

    /// <summary>
    /// Applies exact replacements, exact exclusions, wildcard replacements, and then wildcard exclusions.
    /// Wildcard matching is boundary-aware so <c>*hair</c> matches <c>hair</c>,
    /// <c>black hair</c>, and <c>arm-hair</c>, but not <c>chair</c> or <c>hair piece</c>.
    /// Prompt-weighting decoration (surrounding parentheses and a trailing <c>:number</c> weight)
    /// is ignored when matching and preserved on tags that are kept or replaced.
    /// </summary>
    private static string ApplyFilterTagRules(string rawTags, FilterTagRules rules)
    {
        if (string.IsNullOrWhiteSpace(rawTags))
        {
            return "";
        }
        List<string> updatedTags = [];
        foreach (string rawTag in rawTags.Split(',', StringSplitOptions.RemoveEmptyEntries))
        {
            (string prefix, string coreRaw, string suffix) = SplitPromptWeighting(rawTag);
            string tag = NormalizeTagText(coreRaw);
            if (string.IsNullOrWhiteSpace(tag))
            {
                continue;
            }
            if (rules.ExactReplacementTags.TryGetValue(tag, out string exactReplacement))
            {
                tag = exactReplacement;
            }
            if (rules.ExactExcludedTags.Contains(tag))
            {
                continue;
            }
            foreach (FilterTagRule wildcardReplacementRule in rules.WildcardReplacementRules)
            {
                if (MatchesFilterRule(tag, wildcardReplacementRule))
                {
                    tag = ApplyWildcardReplacement(tag, wildcardReplacementRule);
                    break;
                }
            }
            bool isWildcardExcluded = false;
            foreach (FilterTagRule wildcardExclusionRule in rules.WildcardExclusionRules)
            {
                if (MatchesFilterRule(tag, wildcardExclusionRule))
                {
                    isWildcardExcluded = true;
                    break;
                }
            }
            if (!isWildcardExcluded)
            {
                updatedTags.Add(prefix + tag + suffix);
            }
        }
        return string.Join(", ", updatedTags);
    }

    /// <summary>Generates WD14 tags for the provided base64-encoded image using the specified model.</summary>
    /// <param name="session">The calling user session.</param>
    /// <param name="imageBase64">Base64-encoded image data (PNG/JPG/WEBP).</param>
    /// <param name="modelId">HuggingFace repo ID of the tagger model.</param>
    /// <param name="generalThreshold">Confidence threshold (0.0-1.0) for general tags, or -1.0 to disable general tags.</param>
    /// <param name="characterThreshold">Confidence threshold (0.0-1.0) for character tags, or -1.0 to disable character tags.</param>
    /// <param name="filterTags">Comma-separated tag filters. Use <c>tag</c> to exclude, <c>source:target</c> to replace an exact tag, or wildcard forms like <c>tag*</c>, <c>*tag</c>, and <c>*tag*</c> to substitute only the matching phrase on word boundaries.</param>
    public static async Task<JObject> WD14TaggerGenerateTags(
        Session session,
        string imageBase64,
        string modelId = DefaultModelId,
        float generalThreshold = DefaultGeneralThreshold,
        float characterThreshold = DefaultCharacterThreshold,
        string filterTags = "")
    {
        // Validate inputs to prevent injection
        if (string.IsNullOrWhiteSpace(imageBase64))
        {
            return new JObject { ["success"] = false, ["error"] = "No image data provided." };
        }
        if (string.IsNullOrWhiteSpace(modelId) || !SafeRepoIdPattern.IsMatch(modelId))
        {
            return new JObject { ["success"] = false, ["error"] = "Invalid model ID format." };
        }
        if ((generalThreshold < 0f && generalThreshold != -1f) || generalThreshold > 1f)
        {
            return new JObject { ["success"] = false, ["error"] = "General threshold must be between 0.0 and 1.0, or -1.0 to disable." };
        }
        if ((characterThreshold < 0f && characterThreshold != -1f) || characterThreshold > 1f)
        {
            return new JObject { ["success"] = false, ["error"] = "Character threshold must be between 0.0 and 1.0, or -1.0 to disable." };
        }
        filterTags = SanitizeFilterTags(filterTags);
        FilterTagRules filterRules = ParseFilterTagRules(filterTags);

        string tempOutputPath = Path.Combine(Path.GetTempPath(), $"wd14tagger_{Guid.NewGuid():N}.txt");
        try
        {
            JObject workflow = new()
            {
                ["1"] = new JObject
                {
                    ["class_type"] = "SwarmLoadImageB64",
                    ["inputs"] = new JObject
                    {
                        ["image_base64"] = imageBase64
                    }
                },
                ["2"] = new JObject
                {
                    ["class_type"] = "WD14TaggerGenerate",
                    ["inputs"] = new JObject
                    {
                        ["images"] = new JArray() { "1", 0 },
                        ["model_id"] = modelId,
                        ["general_threshold"] = generalThreshold,
                        ["character_threshold"] = characterThreshold,
                        ["output_path"] = tempOutputPath
                    }
                }
            };
            using Session.GenClaim claim = session.Claim(liveGens: 1);
            await ComfyUIBackendExtension.RunArbitraryWorkflowOnFirstBackend(workflow.ToString(), _ => { }, allowRemote: false);
            if (!File.Exists(tempOutputPath))
            {
                return new JObject { ["success"] = false, ["error"] = "Workflow completed but produced no tag output. Ensure a self-start ComfyUI backend is available and loaded the WD14Tagger custom node." };
            }
            string rawTags = (await File.ReadAllTextAsync(tempOutputPath)).Trim();
            if (filterRules.ExactExcludedTags.Count > 0 || filterRules.ExactReplacementTags.Count > 0 || filterRules.WildcardExclusionRules.Count > 0 || filterRules.WildcardReplacementRules.Count > 0)
            {
                rawTags = ApplyFilterTagRules(rawTags, filterRules);
            }
            return new JObject
            {
                ["success"] = true,
                ["tags"] = rawTags
            };
        }
        catch (FormatException)
        {
            return new JObject { ["success"] = false, ["error"] = "Invalid base64 image data." };
        }
        catch (Exception ex)
        {
            Logs.Error($"WD14Tagger error: {ex.Message}");
            return new JObject { ["success"] = false, ["error"] = $"ComfyUI tagger workflow failed: {ex.Message}" };
        }
        finally
        {
            if (File.Exists(tempOutputPath))
            {
                try { File.Delete(tempOutputPath); } catch { /* best-effort cleanup */ }
            }
        }
    }

    /// <summary>Applies the exclude/replace filter rules to an existing tag string without running any tagger model.</summary>
    /// <param name="session">The calling user session.</param>
    /// <param name="tags">The raw, comma-separated tag string to filter.</param>
    /// <param name="filterTags">Comma-separated tag filters. Use <c>tag</c> to exclude, <c>source:target</c> to replace an exact tag, or wildcard forms like <c>tag*</c>, <c>*tag</c>, and <c>*tag*</c> to substitute only the matching phrase on word boundaries.</param>
    public static Task<JObject> WD14TaggerApplyFilters(
        Session session,
        string tags,
        string filterTags = "")
    {
        if (string.IsNullOrWhiteSpace(tags))
        {
            return Task.FromResult(new JObject { ["success"] = true, ["tags"] = "" });
        }
        filterTags = SanitizeFilterTags(filterTags);
        FilterTagRules filterRules = ParseFilterTagRules(filterTags);
        string filtered = ApplyFilterTagRules(tags, filterRules);
        return Task.FromResult(new JObject { ["success"] = true, ["tags"] = filtered });
    }

}