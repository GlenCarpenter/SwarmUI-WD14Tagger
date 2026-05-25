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
    }

    /// <summary>Allowed characters in a HuggingFace repo ID (namespace/repo-name).</summary>
    private static readonly Regex SafeRepoIdPattern = new(@"^[A-Za-z0-9_\-/\.]+$", RegexOptions.Compiled);

    /// <summary>Maximum allowed byte length for the filterTags string.</summary>
    private const int MaxFilterTagsLength = 4096;

    /// <summary>Parsed filter settings: exact-match exclusions and exact-match substitutions.</summary>
    private record FilterTagRules(HashSet<string> ExcludedTags, Dictionary<string, string> ReplacementTags);

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

    /// <summary>
    /// Parses a comma-separated filter string into exclusion and substitution rules.
    /// Entries of the form <c>source:target</c> replace an exact tag match; all other
    /// entries exclude an exact tag match from the output.
    /// </summary>
    private static FilterTagRules ParseFilterTagRules(string filterTags)
    {
        HashSet<string> excludedTags = new(StringComparer.OrdinalIgnoreCase);
        Dictionary<string, string> replacementTags = new(StringComparer.OrdinalIgnoreCase);
        if (string.IsNullOrWhiteSpace(filterTags))
        {
            return new FilterTagRules(excludedTags, replacementTags);
        }
        foreach (string rawEntry in filterTags.Split(','))
        {
            string entry = rawEntry.Trim();
            if (string.IsNullOrWhiteSpace(entry))
            {
                continue;
            }
            int separatorIndex = entry.IndexOf(':');
            if (separatorIndex > 0 && separatorIndex < entry.Length - 1)
            {
                string sourceTag = entry[..separatorIndex].Trim();
                string targetTag = entry[(separatorIndex + 1)..].Trim();
                if (!string.IsNullOrWhiteSpace(sourceTag) && !string.IsNullOrWhiteSpace(targetTag))
                {
                    replacementTags[sourceTag] = targetTag;
                    continue;
                }
            }
            excludedTags.Add(entry);
        }
        return new FilterTagRules(excludedTags, replacementTags);
    }

    /// <summary>Applies exact-match substitutions first, then exact-match exclusions.</summary>
    private static string ApplyFilterTagRules(string rawTags, FilterTagRules rules)
    {
        if (string.IsNullOrWhiteSpace(rawTags))
        {
            return "";
        }
        IEnumerable<string> updatedTags = rawTags
            .Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Select(t => t.Trim())
            .Where(t => !string.IsNullOrWhiteSpace(t))
            .Select(t => rules.ReplacementTags.TryGetValue(t, out string replacement) ? replacement : t)
            .Where(t => !rules.ExcludedTags.Contains(t));
        return string.Join(", ", updatedTags);
    }

    /// <summary>Generates WD14 tags for the provided base64-encoded image using the specified model.</summary>
    /// <param name="session">The calling user session.</param>
    /// <param name="imageBase64">Base64-encoded image data (PNG/JPG/WEBP).</param>
    /// <param name="modelId">HuggingFace repo ID of the tagger model.</param>
    /// <param name="generalThreshold">Confidence threshold (0.0-1.0) for general tags, or -1.0 to disable general tags.</param>
    /// <param name="characterThreshold">Confidence threshold (0.0-1.0) for character tags, or -1.0 to disable character tags.</param>
    /// <param name="filterTags">Comma-separated exact tag filters. Use <c>tag</c> to exclude or <c>source:target</c> to substitute.</param>
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
            if (filterRules.ExcludedTags.Count > 0 || filterRules.ReplacementTags.Count > 0)
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

}
