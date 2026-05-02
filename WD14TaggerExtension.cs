using Newtonsoft.Json.Linq;
using SwarmUI.Accounts;
using SwarmUI.Media;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace GlenCarpenter.Extensions.WD14TaggerExtension;

/// <summary>WD14Tagger - image tagging using SmilingWolf WD14 ONNX models from HuggingFace.</summary>
public class WD14TaggerExtension : Extension
{
    /// <summary>Path to this extension's directory, stored for use by the static API class.</summary>
    public static string ExtFolder;

    /// <summary>ExtraMeta key used to cache prompt-tag generated WD14 tags for this input.</summary>
    public const string PromptTagCacheKey = "wd14tagger_prompt_tags";

    /// <summary>
    /// Attempts to get an image source from the generation input for prompt-tag tagging.
    /// Prefers init image, then first prompt image.
    /// </summary>
    private static Image GetPromptTagImageSource(T2IParamInput input)
    {
        if (input.TryGet(T2IParamTypes.InitImage, out Image initImage) && initImage is not null)
        {
            return initImage;
        }
        if (input.TryGet(T2IParamTypes.PromptImages, out List<Image> promptImages) && promptImages is not null && promptImages.Count > 0)
        {
            return promptImages[0];
        }
        return null;
    }

    /// <summary>
    /// Generates WD14 tags for use inside a prompt tag expansion.
    /// Returns an empty string when no usable image source is available.
    /// </summary>
    private static string GeneratePromptTagTags(T2IPromptHandling.PromptTagContext context)
    {
        if (context?.Input is null)
        {
            return "";
        }
        if (context.Input.ExtraMeta.TryGetValue(PromptTagCacheKey, out object existing) && existing is string cached)
        {
            return cached;
        }
        Image source = GetPromptTagImageSource(context.Input);
        if (source is null)
        {
            context.TrackWarning("WD14Tagger prompt tag '<wd14tagger>' was used, but no init image or prompt image is available to tag.");
            context.Input.ExtraMeta[PromptTagCacheKey] = "";
            return "";
        }
        Session promptTagSession = context.Input.SourceSession;
        (string model, float threshold, string filterTags) = WD14TaggerAPI.GetPromptTagSettingsForSession(promptTagSession);
        JObject result;
        try
        {
            result = WD14TaggerAPI.WD14TaggerGenerateTags(context.Input.SourceSession, source.AsBase64, model, threshold, filterTags).GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            context.TrackWarning($"WD14Tagger prompt tag failed: {ex.Message}");
            context.Input.ExtraMeta[PromptTagCacheKey] = "";
            return "";
        }
        bool success = result?["success"]?.Value<bool>() ?? false;
        if (!success)
        {
            string err = result?["error"]?.Value<string>() ?? "Unknown WD14Tagger error.";
            context.TrackWarning($"WD14Tagger prompt tag failed: {err}");
            context.Input.ExtraMeta[PromptTagCacheKey] = "";
            return "";
        }
        string tags = result?["tags"]?.Value<string>() ?? "";
        if (string.IsNullOrWhiteSpace(tags))
        {
            context.TrackWarning("WD14Tagger prompt tag produced no tags above the current threshold.");
            context.Input.ExtraMeta[PromptTagCacheKey] = "";
            return "";
        }
        context.Input.ExtraMeta[PromptTagCacheKey] = tags;
        return tags;
    }

    public override void OnPreInit()
    {
        ExtFolder = FilePath;
        ScriptFiles.Add("Assets/wd14tagger.js");
        StyleSheetFiles.Add("Assets/wd14tagger.css");

        // Register <wd14tagger> prompt token. It expands to WD14 tags generated
        // from the request's init image (or first prompt image) before generation.
        T2IPromptHandling.PromptTagBasicProcessors["wd14tagger"] = (data, context) =>
        {
            return GeneratePromptTagTags(context);
        };
        T2IPromptHandling.PromptTagLengthEstimators["wd14tagger"] = (data, context) =>
        {
            return context?.Input?.ExtraMeta.TryGetValue(PromptTagCacheKey, out object existing) == true && existing is string cached ? cached : "";
        };
    }

    public override void OnInit()
    {
        WD14TaggerAPI.Register();
    }
}
