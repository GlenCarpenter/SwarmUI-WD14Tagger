using Newtonsoft.Json.Linq;
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

    /// <summary>Hidden T2I param for the WD14 tagger model to use with the <wd14tagger> prompt tag.</summary>
    public static T2IRegisteredParam<string> ModelParam;

    /// <summary>Hidden T2I param for the WD14 tagger confidence threshold (0.0–1.0) for the <wd14tagger> prompt tag.</summary>
    public static T2IRegisteredParam<double> ThresholdParam;

    /// <summary>Hidden T2I param for comma-separated tags to filter out when using the <wd14tagger> prompt tag.</summary>
    public static T2IRegisteredParam<string> FilterTagsParam;

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
        string model = context.Input.Get(ModelParam, "SmilingWolf/wd-eva02-large-tagger-v3");
        float threshold = (float)context.Input.Get(ThresholdParam, 0.35);
        string filterTags = context.Input.Get(FilterTagsParam, "");
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

        ModelParam = T2IParamTypes.Register<string>(new(
            Name: "WD14 Tagger Model",
            Description: "Model to use when the <wd14tagger> prompt tag is processed.",
            Default: "SmilingWolf/wd-eva02-large-tagger-v3",
            VisibleNormally: false,
            HideFromMetadata: true,
            DoNotPreview: true
        ));
        ThresholdParam = T2IParamTypes.Register<double>(new(
            Name: "WD14 Tagger Threshold",
            Description: "Confidence threshold (0.0–1.0) for the <wd14tagger> prompt tag.",
            Default: "0.35",
            Min: 0,
            Max: 1,
            ViewType: ParamViewType.SLIDER,
            VisibleNormally: false,
            HideFromMetadata: true,
            DoNotPreview: true
        ));
        FilterTagsParam = T2IParamTypes.Register<string>(new(
            Name: "WD14 Tagger Filter Tags",
            Description: "Comma-separated tags to exclude when the <wd14tagger> prompt tag is processed.",
            Default: "",
            VisibleNormally: false,
            HideFromMetadata: true,
            DoNotPreview: true
        ));

        Logs.Info("Register <wd14tagger> prompt token");
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
        Logs.Info("WD14Tagger extension initialized.");
        WD14TaggerAPI.Register();
    }
}
