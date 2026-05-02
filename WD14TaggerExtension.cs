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

    /// <summary>Parameter group for WD14 Tagger controls.</summary>
    public static T2IParamGroup WD14TaggerGroup;

    /// <summary>WD14 tagger model to use for tag generation.</summary>
    public static T2IRegisteredParam<string> ModelParam;

    /// <summary>Confidence threshold (0.0-1.0) for including a tag.</summary>
    public static T2IRegisteredParam<double> ThresholdParam;

    /// <summary>Comma-separated list of tags to exclude from the output.</summary>
    public static T2IRegisteredParam<string> FilterTagsParam;

    /// <summary>How generated tags are inserted into the prompt (replace, prepend, or append).</summary>
    public static T2IRegisteredParam<string> InsertModeParam;

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
        string model = context.Input.Get(ModelParam, WD14TaggerAPI.DefaultModelId);
        float threshold = (float)context.Input.Get(ThresholdParam, (double)WD14TaggerAPI.DefaultThreshold);
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

        WD14TaggerGroup = new("WD14 Tagger", Toggles: false, Open: false, OrderPriority: 100, Description: "Settings for WD14 image tagging (Generate Tags button and <wd14tagger> prompt tag).");
        ModelParam = T2IParamTypes.Register<string>(new(
            Name: "[WD14 Tagger] Model",
            Description: "WD14 tagger model to use for tag generation.",
            Default: "SmilingWolf/wd-eva02-large-tagger-v3",
            GetValues: _ => [
                "SmilingWolf/wd-eva02-large-tagger-v3///WD EVA02 Large v3 (default)",
                "SmilingWolf/wd-vit-large-tagger-v3///WD ViT Large v3",
                "SmilingWolf/wd-vit-tagger-v3///WD ViT v3",
                "SmilingWolf/wd-swinv2-tagger-v3///WD SwinV2 v3",
                "SmilingWolf/wd-convnext-tagger-v3///WD ConvNext v3",
                "SmilingWolf/wd-v1-4-swinv2-tagger-v2///WD SwinV2 v2",
                "SmilingWolf/wd-v1-4-vit-tagger-v2///WD ViT v2",
                "SmilingWolf/wd-v1-4-convnext-tagger-v2///WD ConvNext v2",
                "deepghs/pixai-tagger-v0.9-onnx///PixAI Tagger v0.9"
            ],
            Group: WD14TaggerGroup,
            HideFromMetadata: true,
            IntentionalUnused: true,
            OrderPriority: 1
        ));
        ThresholdParam = T2IParamTypes.Register<double>(new(
            Name: "[WD14 Tagger] Threshold",
            Description: "Confidence threshold (0.0–1.0) for including a tag. Tags below this score are excluded.",
            Default: "0.35",
            Min: 0,
            Max: 1,
            Step: 0.05,
            Group: WD14TaggerGroup,
            ViewType: ParamViewType.SLIDER,
            HideFromMetadata: true,
            IntentionalUnused: true,
            OrderPriority: 2
        ));
        FilterTagsParam = T2IParamTypes.Register<string>(new(
            Name: "[WD14 Tagger] Filter Tags",
            Description: "Comma-separated list of tags to exclude from the output.",
            Default: "",
            Group: WD14TaggerGroup,
            HideFromMetadata: true,
            IntentionalUnused: true,
            OrderPriority: 3
        ));
        InsertModeParam = T2IParamTypes.Register<string>(new(
            Name: "[WD14 Tagger] Insert Mode",
            Description: "How generated tags are inserted into the prompt: replace the entire prompt, prepend before it, or append after it.",
            Default: "replace",
            GetValues: _ => ["replace", "prepend", "append"],
            Group: WD14TaggerGroup,
            HideFromMetadata: true,
            IntentionalUnused: true,
            OrderPriority: 4
        ));

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
