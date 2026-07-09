using System.IO;
using System.Globalization;
using FreneticUtilities.FreneticExtensions;
using Newtonsoft.Json.Linq;
using SwarmUI.Accounts;
using SwarmUI.Builtin_ComfyUIBackend;
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

    /// <summary>Available model IDs for prompt-tag explicit model selection.</summary>
    public static readonly string[] AvailableModelIds =
    [
        "SmilingWolf/wd-eva02-large-tagger-v3",
        "SmilingWolf/wd-vit-large-tagger-v3",
        "SmilingWolf/wd-vit-tagger-v3",
        "SmilingWolf/wd-swinv2-tagger-v3",
        "SmilingWolf/wd-convnext-tagger-v3",
        "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
        "SmilingWolf/wd-v1-4-vit-tagger-v2",
        "SmilingWolf/wd-v1-4-convnext-tagger-v2",
        "deepghs/pixai-tagger-v0.9-onnx",
        "fancyfeast/joytag",
        "Camais03/camie-tagger",
        "Camais03/camie-tagger-v2",
        "lodestones/taggerine",
        "animetimm/eva02_large_patch14_448.dbv4-full",
        "animetimm/convnextv2_huge.dbv4-full",
        "animetimm/caformer_b36.dbv4-full",
        "animetimm/swinv2_base_window8_256.dbv4-full",
        "animetimm/vit_base_patch16_224.dbv4-full",
        "animetimm/mobilenetv3_large_150d.dbv4-full"
    ];

    /// <summary>Available dropdown values for the WD14 tagger model parameter.</summary>
    public static readonly string[] AvailableModelValues =
    [
        "SmilingWolf/wd-eva02-large-tagger-v3///WD EVA02 Large v3 (default)",
        "SmilingWolf/wd-vit-large-tagger-v3///WD ViT Large v3",
        "SmilingWolf/wd-vit-tagger-v3///WD ViT v3",
        "SmilingWolf/wd-swinv2-tagger-v3///WD SwinV2 v3",
        "SmilingWolf/wd-convnext-tagger-v3///WD ConvNext v3",
        "SmilingWolf/wd-v1-4-swinv2-tagger-v2///WD SwinV2 v2",
        "SmilingWolf/wd-v1-4-vit-tagger-v2///WD ViT v2",
        "SmilingWolf/wd-v1-4-convnext-tagger-v2///WD ConvNext v2",
        "deepghs/pixai-tagger-v0.9-onnx///PixAI Tagger v0.9",
        "fancyfeast/joytag///JoyTag",
        "Camais03/camie-tagger///Camie Tagger v1",
        "Camais03/camie-tagger-v2///Camie Tagger v2",
        "lodestones/taggerine///Taggerine (DINOv3 ViT-H/16+)",
        "animetimm/eva02_large_patch14_448.dbv4-full///AnimeTimm EVA02 Large v4 (gated)",
        "animetimm/convnextv2_huge.dbv4-full///AnimeTimm ConvNeXtV2 Huge v4 (gated)",
        "animetimm/caformer_b36.dbv4-full///AnimeTimm CAFormer B36 v4 (gated)",
        "animetimm/swinv2_base_window8_256.dbv4-full///AnimeTimm SwinV2 Base v4 (gated)",
        "animetimm/vit_base_patch16_224.dbv4-full///AnimeTimm ViT Base v4 (gated)",
        "animetimm/mobilenetv3_large_150d.dbv4-full///AnimeTimm MobileNetV3 Large v4 (gated)"
    ];

    /// <summary>Parameter group for WD14 Tagger controls.</summary>
    public static T2IParamGroup WD14TaggerGroup;

    /// <summary>WD14 tagger model to use for tag generation.</summary>
    public static T2IRegisteredParam<string> ModelParam;

    /// <summary>Confidence threshold (0.0-1.0) for including general tags.</summary>
    public static T2IRegisteredParam<double> GeneralThresholdParam;

    /// <summary>Confidence threshold (0.0-1.0) for including character tags.</summary>
    public static T2IRegisteredParam<double> CharacterThresholdParam;

    /// <summary>Comma-separated tag rules to exclude or substitute tags in the output.</summary>
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

    /// <summary>Resolves an optional model override from a prompt tag like &lt;wd14tagger:model-id&gt;.</summary>
    private static string ResolvePromptTagModel(string requestedModel, T2IPromptHandling.PromptTagContext context)
    {
        string defaultModel = context.Input.Get(ModelParam, WD14TaggerAPI.DefaultModelId);
        if (string.IsNullOrWhiteSpace(requestedModel))
        {
            return defaultModel;
        }
        string matched = T2IParamTypes.GetBestModelInList(requestedModel.Trim(), AvailableModelIds);
        if (matched is null)
        {
            context.TrackWarning($"WD14Tagger prompt tag requested model '{requestedModel}', but that model is not recognized. Using '{defaultModel}' instead.");
            return defaultModel;
        }
        return matched;
    }

    /// <summary>Parses comma-separated prompt tag arguments while tolerating surrounding whitespace.</summary>
    private static string[] ParsePromptTagArgs(string data)
    {
        if (string.IsNullOrWhiteSpace(data))
        {
            return [];
        }
        string[] parts = data.Split(',');
        for (int i = 0; i < parts.Length; i++)
        {
            parts[i] = parts[i].Trim();
        }
        return parts;
    }

    /// <summary>Parses a prompt tag threshold override and falls back to the provided default on invalid input.</summary>
    private static float ResolvePromptTagThreshold(string rawValue, string thresholdName, float defaultValue, T2IPromptHandling.PromptTagContext context)
    {
        if (string.IsNullOrWhiteSpace(rawValue))
        {
            return defaultValue;
        }
        if (!float.TryParse(rawValue, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed))
        {
            context.TrackWarning($"WD14Tagger prompt tag requested {thresholdName} '{rawValue}', but it is not a valid number. Using the parameter value instead.");
            return defaultValue;
        }
        if (parsed < 0 || parsed > 1)
        {
            context.TrackWarning($"WD14Tagger prompt tag requested {thresholdName} '{rawValue}', but it must be between 0 and 1. Using the parameter value instead.");
            return defaultValue;
        }
        return parsed;
    }

    /// <summary>Resolves the effective prompt-tag settings from optional model and threshold positional arguments.</summary>
    private static void ResolvePromptTagSettings(string data, T2IPromptHandling.PromptTagContext context, out string model, out float generalThreshold, out float characterThreshold)
    {
        float defaultGeneralThreshold = context.Input.TryGet(GeneralThresholdParam, out double generalThreshVal) ? (float)generalThreshVal : -1f;
        float defaultCharacterThreshold = context.Input.TryGet(CharacterThresholdParam, out double charThreshVal) ? (float)charThreshVal : -1f;
        string[] args = ParsePromptTagArgs(data);
        if (args.Length > 3)
        {
            context.TrackWarning("WD14Tagger prompt tag received extra positional arguments. Expected '<wd14tagger:model,general_threshold,character_threshold>'. Extra values were ignored.");
        }
        string requestedModel = args.Length > 0 ? args[0] : null;
        model = ResolvePromptTagModel(requestedModel, context);
        generalThreshold = args.Length > 1 ? ResolvePromptTagThreshold(args[1], "General Threshold", defaultGeneralThreshold, context) : defaultGeneralThreshold;
        characterThreshold = args.Length > 2 ? ResolvePromptTagThreshold(args[2], "Character Threshold", defaultCharacterThreshold, context) : defaultCharacterThreshold;
    }

    /// <summary>Builds a cache key for prompt-tag results based on the effective tagger settings.</summary>
    private static string BuildPromptTagCacheKey(string model, float generalThreshold, float characterThreshold, string filterTags)
    {
        return $"{model}|{generalThreshold}|{characterThreshold}|{filterTags}";
    }

    /// <summary>
    /// Generates WD14 tags for use inside a prompt tag expansion.
    /// Returns an empty string when no usable image source is available.
    /// </summary>
    private static string GeneratePromptTagTags(string data, T2IPromptHandling.PromptTagContext context)
    {
        if (context?.Input is null)
        {
            return "";
        }
        string filterTags = context.Input.Get(FilterTagsParam, "");
        ResolvePromptTagSettings(data, context, out string model, out float generalThreshold, out float characterThreshold);
        string cacheKey = BuildPromptTagCacheKey(model, generalThreshold, characterThreshold, filterTags);
        Dictionary<string, string> cache = context.Input.ExtraMeta.GetOrCreate(PromptTagCacheKey, () => new Dictionary<string, string>()) as Dictionary<string, string>;
        if (cache.TryGetValue(cacheKey, out string cached))
        {
            return cached;
        }
        Image source = GetPromptTagImageSource(context.Input);
        if (source is null)
        {
            context.TrackWarning("WD14Tagger prompt tag '<wd14tagger>' was used, but no init image or prompt image is available to tag.");
            cache[cacheKey] = "";
            return "";
        }
        JObject result;
        try
        {
            result = WD14TaggerAPI.WD14TaggerGenerateTags(context.Input.SourceSession, source.AsBase64, model,
                generalThreshold,
                characterThreshold,
                filterTags).GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            context.TrackWarning($"WD14Tagger prompt tag failed: {ex.Message}");
            cache[cacheKey] = "";
            return "";
        }
        bool success = result?["success"]?.Value<bool>() ?? false;
        if (!success)
        {
            string err = result?["error"]?.Value<string>() ?? "Unknown WD14Tagger error.";
            context.TrackWarning($"WD14Tagger prompt tag failed: {err}");
            cache[cacheKey] = "";
            return "";
        }
        string tags = result?["tags"]?.Value<string>() ?? "";
        if (string.IsNullOrWhiteSpace(tags))
        {
            context.TrackWarning("WD14Tagger prompt tag produced no tags above the current threshold.");
            cache[cacheKey] = "";
            return "";
        }
        cache[cacheKey] = tags;
        return tags;
    }

    public override void OnPreInit()
    {
        ExtFolder = FilePath;
        ScriptFiles.Add("Assets/wd14tagger.js");
        StyleSheetFiles.Add("Assets/wd14tagger.css");
        ComfyUISelfStartBackend.CustomNodePaths.Add(Path.GetFullPath($"{FilePath}/ComfyNodes"));

        WD14TaggerGroup = new("WD14 Tagger", Toggles: false, Open: false, OrderPriority: 100, Description: "Settings for WD14 image tagging (Generate Tags button and <wd14tagger> prompt tag).");
        ModelParam = T2IParamTypes.Register<string>(new(
            Name: "[WD14 Tagger] Model",
            Description: "Image tagger model to use for tag generation. Note: 'animetimm' models are gated on HuggingFace \u2013 you must accept each model's terms on its HuggingFace page and be logged in (huggingface-cli login or an HF_TOKEN environment variable) before they can download.",
            Default: "SmilingWolf/wd-eva02-large-tagger-v3",
            GetValues: _ => [.. AvailableModelValues],
            Group: WD14TaggerGroup,
            IntentionalUnused: true,
            OrderPriority: 1
        ));
        GeneralThresholdParam = T2IParamTypes.Register<double>(new(
            Name: "[WD14 Tagger] General Threshold",
            Description: "Confidence threshold (0.0–1.0) for including general tags. Tags below this score are excluded. Uncheck to disable general tags entirely.",
            Default: "0.35",
            Min: 0,
            Max: 1,
            Step: 0.05,
            Group: WD14TaggerGroup,
            ViewType: ParamViewType.SLIDER,
            Toggleable: true,
            IntentionalUnused: true,
            OrderPriority: 2
        ));
        CharacterThresholdParam = T2IParamTypes.Register<double>(new(
            Name: "[WD14 Tagger] Character Threshold",
            Description: "Confidence threshold (0.0–1.0) for including character tags. Tags below this score are excluded. Uncheck to disable character tags entirely.",
            Default: "0.85",
            Min: 0,
            Max: 1,
            Step: 0.05,
            Group: WD14TaggerGroup,
            ViewType: ParamViewType.SLIDER,
            Toggleable: true,
            IntentionalUnused: true,
            OrderPriority: 3
        ));
        FilterTagsParam = T2IParamTypes.Register<string>(new(
            Name: "[WD14 Tagger] Filter Tags",
            Description: "Comma-separated tag rules. Use 'tag' to exclude, 'source:target' to replace an exact tag, or wildcard forms like 'tag*:new', '*tag:new', and '*tag*:new' to substitute only the matching phrase on word boundaries. Non-alphanumeric separators like spaces, '+', and '-' count as boundaries.",
            Default: "",
            Group: WD14TaggerGroup,
            IntentionalUnused: true,
            OrderPriority: 6
        ));
        InsertModeParam = T2IParamTypes.Register<string>(new(
            Name: "[WD14 Tagger] Insert Mode",
            Description: "How generated tags are inserted into the prompt: replace the entire prompt, prepend before it, or append after it. Note that the presence of a '<wd14tagger>' prompt tag will always take precedence.",
            Default: "replace",
            GetValues: _ => ["replace", "prepend", "append"],
            Group: WD14TaggerGroup,
            IntentionalUnused: true,
            OrderPriority: 7
        ));

        // Register <wd14tagger> prompt token. It expands to WD14 tags generated
        // from the request's init image (or first prompt image) before generation.
        T2IPromptHandling.PromptTagBasicProcessors["wd14tagger"] = (data, context) =>
        {
            return GeneratePromptTagTags(data, context);
        };
        T2IPromptHandling.PromptTagLengthEstimators["wd14tagger"] = (data, context) =>
        {
            if (context?.Input is null)
            {
                return "";
            }
            string filterTags = context.Input.Get(FilterTagsParam, "");
            ResolvePromptTagSettings(data, context, out string model, out float generalThreshold, out float characterThreshold);
            string cacheKey = BuildPromptTagCacheKey(model, generalThreshold, characterThreshold, filterTags);
            if (context.Input.ExtraMeta.TryGetValue(PromptTagCacheKey, out object existing)
                && existing is Dictionary<string, string> cache
                && cache.TryGetValue(cacheKey, out string cached))
            {
                return cached;
            }
            return "";
        };
    }

    public override void OnInit()
    {
        WD14TaggerAPI.Register();
    }
}
