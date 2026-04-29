using SwarmUI.Core;
using SwarmUI.Utils;

namespace GlenCarpenter.Extensions.WD14TaggerExtension;

/// <summary>WD14Tagger - image tagging using SmilingWolf WD14 ONNX models from HuggingFace.</summary>
public class WD14TaggerExtension : Extension
{
    /// <summary>Path to this extension's directory, stored for use by the static API class.</summary>
    public static string ExtFolder;

    public override void OnPreInit()
    {
        ExtFolder = FilePath;
        ScriptFiles.Add("Assets/wd14tagger.js");
        StyleSheetFiles.Add("Assets/wd14tagger.css");
    }

    public override void OnInit()
    {
        Logs.Info("WD14Tagger extension initialized.");
        WD14TaggerAPI.Register();
    }
}
