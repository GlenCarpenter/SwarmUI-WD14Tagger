using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;
using SwarmUI.Accounts;
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

    /// <summary>Default confidence threshold for tag inclusion.</summary>
    public const float DefaultThreshold = 0.35f;

    /// <summary>Registers all API calls for this extension.</summary>
    public static void Register()
    {
        API.RegisterAPICall(WD14TaggerGenerateTags, true, WD14TaggerPermissions.PermGenerateTags);
    }

    /// <summary>Allowed characters in a HuggingFace repo ID (namespace/repo-name).</summary>
    private static readonly Regex SafeRepoIdPattern = new(@"^[A-Za-z0-9_\-/\.]+$", RegexOptions.Compiled);

    /// <summary>Maximum allowed byte length for the filterTags string.</summary>
    private const int MaxFilterTagsLength = 4096;

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

    /// <summary>Guards one-time dependency installation per process lifetime.</summary>
    private static volatile bool _dependenciesEnsured = false;
    private static readonly SemaphoreSlim _depLock = new(1, 1);

    /// <summary>
    /// Runs <c>pip install -r requirements.txt</c> once per process lifetime using the resolved
    /// Python executable, so all required packages are present before the first inference call.
    /// </summary>
    private static async Task EnsureDependenciesAsync()
    {
        if (_dependenciesEnsured) return;
        await _depLock.WaitAsync();
        try
        {
            if (_dependenciesEnsured) return;
            string requirementsPath = Path.GetFullPath($"{WD14TaggerExtension.ExtFolder}requirements.txt");
            if (!File.Exists(requirementsPath))
            {
                Logs.Warning("WD14Tagger: requirements.txt not found, skipping dependency check.");
                _dependenciesEnsured = true;
                return;
            }
            Logs.Info("WD14Tagger: Checking/installing Python dependencies...");
            ProcessStartInfo psi = new()
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            };
            // Mirror PythonLaunchHelper.LaunchGeneric's Python discovery so pip installs into the correct isolated environment
            if (File.Exists("./dlbackend/comfy/python_embeded/python.exe"))
            {
                psi.FileName = Path.GetFullPath("./dlbackend/comfy/python_embeded/python.exe");
                psi.WorkingDirectory = Path.GetFullPath("./dlbackend/comfy/");
                psi.Environment["PATH"] = PythonLaunchHelper.ReworkPythonPaths(Path.GetFullPath("./dlbackend/comfy/python_embeded"));
            }
            else if (File.Exists("./dlbackend/ComfyUI/venv/bin/python"))
            {
                psi.FileName = Path.GetFullPath("./dlbackend/ComfyUI/venv/bin/python");
                psi.Environment["PATH"] = PythonLaunchHelper.ReworkPythonPaths(Path.GetFullPath("./dlbackend/ComfyUI/venv/bin"));
            }
            else
            {
                psi.FileName = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "python" : "python3";
            }
            PythonLaunchHelper.CleanEnvironmentOfPythonMess(psi, "WD14Tagger: ");
            psi.ArgumentList.Add("-m");
            psi.ArgumentList.Add("pip");
            psi.ArgumentList.Add("install");
            psi.ArgumentList.Add("--quiet");
            psi.ArgumentList.Add("-r");
            psi.ArgumentList.Add(requirementsPath);
            using Process process = Process.Start(psi);
            await process.WaitForExitAsync();
            if (process.ExitCode != 0)
            {
                string stderr = await process.StandardError.ReadToEndAsync();
                Logs.Warning($"WD14Tagger: pip install exited with code {process.ExitCode}. {stderr.Trim()}");
            }
            else
            {
                Logs.Info("WD14Tagger: Python dependencies ready.");
            }
            _dependenciesEnsured = true;
        }
        finally
        {
            _depLock.Release();
        }
    }

    /// <summary>Generates WD14 tags for the provided base64-encoded image using the specified model.</summary>
    /// <param name="session">The calling user session.</param>
    /// <param name="imageBase64">Base64-encoded image data (PNG/JPG/WEBP).</param>
    /// <param name="modelId">HuggingFace repo ID of the tagger model.</param>
    /// <param name="threshold">Confidence threshold (0.0-1.0) for including a tag.</param>
    /// <param name="filterTags">Comma-separated list of tags to exclude from the output.</param>
    public static async Task<JObject> WD14TaggerGenerateTags(
        Session session,
        string imageBase64,
        string modelId = DefaultModelId,
        float threshold = DefaultThreshold,
        string filterTags = "")
    {
        await EnsureDependenciesAsync();

        // Validate inputs to prevent injection
        if (string.IsNullOrWhiteSpace(imageBase64))
        {
            return new JObject { ["success"] = false, ["error"] = "No image data provided." };
        }
        if (string.IsNullOrWhiteSpace(modelId) || !SafeRepoIdPattern.IsMatch(modelId))
        {
            return new JObject { ["success"] = false, ["error"] = "Invalid model ID format." };
        }
        if (threshold < 0f || threshold > 1f)
        {
            return new JObject { ["success"] = false, ["error"] = "Threshold must be between 0.0 and 1.0." };
        }
        filterTags = SanitizeFilterTags(filterTags);
        // Build a set of lowercased filtered tags for fast lookup
        HashSet<string> filteredTagSet = new(StringComparer.OrdinalIgnoreCase);
        if (!string.IsNullOrWhiteSpace(filterTags))
        {
            foreach (string ft in filterTags.Split(','))
            {
                string trimmed = ft.Trim();
                if (!string.IsNullOrEmpty(trimmed))
                {
                    filteredTagSet.Add(trimmed);
                }
            }
        }

        string tempImagePath = null;
        try
        {
            byte[] imageBytes = Convert.FromBase64String(imageBase64);
            // Use a temp file with .png extension so PIL can auto-detect the format
            tempImagePath = Path.Combine(Path.GetTempPath(), $"wd14tagger_{Guid.NewGuid():N}.png");
            await File.WriteAllBytesAsync(tempImagePath, imageBytes);

            string scriptPath = Path.GetFullPath($"{WD14TaggerExtension.ExtFolder}wd14_tagger_inference.py");
            string modelDir = Path.GetFullPath("Models/wd14_tagger");

            using Process process = PythonLaunchHelper.LaunchGeneric(scriptPath, false,
            [
                "--image_path", tempImagePath,
                "--repo_id", modelId,
                "--model_dir", modelDir,
                "--threshold", threshold.ToString("F2", System.Globalization.CultureInfo.InvariantCulture)
            ]);
            Task<string> stdoutTask = process.StandardOutput.ReadToEndAsync();
            Task<string> stderrTask = process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync(CancellationToken.None);
            string stdout = (await stdoutTask).Trim();
            string stderr = (await stderrTask).Trim();

            // The script writes intermediate JSON lines: {"info": ...}, {"error": ...}, {"progress": ...}, then {"success": ...}
            string resultLine = null;
            foreach (string line in stdout.Split('\n'))
            {
                string trimmed = line.Trim();
                if (!trimmed.StartsWith("{")) { continue; }
                JObject parsed;
                try { parsed = JObject.Parse(trimmed); }
                catch { continue; }
                if (parsed.ContainsKey("info"))
                {
                    Logs.Info($"WD14Tagger: {parsed["info"].Value<string>()}");
                }
                else if (parsed.ContainsKey("error"))
                {
                    Logs.Error($"WD14Tagger: {parsed["error"].Value<string>()}");
                }
                else if (parsed.ContainsKey("success"))
                {
                    resultLine = trimmed;
                }
            }
            if (!string.IsNullOrWhiteSpace(stderr))
            {
                Logs.Warning($"WD14Tagger stderr: {stderr}");
            }

            if (string.IsNullOrWhiteSpace(resultLine))
            {
                Logs.Warning($"WD14Tagger: Python script produced no result. stdout='{stdout}'");
                return new JObject { ["success"] = false, ["error"] = "Tagger produced no output. Check server logs for details." };
            }

            JObject result = JObject.Parse(resultLine);
            // Apply tag filtering if any filters were specified
            if (filteredTagSet.Count > 0 && result["success"]?.Value<bool>() == true)
            {
                string rawTags = result["tags"]?.Value<string>() ?? "";
                IEnumerable<string> filteredTags = rawTags
                    .Split(',', StringSplitOptions.RemoveEmptyEntries)
                    .Select(t => t.Trim())
                    .Where(t => !filteredTagSet.Contains(t));
                result["tags"] = string.Join(", ", filteredTags);
            }
            return result;
        }
        catch (FormatException)
        {
            return new JObject { ["success"] = false, ["error"] = "Invalid base64 image data." };
        }
        catch (Exception ex)
        {
            Logs.Error($"WD14Tagger error: {ex.Message}");
            return new JObject { ["success"] = false, ["error"] = ex.Message };
        }
        finally
        {
            if (tempImagePath != null && File.Exists(tempImagePath))
            {
                try { File.Delete(tempImagePath); } catch { /* best-effort cleanup */ }
            }
        }
    }

}
