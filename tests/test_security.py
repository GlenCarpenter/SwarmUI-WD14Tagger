"""Security-boundary tests for model downloads and the Comfy bridge node."""

import math
import os
import tempfile
import unittest
from unittest import mock

from ComfyNodes.WD14TaggerNode import (
    _validated_model_directory,
    _validated_output_path,
    _validated_threshold,
)
from wd14tagger import common
from wd14tagger.models import MODEL_REVISIONS, require_supported_model


class ModelPolicyTests(unittest.TestCase):
    def test_revisions_are_immutable_commit_hashes(self):
        self.assertEqual(19, len(MODEL_REVISIONS))
        for revision in MODEL_REVISIONS.values():
            self.assertEqual(40, len(revision))
            self.assertTrue(all(character in "0123456789abcdef" for character in revision))

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(ValueError):
            require_supported_model("attacker/unreviewed-model")

    def test_download_uses_reviewed_revision(self):
        repo_id = "SmilingWolf/wd-eva02-large-tagger-v3"
        with tempfile.TemporaryDirectory() as model_dir, mock.patch.object(common, "hf_hub_download") as download:
            common.ensure_hf_files(repo_id, model_dir, ["model.onnx"], "test")
        download.assert_called_once_with(
            repo_id=repo_id,
            filename="model.onnx",
            revision=MODEL_REVISIONS[repo_id],
            local_dir=model_dir,
        )


class ComfyBoundaryTests(unittest.TestCase):
    def test_output_path_is_limited_to_random_temporary_name(self):
        accepted = os.path.join(tempfile.gettempdir(), f"wd14tagger_{'a' * 32}.txt")
        self.assertEqual(os.path.normcase(os.path.realpath(accepted)), _validated_output_path(accepted))
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            dotnet_temp = os.path.join(local_app_data, "Temp", f"wd14tagger_{'b' * 32}.txt")
            self.assertEqual(os.path.normcase(os.path.realpath(dotnet_temp)), _validated_output_path(dotnet_temp))
        with self.assertRaises(ValueError):
            _validated_output_path(os.path.join(tempfile.gettempdir(), "not-random.txt"))
        with self.assertRaises(ValueError):
            _validated_output_path(os.path.join(os.getcwd(), f"wd14tagger_{'a' * 32}.txt"))

    def test_model_directory_requires_expected_layout(self):
        repo_id = "SmilingWolf/wd-eva02-large-tagger-v3"
        accepted = os.path.join("X:/models", "wd14_tagger", repo_id.replace("/", "_"))
        self.assertEqual(os.path.realpath(os.path.abspath(accepted)), _validated_model_directory(accepted, repo_id))
        with self.assertRaises(ValueError):
            _validated_model_directory("X:/models/unrelated", repo_id)

    def test_threshold_validation_rejects_non_finite_values(self):
        self.assertEqual(0.0, _validated_threshold(0, "threshold"))
        self.assertEqual(-1.0, _validated_threshold(-1, "threshold"))
        for invalid in (math.nan, math.inf, -math.inf, -0.1, 1.1):
            with self.assertRaises(ValueError):
                _validated_threshold(invalid, "threshold")


if __name__ == "__main__":
    unittest.main()
