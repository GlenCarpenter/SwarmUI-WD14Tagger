"""Allowlisted WD14 Tagger model repositories and reviewed revisions."""


MODEL_REVISIONS = {
    "SmilingWolf/wd-eva02-large-tagger-v3": "b25b82a03f7282e41aa2f257a52c7583b710bd1c",
    "SmilingWolf/wd-vit-large-tagger-v3": "ae469aa2e4706a3af08d3673cf73a11d1add314c",
    "SmilingWolf/wd-vit-tagger-v3": "7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f",
    "SmilingWolf/wd-swinv2-tagger-v3": "627aef95638667ddcaa3ac8ae625e88ea5b02f51",
    "SmilingWolf/wd-convnext-tagger-v3": "d39e46de298d27340111b64965e20b8185c407e6",
    "SmilingWolf/wd-v1-4-swinv2-tagger-v2": "cdb0c7fdc70646f0af29c6f80f8df564344a69b6",
    "SmilingWolf/wd-v1-4-vit-tagger-v2": "1f3f3e8ae769634e31e1ef696df11ec37493e4f2",
    "SmilingWolf/wd-v1-4-convnext-tagger-v2": "4b34d1b07bdd8e95494072648960b8a6adcbc0ff",
    "deepghs/pixai-tagger-v0.9-onnx": "d8cf666911a2c3d10d586d7823259192313c7eb7",
    "fancyfeast/joytag": "6b7f16331a6ccf0fdce37d5a9564715f6e772b22",
    "Camais03/camie-tagger": "9afb19b5e21e7916f599edda74eb373cac63ba24",
    "Camais03/camie-tagger-v2": "7d40c1b85b86ab4f607b2caf26b1b50c99db743e",
    "lodestones/taggerine": "ba76a13a17a9d1844298a68c142c5aae6191505d",
    "animetimm/eva02_large_patch14_448.dbv4-full": "7f11ec9fdb54dfbfd7cd7fad2ecc452a25b57acb",
    "animetimm/convnextv2_huge.dbv4-full": "18177355d1448a69bafb0410a0608e144f714e8b",
    "animetimm/caformer_b36.dbv4-full": "aac0699c88553d50eb673b41a81d8222936b22b2",
    "animetimm/swinv2_base_window8_256.dbv4-full": "e09b265f59b8cec01101f01ec380a7a6dd4db733",
    "animetimm/vit_base_patch16_224.dbv4-full": "055ad45c6c2531561e59b054ddef216ba204e407",
    "animetimm/mobilenetv3_large_150d.dbv4-full": "89c7ef44d18398714f61fbc2629740e51f012b89",
}

SUPPORTED_MODEL_IDS = frozenset(MODEL_REVISIONS)


def require_supported_model(repo_id: str) -> str:
    """Return a normalized allowlisted model ID or raise a user-facing error."""
    normalized = (repo_id or "").strip()
    if normalized not in SUPPORTED_MODEL_IDS:
        raise ValueError(f"Unsupported WD14 Tagger model ID: {normalized or '<empty>'}")
    return normalized


def model_revision(repo_id: str) -> str:
    """Return the immutable, reviewed HuggingFace revision for a model."""
    return MODEL_REVISIONS[require_supported_model(repo_id)]
