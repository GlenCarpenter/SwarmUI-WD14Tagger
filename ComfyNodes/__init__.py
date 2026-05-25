import traceback

NODE_CLASS_MAPPINGS = {}

try:
    from . import WD14TaggerNode
    NODE_CLASS_MAPPINGS.update(WD14TaggerNode.NODE_CLASS_MAPPINGS)
except Exception:
    print("Error: [WD14Tagger] WD14TaggerNode not available")
    traceback.print_exc()