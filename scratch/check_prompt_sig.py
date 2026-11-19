import sys
sys.path.insert(0, ".")
import inspect
from generation.manifest import build_prompt_from_attributes
print(inspect.signature(build_prompt_from_attributes))
