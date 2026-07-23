import os
import sys
import types
from typing import Any

# ==============================================================================
# DYNAMIC KAULDRON MOCK (To bypass Windows dependency issues)
# ==============================================================================
class DummyType:
    def __getitem__(self, item):
        return Any
    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return self

from flax import linen as nn
class FlaxIdentity(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x

class ShardingMock:
    REPLICATED = None
    def with_sharding_constraint(self, x, sharding=None):
        return x
    def device_put(self, x, sharding=None):
        return x

class DynamicMockModule(types.ModuleType):
    def __getattr__(self, name):
        if name == 'Identity':
            return FlaxIdentity
        if name == 'sharding':
            return ShardingMock()
        if name == 'nn':
            return sys.modules.get('kauldron.kd.nn')
        if name in ('ktyping', 'typing', 'kontext', 'kd'):
            return sys.modules.get(f'kauldron.{name}')
        return DummyType()

def init_mocks():
    """Registers dynamic mock modules in sys.modules and adds path dependencies."""
    # Register mocked modules in sys.modules
    kauldron_module = DynamicMockModule('kauldron')
    sys.modules['kauldron'] = kauldron_module
    sys.modules['kauldron.ktyping'] = DynamicMockModule('kauldron.ktyping')
    sys.modules['kauldron.typing'] = DynamicMockModule('kauldron.typing')
    sys.modules['kauldron.kontext'] = DynamicMockModule('kauldron.kontext')
    sys.modules['kauldron.kd'] = DynamicMockModule('kauldron.kd')
    sys.modules['kauldron.kd.sharding'] = ShardingMock()
    sys.modules['kauldron.kd.nn'] = DynamicMockModule('kauldron.kd.nn')
    sys.modules['kauldron.kd.nn'].Identity = FlaxIdentity
    
    # Locate and append the local gemma repository to python path
    # C:\Users\SEC\Downloads\gemma\oabf_engine\oabf\utils.py -> parent of oabf_engine is downloads/gemma
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gemma_repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    sys.path.append(os.path.join(gemma_repo_root, 'gemma'))
