# src/registry.py

class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def register(self, name=None):
        def _register_decorator(cls):
            key = name if name is not None else cls.__name__

            if key in self._module_dict:
                raise KeyError(f"{key} is already registered in {self._name}")

            self._module_dict[key] = cls
            return cls

        return _register_decorator

    def get(self, key):
        if key not in self._module_dict:
            raise KeyError(f"'{key}' is not registered in {self._name}. Available: {self._module_dict.keys()}")
        return self._module_dict[key]


MODEL_REGISTRY = Registry("MODEL")
TRAINER_REGISTRY = Registry("TRAINER")
TESTER_REGISTRY = Registry("TESTER")


