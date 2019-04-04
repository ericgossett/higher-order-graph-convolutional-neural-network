
class SaverNotInitialized(Exception):
    """The models saver was not initialized."""
    pass

class ConfigMissingARequiredKey(Exception):
    """A required key is missing from the config dict."""
    pass