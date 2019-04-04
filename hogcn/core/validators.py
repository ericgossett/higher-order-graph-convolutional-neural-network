from .exceptions import ConfigMissingARequiredKey

def config_validator(config):
    """Checks if the config dict contains the required keys."""
    required_keys = [
        'name',
        'batch_size',
        'num_epochs',
        'iterations_per_epoch',
        'learning_rate',
        'summary_dir',
        'save_dir',
        'saver_max_to_keep'
    ]
    for key in required_keys:
        if key not in config:
            raise ConfigMissingARequiredKey(
                'They key \'{}\' is missing from config dict'.format(key)
            )
