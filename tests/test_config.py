import pytest
import yaml

def test_config_structure():
    with open("config/settings.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    assert 'database' in cfg
    assert 'training' in cfg
    assert 'artifacts' in cfg
    assert 'search_tuning' in cfg

def test_config_required_fields():
    with open("config/settings.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    assert cfg['database']['project_id'] is not None
    assert cfg['training']['model_name'] is not None
    assert cfg['training']['extra_feature_dim'] == 2

def test_alpha_range():
    with open("config/settings.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    alpha = cfg['search_tuning']['alpha']
    assert 0 <= alpha <= 1
