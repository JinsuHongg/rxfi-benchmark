import sys
from pathlib import Path
import pytest
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'training'))
from train_vit_classifier import (
    class_mapping_for_representation,
    encoded_label,
    flare_class_to_band,
    validate_class_order,
)

@pytest.mark.parametrize('raw,expected', [('FQ','FQ'),('A1.0','A'),('B3.4','B'),('C9.9','C'),('M2.1','M'),('X1.7','X')])
def test_flare_class_to_band(raw, expected):
    assert flare_class_to_band(raw) == expected

@pytest.mark.parametrize('raw', ['', 'Q1.0', 'M', 'FQ1', 'C1.x'])
def test_flare_class_to_band_rejects_malformed_values(raw):
    with pytest.raises(ValueError): flare_class_to_band(raw)

def test_ordinal_band_mapping_is_fixed():
    assert class_mapping_for_representation('ordinal_band') == {'FQ/A':0,'B':1,'C':2,'M':3,'X':4}

@pytest.mark.parametrize('raw', ['FQ', 'A1.0', 'A9.9'])
def test_ordinal_band_encoding_combines_fq_and_a(raw):
    assert encoded_label(raw, {'target_representation': 'ordinal_band'}) == 'FQ/A'

def test_ordinal_band_encoding_uses_five_ordered_classes():
    config = {'target_representation': 'ordinal_band'}
    assert [encoded_label(raw, config) for raw in ('B3.4', 'C9.9', 'M2.1', 'X1.7')] == ['B', 'C', 'M', 'X']

def test_project2_five_class_configs_share_the_new_mapping_and_output_namespace():
    root = Path(__file__).resolve().parents[1]
    channels = ('hmi_m', 'aia131', 'aia193', 'all13')
    for channel in channels:
        path = root / 'configs' / 'project2' / f'vit_small_224_{channel}_max_flare_class_ordinal5.yaml'
        config = yaml.safe_load(path.read_text())
        assert config['class_order'] == ['FQ/A', 'B', 'C', 'M', 'X']
        assert config['target_representation'] == 'ordinal_band'
        assert config['experiment_name'].endswith('_ordinal5')
        assert config['output_dir'].endswith('_ordinal5')

def test_class_order_validation_rejects_legacy_six_class_config():
    mapping = class_mapping_for_representation('ordinal_band')
    with pytest.raises(ValueError, match='class_order'):
        validate_class_order({'class_order': ['FQ', 'A', 'B', 'C', 'M', 'X']}, mapping)

def test_class_order_validation_accepts_new_five_class_config():
    mapping = class_mapping_for_representation('ordinal_band')
    validate_class_order({'class_order': ['FQ/A', 'B', 'C', 'M', 'X']}, mapping)

def test_legacy_six_class_representation_remains_explicit_and_reproducible():
    assert class_mapping_for_representation('ordinal_band6') == {'FQ':0, 'A':1, 'B':2, 'C':3, 'M':4, 'X':5}
    assert encoded_label('A1.0', {'target_representation': 'ordinal_band6'}) == 'A'
