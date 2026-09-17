import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_vit_classifier import flare_class_to_band, class_mapping_for_representation

@pytest.mark.parametrize('raw,expected', [('FQ','FQ'),('A1.0','A'),('B3.4','B'),('C9.9','C'),('M2.1','M'),('X1.7','X')])
def test_flare_class_to_band(raw, expected):
    assert flare_class_to_band(raw) == expected

@pytest.mark.parametrize('raw', ['', 'Q1.0', 'M', 'FQ1', 'C1.x'])
def test_flare_class_to_band_rejects_malformed_values(raw):
    with pytest.raises(ValueError): flare_class_to_band(raw)

def test_ordinal_band_mapping_is_fixed():
    assert class_mapping_for_representation('ordinal_band') == {'FQ':0,'A':1,'B':2,'C':3,'M':4,'X':5}
