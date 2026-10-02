"""Regression checks for input validation and exports."""
import importlib.util
from pathlib import Path
import tempfile

import numpy as np
import pytest
from lgwm.data.transitions import TextTable

ROOT = Path(__file__).resolve().parents[1]


def tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def owned_tmp():
    (ROOT / 'outputs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as directory:
        yield Path(directory)


def test_text_table_rejects_missing_nonempty_text(owned_tmp):
    path = owned_tmp / 'text.npz'
    emb = np.zeros((2, 384), np.float32); emb[1, 0] = 1
    np.savez(path, vocab=np.array(['', 'known']), emb=emb)
    table = TextTable(path)
    assert table.ids([None, '', 'known']).tolist() == [0, 0, 1]
    with pytest.raises(ValueError, match='missing 1 nonempty'):
        table.ids(['unknown'])


@pytest.mark.parametrize('vocab,emb', [
    (['', ''], np.zeros((2, 384))),
    ([''], np.zeros((1, 383))),
    (['', 'bad'], np.full((2, 384), np.nan)),
    ([], np.zeros((0, 384))),
])
def test_invalid_text_table_fails_before_loading_batches(owned_tmp, vocab, emb):
    path = owned_tmp / 'text.npz'
    np.savez(path, vocab=np.array(vocab), emb=emb)
    with pytest.raises(ValueError):
        TextTable(path)


def test_stage_a_export_rejects_incomplete_checkpoints():
    module = tool('export_stage_a_init')
    for state in ({'complete': False}, {}, {'complete': 1}):
        with pytest.raises(ValueError, match='incomplete'):
            module.check_complete(state)
    module.check_complete({'complete': True, 'step': 10, 'epochs': 1, 'frames': 100})
