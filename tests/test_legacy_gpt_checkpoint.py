import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from scripts.generate_lm import GenerateLm
from uer.model_builder import build_model
from uer.model_loader import load_model
from uer.opts import model_opts
from uer.trainer import init_model


WORD = "embedding.word.embedding.weight"
POSITION = "embedding.pos.embedding.weight"
HEAD = "target.lm.output_layer.weight"
BIAS = "target.lm.output_layer.bias"
LEGACY = {
    WORD: "embedding.word_embedding.weight",
    POSITION: "embedding.position_embedding.weight",
    HEAD: "target.output_layer.weight",
    BIAS: "target.output_layer.bias",
}


def tiny_gpt_args(**overrides):
    parser = argparse.ArgumentParser()
    model_opts(parser)
    args = parser.parse_args([])
    config_path = Path(__file__).resolve().parents[1] / "models/gpt2/distil_config.json"
    with config_path.open(encoding="utf-8") as config:
        vars(args).update(json.load(config))
    vars(args).update(emb_size=8, hidden_size=8, feedforward_size=16,
                      heads_num=2, layers_num=1, max_seq_length=8, dropout=0,
                      has_lmtarget_bias=True, tokenizer=SimpleNamespace(vocab=range(11)))
    vars(args).update(overrides)
    return args


class LegacyGptCheckpointTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.checkpoint = Path(directory.name) / "gpt.bin"

    def save(self, state):
        torch.save(state, self.checkpoint)
        return self.checkpoint

    def assert_weights(self, model, expected):
        actual = model.state_dict()
        for key, tensor in expected.items():
            with self.subTest(key=key):
                self.assertTrue(torch.equal(actual[key], tensor), key)

    def test_generation_loads_all_four_legacy_tensors(self):
        args = tiny_gpt_args()
        expected_model = GenerateLm(args).eval()
        expected = expected_model.state_dict()
        checkpoint = self.save({LEGACY.get(key, key): value for key, value in expected.items()})

        # The wrapper should be returned intact, with weights loaded into its module.
        model = torch.nn.DataParallel(GenerateLm(args).eval())
        self.assertIs(load_model(model, checkpoint), model)
        self.assert_weights(model.module, expected)
        with torch.no_grad():
            src = torch.tensor([[1, 2, 3]])
            seg = torch.ones_like(src)
            self.assertTrue(torch.equal(model(src, seg), expected_model(src, seg)))

    def test_generation_keeps_explicit_current_keys(self):
        args = tiny_gpt_args()
        model = GenerateLm(args)
        expected = {key: torch.full_like(model.state_dict()[key], i)
                    for i, key in enumerate(LEGACY, 1)}
        mixed = dict(expected)
        mixed.update({old: torch.full_like(expected[key], -1) for key, old in LEGACY.items()})
        self.assert_weights(load_model(model, self.save(mixed)), expected)

    def test_pretraining_initialization_loads_tied_legacy_weights(self):
        args = tiny_gpt_args()
        expected = build_model(args).state_dict()
        args.pretrained_model_path = self.save({
            LEGACY.get(key, key): value for key, value in expected.items()
        })

        model = init_model(args)
        self.assertIs(model.embedding.word.embedding.weight, model.target.lm.output_layer.weight)
        self.assert_weights(model, expected)

    def test_pretraining_prefers_either_current_tied_key(self):
        args = tiny_gpt_args()
        weight = build_model(args).embedding.word.embedding.weight
        current = torch.full_like(weight, 1)
        legacy = torch.full_like(weight, -1)
        for canonical, other in ((WORD, HEAD), (HEAD, WORD)):
            with self.subTest(canonical=canonical):
                args.pretrained_model_path = self.save({canonical: current, LEGACY[other]: legacy})
                self.assert_weights(init_model(args), {WORD: current, HEAD: current})

    def test_current_checkpoint_is_unchanged(self):
        args = tiny_gpt_args(has_lmtarget_bias=False)
        expected = GenerateLm(args).state_dict()
        self.assert_weights(load_model(GenerateLm(args), self.save(expected)), expected)

    def test_noncausal_lm_does_not_receive_gpt_aliases(self):
        model = GenerateLm(tiny_gpt_args(mask="fully_visible"))
        expected = {key: model.state_dict()[key].clone() for key in LEGACY}
        aliases = {old: expected[key] + 100 for key, old in LEGACY.items()}
        self.assert_weights(load_model(model, self.save(aliases)), expected)


if __name__ == "__main__":
    unittest.main()
