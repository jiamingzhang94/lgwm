"""Training checks on synthetic inputs."""
import unittest

import numpy as np
import torch

from lgwm.data.dataset import Batch
from lgwm.data.transitions import MixedSources, QuotaSampler
from lgwm.models.lgwm import LGWM
from lgwm.train.trainer import consumed_batches, param_groups, rng_state, rollout_choice, set_rng_state

CFG = {"rollout": {"enable_after": 5, "p": 0.3, "horizon": [2, 3], "tf_anneal": [1.0, 0.5]}}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Sized(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


def synthetic(batch=2, horizon=None):
    g = torch.Generator().manual_seed(0)
    shape = (batch, 3, 512, 256)
    action = dict(a_type=torch.tensor([0, 3])[:batch], a_coord=torch.rand(batch, 2, generator=g),
                  a_has_coord=torch.tensor([True, False])[:batch], a_coord2=torch.zeros(batch, 2),
                  a_has_coord2=torch.zeros(batch, dtype=torch.bool), a_text=torch.randn(batch, 384, generator=g),
                  a_has_text=torch.tensor([True, False])[:batch], a_dir=torch.tensor([-1, 1])[:batch],
                  src_id=torch.tensor([0, 1])[:batch])
    if horizon is None:
        return Batch(img_t=torch.randint(0, 256, shape, dtype=torch.uint8, generator=g),
                     img_t1=torch.randint(0, 256, shape, dtype=torch.uint8, generator=g), **action).to(DEVICE)
    frames = torch.randint(0, 256, (batch, horizon + 1, 3, 512, 256), dtype=torch.uint8, generator=g)
    return {"frames": frames.to(DEVICE), **{k: v.unsqueeze(1).repeat(1, horizon, *([1] * (v.ndim - 1))).to(DEVICE)
                                            for k, v in action.items()}}


class ScheduleTests(unittest.TestCase):
    def test_rollout_choice_is_a_function_of_step(self):
        self.assertEqual([rollout_choice(s, CFG) for s in range(40)], [rollout_choice(s, CFG) for s in range(40)])
        self.assertFalse(any(rollout_choice(s, CFG)[0] for s in range(5)))
        self.assertTrue(any(rollout_choice(s, CFG)[0] for s in range(5, 60)))

    def test_consumed_batches_partition_steps(self):
        counts = consumed_batches(200, CFG)
        self.assertEqual(sum(counts.values()), 200)
        manual = sum(1 for s in range(200) if rollout_choice(s, CFG)[0] and rollout_choice(s, CFG)[1] == 3)
        self.assertEqual(counts[3], manual)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA RNG state needs a GPU")
    def test_rng_state_survives_weights_only_checkpoint(self):
        import io
        buffer = io.BytesIO()
        torch.save({"rng": rng_state(torch.device("cuda", 0))}, buffer)
        expected = np.random.random(3)
        buffer.seek(0)
        state = torch.load(buffer, map_location="cuda", weights_only=True)["rng"]
        set_rng_state(state, torch.device("cuda", 0))
        np.testing.assert_array_equal(np.random.random(3), expected)

    def test_sampler_resume_offset_continues_the_same_stream(self):
        mixed = MixedSources({"a": Sized(7), "b": Sized(3)})
        full = list(QuotaSampler(mixed, {"a": 0.4, "b": 0.6}, 64, rank=1, world=2))
        tail = list(QuotaSampler(mixed, {"a": 0.4, "b": 0.6}, 64, rank=1, world=2, start=10))
        self.assertEqual(full[10:], tail)
        self.assertEqual(len(full), 32)


class ModelStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = LGWM("vit_small_patch14_dinov2.lvd142m", False, pred_depth=2, pred_heads=6).to(DEVICE)
        cls.kwargs = dict(tau_c=torch.tensor([3.16, 20.7], device=DEVICE), lambda_hi=8.0, invdyn_w=0.5)

    def setUp(self):
        self.model.zero_grad(set_to_none=True)

    def test_param_groups_cover_trainable_parameters_once(self):
        groups = param_groups(self.model, 0.05)
        ids = [id(p) for g in groups for p in g["params"]]
        trainable = [id(p) for p in self.model.parameters() if p.requires_grad]
        self.assertEqual(sorted(ids), sorted(trainable))
        self.assertAlmostEqual(groups[0]["lr_scale"], 0.75 ** (len(self.model.encoder.vit.blocks) + 1))

    def test_single_step_gradients_and_frozen_target(self):
        loss, stats = self.model(synthetic(), mode="single", **self.kwargs)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name in ("encoder", "action_encoder", "predictor", "invdyn"):
            grads = [p.grad for p in getattr(self.model, name).parameters() if p.grad is not None]
            self.assertTrue(grads and any(g.abs().sum() > 0 for g in grads), name)
        self.assertTrue(all(p.grad is None for p in self.model.target.parameters()))

    def test_rollout_window_step(self):
        loss, stats = self.model(synthetic(horizon=3), mode="window", horizon=3, tf_prob=0.5,
                                 rollout_rand=[0.9, 0.1, 0.4], **self.kwargs)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats["rollout_H"], 3.0)
        self.assertTrue(any(p.grad is not None for p in self.model.predictor.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.invdyn.parameters()))

    def test_ema_update_moves_target_toward_online(self):
        before = [p.detach().clone() for p in self.model.target.parameters()]
        with torch.no_grad():
            for p in self.model.encoder.parameters():
                p.add_(0.01)
        self.model.target.update(self.model.encoder, 0.9)
        moved = [not torch.equal(a, b) for a, b in zip(before, self.model.target.parameters())]
        self.assertTrue(all(moved))
        with torch.no_grad():
            for p in self.model.encoder.parameters():
                p.sub_(0.01)


if __name__ == "__main__":
    unittest.main()
