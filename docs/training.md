# Training with rsl_rl

PPO from [rsl_rl](https://github.com/leggedrobotics/rsl_rl) trains any `WaterEnv` through one wrapper.

## Train

```bash
uv run --extra rl examples/train_rsl_rl.py --num-envs 1024 --iterations 300
```

The example trains an International Moth to hold its starting ride height, with the reward defined at the top of the script. Add `--device mps` or `--device cuda` to run more envs.

## Checkpoints

Each run writes to `logs/rsl_rl/moth/<timestamp>/`: `model_0.pt`, `model_50.pt`, … (one every 50 iterations) plus the last, and TensorBoard events.

```bash
uv run --extra rl tensorboard --logdir logs/rsl_rl
```

## Watch a checkpoint

```bash
uv run --extra rl --extra viz examples/train_rsl_rl.py --play logs/rsl_rl/moth/<timestamp>/model_299.pt
```

This draws 4 envs with the deterministic policy. `--viewer null` runs it without a window.

## Use it on your own env

```python
from rsl_rl.runners import OnPolicyRunner
from watergym.rl.rsl_rl import RslRlVecEnv

runner = OnPolicyRunner(RslRlVecEnv(env), train_cfg, log_dir, "cpu")
runner.learn(300)
```

`train_cfg` is the dict in the example script. The wrapper gives rsl_rl one observation group, `"policy"`, which is `env.observe()`.

How the episode ends maps to rsl_rl like this:

| WaterEnv | rsl_rl |
|---|---|
| `terminated` (capsized) | `dones = 1`, `time_outs = 0` |
| `truncated` (episode length reached) | `dones = 1`, `time_outs = 1`, so PPO bootstraps the return |
| finished envs reset inside `step()` | the returned observation is the new episode's first |

Actions are unbounded Gaussian samples. `WaterEnv` clamps them to [-1, 1].
