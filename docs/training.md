# Training with rsl_rl

PPO from [rsl_rl](https://github.com/leggedrobotics/rsl_rl) trains any `WaterEnv` through one wrapper.

## Train

```bash
uv run --extra rl examples/train_rsl_rl.py --num-envs 1024 --iterations 300
```

The example trains an International Moth to hold its starting ride height, with the reward defined at the top of the script. Add `--device mps` or `--device cuda` to run more envs.

## Train on the benchmark task

```bash
uv run --extra rl examples/train_rsl_rl.py --task RideControl-Moth-v0 --hs 0.3 --num-envs 1024 --iterations 300
```

With `--task` the reward, terminations and 20 s timeout are the task's (see [benchmark.md](benchmark.md)). The policy reads the six sensor observations; the critic reads the 16 privileged ones, which add the true ride height, the foil depth, ventilation and the sea state. `--hs` and `--tp` set the head sea every env trains in, and `--seed` (default 0) seeds the waves and the network. The task's evaluation seeds are refused.

Score a checkpoint on the benchmark grid next to the wand and zero baselines. A policy name ending in `.pt` loads the actor of an rsl_rl checkpoint:

```bash
uv run --extra rl --with matplotlib python benchmarks/ride_control_sweep.py --quick --policy wand zero logs/rsl_rl/RideControl-Moth-v0/<timestamp>/model_299.pt
```

## Checkpoints

Each run writes to `logs/rsl_rl/moth/<timestamp>/` (or `logs/rsl_rl/<task>/<timestamp>/`): `model_0.pt`, `model_50.pt`, … (one every 50 iterations) plus the last, and TensorBoard events.

```bash
uv run --extra rl tensorboard --logdir logs/rsl_rl
```

## Watch a checkpoint

```bash
uv run --extra rl --extra viz examples/train_rsl_rl.py --play logs/rsl_rl/moth/<timestamp>/model_299.pt
```

This draws 4 envs with the deterministic policy. `--viewer null` runs it without a window. For a task checkpoint, pass the same `--task`; the printed ride-height RMS is then the error from the task's 0.6 m target.

## Use it on your own env

```python
from rsl_rl.runners import OnPolicyRunner
from watergym.rl.rsl_rl import TRAIN_CFG, RslRlVecEnv

runner = OnPolicyRunner(RslRlVecEnv(env), TRAIN_CFG, log_dir, "cpu")
runner.learn(300)
```

`TRAIN_CFG` lives in `watergym.rl.rsl_rl`. The wrapper gives rsl_rl two observation groups. For a plain `WaterEnv` both `"policy"` and `"critic"` are `env.observe()`. For a task, `"policy"` is the task's observations and `"critic"` is `info["privileged_obs"]`, and `TRAIN_CFG["obs_groups"]` routes them to the actor and the critic:

```python
"obs_groups": {"actor": ["policy"], "critic": ["critic"]}
```

How the episode ends maps to rsl_rl like this:

| WaterEnv | rsl_rl |
|---|---|
| `terminated` (capsized, or the task's crash) | `dones = 1`, `time_outs = 0` |
| `truncated` (episode length reached) | `dones = 1`, `time_outs = 1`, so PPO bootstraps the return |
| finished envs reset inside `step()` | the returned observation is the new episode's first |

Actions are unbounded Gaussian samples. `WaterEnv` clamps them to [-1, 1].
