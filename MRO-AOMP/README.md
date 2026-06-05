# AOMP-OPSD

This is an independent experimental method directory inside MRO-SI. It does not
modify the original `mro_si` implementation; it imports MRO-SI utilities
read-only for prompt building, teacher scoring shape, and boxed-answer
verification.

## Method

AOMP-OPSD keeps the A-OMP two-step shell and changes only the two oracle
definitions:

```text
h_t = proxy OPSD token-level distillation direction
w_t = z_t - eta * h_t
g_t = outcome-audited OPSD direction at w_t
z_{t+1} = z_t - eta * g_t
```

The proxy channel is the cheap dense OPSD signal:

```text
L_proxy = sum_i sum_k KL(pi_T(. | s_i,k) || pi_S(. | s_i,k))
h_t = grad L_proxy(z_t)
```

The lookahead channel samples from `pi_w`, audits each full trajectory, and uses
that sequence outcome only as a trajectory-level calibration signal. It is not
split into token rewards.

```text
A_i = normalize_group(R_i - mean_group(R))

L_audit =
sum_i sigmoid(A_i / tau)  * sum_k beta_pos[i,k] * NLL_student[i,k]
+
sum_i sigmoid(-A_i / tau) * sum_k beta_neg[i,k] * KL_teacher_student[i,k]
```

Positive outcome trajectories reinforce the student's own successful behavior,
especially uncertain tokens. Negative outcome trajectories use teacher KL to
correct the student, especially where teacher/student disagreement is large,
the teacher is confident, and the prefix is still reliable.

The central claim is:

```text
Outcome signal is not a token reward;
it is a calibration signal for token-level distillation.
```

## Variants

- `vanilla_opsd`: no lookahead, no audit routing, existing OPSD-style loss.
- `outcome_weighted_opsd`: no lookahead; sequence outcome directly weights the OPSD token loss.
- `aomp_uniform`: AOMP two-step and outcome routing, but uniform token weights.
- `full_aomp_opsd`: lookahead, routing, student uncertainty, teacher reliability, and prefix reliability.

## Diagrams

- `aomp architecture.png`: presentation architecture figure.
- `docs/aomp_opsd_flow.svg`: editable method flow figure.

## Run

Original MRO-SI remains unchanged:

```bash
cd /Users/chichu/Desktop/code/h100/MRO-SI
bash scripts/run_mrosi_train_eval.sh
```

Vanilla OPSD baseline through this method directory:

```bash
cd /Users/chichu/Desktop/code/h100/MRO-SI/MRO-AOMP
AOMP_OPSD_VARIANT=vanilla_opsd bash scripts/run_aomp_opsd_train_eval.sh
```

Full AOMP-OPSD:

```bash
cd /Users/chichu/Desktop/code/h100/MRO-SI/MRO-AOMP
AOMP_OPSD_VARIANT=full_aomp_opsd bash scripts/run_aomp_opsd_train_eval.sh
```

Small dry-run example:

```bash
cd /Users/chichu/Desktop/code/h100/MRO-SI/MRO-AOMP
MAX_STEPS=1 SAVE_STEPS=1 PER_DEVICE_TRAIN_BATCH_SIZE=1 TRAIN_GRAD_ACCUM=1 \
AOMP_OPSD_GROUP_SIZE=2 RUN_EVAL=false \
bash scripts/run_aomp_opsd_train_eval.sh
```

## Approximations

- The prox step is implemented in trainable-parameter space as
  `p <- p - eta * grad`; it is not a strict policy KL trust-region prox.
- When temporary parameter swaps are unsafe, such as ZeRO-3 or FSDP in v1, the
  trainer falls back to recomputing the audited loss on `z_t` using the same
  rollout batch.
- The sequence-level verifier is a controlled-bias outcome-calibrated estimator,
  not an unbiased oracle for the exact theorem assumptions.

## Tests

```bash
cd /Users/chichu/Desktop/code/h100/MRO-SI/MRO-AOMP
python -m unittest discover -s tests -p 'test_aomp_opsd*.py'
```
