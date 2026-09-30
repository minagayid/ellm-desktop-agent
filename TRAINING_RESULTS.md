# Local training results

**Outcome: rejected. The candidate is not suitable for real desktop tasks and was not promoted.**

On 2026-09-30, the project continued the public `minagayid/ELLM` LoRA adapter locally for two epochs using 220 synthetic training examples and 88 synthetic development examples. Training loss fell from 1.3983 to 1.0394, and development loss fell from 0.9544 to 0.6858. No data or model weights were uploaded.

The frozen held-out split contains 88 synthetic examples, including 8 critical clarification/refusal cases. Results from the one baseline-versus-candidate evaluation:

| Measure | Existing ELLM adapter | Local candidate | Gate |
| --- | ---: | ---: | ---: |
| Valid JSON plans | 0/88 (0.0%) | 30/88 (34.1%) | At least 95% |
| Correct action | 0/88 (0.0%) | 12/88 (13.6%) | At least 80% |
| Exact tool plan, including arguments | 0/80 (0.0%) | 5/80 (6.25%) | At least 60% |
| Unsafe choices on critical cases | 8/8 | 8/8 | 0 |

The candidate improved over the original adapter on the aggregate action and tool-plan measures, but failed every minimum threshold and chose an unsafe action on all 8 critical cases. The candidate remains a local, unpromoted artifact; it is excluded from version control. The runtime's approval checks reduce what a model can execute, but they do not make incorrect plans reliable.

This small, templated benchmark is useful for rejecting this candidate, not for claiming general performance. It does not assess factual summaries or real-world robustness. The current result does not support using either adapter as a dependable desktop agent.
