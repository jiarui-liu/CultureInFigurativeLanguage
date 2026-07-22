"""Evaluation suite for the continued-pretrained cultural checkpoints.

Implements four Hindi language/culture/idiom benchmarks used to compare a
continued-pretrained (CPT) checkpoint against its untrained base model:

- ``mabl``        : MABL Hindi figurative-meaning inference (2-choice MC).
- ``milu``        : MILU Hindi cultural-knowledge exam QA (4-choice MC).
- ``global_piqa`` : Global PIQA Hindi cultural physical-commonsense (2-choice MC).
- ``idiomce``     : IdiomCE English->Hindi idiomatic translation (generation +
                    OpenAI LLM-as-judge).

The three multiple-choice tasks are scored with base-model log-likelihood (no
instruction following required); IdiomCE is a generation task scored by an
OpenAI judge model.

Note: :class:`culture.evaluation.scorer.HFModel` is intentionally *not* imported
here so that the torch-free parts (tasks, judge, compare_results) can be used
without a torch install. Import it directly:
``from culture.evaluation.scorer import HFModel``.
"""

__all__ = []
