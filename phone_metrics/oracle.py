"""Oracle-boundary phone classification accuracy.

Scope item (2): phone-recognition accuracy when the ground-truth segment
boundaries are *given*, so recognition collapses to assigning one phone label
per GT segment. This module only **scores**: it pairs the ground-truth
utterances with a caller-supplied label per GT segment and reports exact-match
accuracy, both micro (over all scored segments) and macro over languages.

Everything model-specific stays with the caller. The center-frame feature
lookup, the projection that turns a frame into a predicted label, any
closure/release labeling scheme, closure merging, and vocabulary restriction
all happen upstream and arrive here as a flat list of predictions. A segment
the model cannot score (e.g. a diphthong with no single target vector, or a
phone outside a language's restricted vocabulary) is passed as ``None``; the
required ``none_is`` argument decides whether such a segment is ``"ignored"``
(dropped from the denominator) or counted as ``"incorrect"`` (in the
denominator, never correct). Silence segments (``ipa_label == "_"``) are
intrinsically unscorable and are always dropped regardless of the prediction.

Both notebook metrics reduce to this call:

* TIMIT: ``label="raw"`` (score against the merged TIMIT phone), one language
  (``"eng"``), so ``macro_language == accuracy``.
* VoxAngeles: ``label="ipa"``, many languages, ``macro_language`` averages the
  per-language accuracies.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .datasets import Utterance
from .timit import SILENCE


@dataclass
class OracleAccuracy:
    """Result of :func:`oracle_phone_accuracy`.

    ``per_language`` maps each language to its ``(correct, total)`` counts over
    scored (non-silence, non-``None``) segments.
    """

    correct: int
    total: int
    per_language: dict[str, tuple[int, int]]

    @property
    def accuracy(self) -> float:
        """Micro accuracy: correct over all scored segments."""
        return self.correct / self.total

    @property
    def macro_language(self) -> float:
        """Mean of the per-language accuracies (unweighted by segment count)."""
        accs = [c / t for c, t in self.per_language.values()]
        return sum(accs) / len(accs)


def oracle_phone_accuracy(
    utterances: Sequence[Utterance],
    predictions: Sequence[str | None],
    none_is: str,
    label: str = "ipa",
) -> OracleAccuracy:
    """Score oracle-boundary phone classification.

    ``predictions`` is one label (or ``None``) per ground-truth segment,
    flattened across ``utterances`` in order — utterance by utterance, then
    segment by segment. Its length must equal the total number of GT segments;
    a mismatch raises ``ValueError`` rather than silently truncating.

    ``none_is`` governs ``None`` predictions on non-silence segments:
    ``"ignored"`` drops them from the denominator, ``"incorrect"`` counts them
    as scored-and-wrong.

    ``label`` selects which segment field to score against: ``"ipa"`` for
    :attr:`Seg.ipa_label`, ``"raw"`` for :attr:`Seg.raw_label`.
    """
    if none_is not in ("ignored", "incorrect"):
        raise ValueError(f"none_is must be 'ignored' or 'incorrect', got {none_is!r}")
    if label not in ("ipa", "raw"):
        raise ValueError(f"label must be 'ipa' or 'raw', got {label!r}")
    attr = f"{label}_label"

    segs = [(utt, seg) for utt in utterances for seg in utt.segments]
    if len(predictions) != len(segs):
        raise ValueError(
            f"got {len(predictions)} predictions for {len(segs)} ground-truth segments"
        )

    per_language: dict[str, list[int]] = {}
    for (utt, seg), pred in zip(segs, predictions):
        if seg.ipa_label == SILENCE:
            continue
        if pred is None and none_is == "ignored":
            continue
        counts = per_language.setdefault(utt.language, [0, 0])
        counts[0] += int(pred is not None and getattr(seg, attr) == pred)
        counts[1] += 1

    if not per_language:
        raise ValueError("no scorable segments (all silence or None predictions)")

    per_lang = {lang: (c, t) for lang, (c, t) in per_language.items()}
    correct = sum(c for c, _ in per_lang.values())
    total = sum(t for _, t in per_lang.values())
    return OracleAccuracy(correct=correct, total=total, per_language=per_lang)
