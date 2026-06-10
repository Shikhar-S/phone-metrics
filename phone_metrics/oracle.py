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
denominator, never correct). At most one leading and one trailing silence
segment (``ipa_label == "_"``) are stripped independently from reference and
prediction before scoring; internal silence segments are scored normally.

Both notebook metrics reduce to this call:

* TIMIT: ``label="raw"`` (score against the merged TIMIT phone), one language
  (``"eng"``), so ``macro_language == accuracy``.
* VoxAngeles: ``label="ipa"``, many languages, ``macro_language`` averages the
  per-language accuracies.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .datasets import Utterance
from .timit import SILENCE


@dataclass
class OracleAccuracy:
    """Result of :func:`oracle_phone_accuracy`.

    ``per_language`` maps each language to its ``(correct, total)`` counts over
    scored (non-silence, non-``None``) segments, in sorted language order.
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
        return float(np.mean(accs))


def _strip_edge_silence(seq: Sequence, *, is_silence) -> list:
    seq = list(seq)
    if seq and is_silence(seq[0]):
        seq = seq[1:]
    if seq and is_silence(seq[-1]):
        seq = seq[:-1]
    return seq


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

    total_segments = sum(len(utt.segments) for utt in utterances)
    if len(predictions) != total_segments:
        raise ValueError(
            f"got {len(predictions)} predictions for {total_segments} ground-truth segments"
        )

    # Single Python pass to pull the scored field, the silence marker, the
    # prediction, and the language off each Seg; everything after this is
    # vectorized.
    gt, ipa, lang, pred = [], [], [], []
    offset = 0
    for utt in utterances:
        utt_predictions = predictions[offset : offset + len(utt.segments)]
        offset += len(utt.segments)

        ref = [(getattr(seg, attr), seg.ipa_label, utt.language) for seg in utt.segments]
        ref = _strip_edge_silence(ref, is_silence=lambda item: item[1] == SILENCE)
        pred_labels = _strip_edge_silence(
            utt_predictions,
            is_silence=lambda item: item == SILENCE,
        )
        if len(pred_labels) != len(ref):
            raise ValueError(
                f"after edge-silence stripping, {utt.audio_path} has "
                f"{len(pred_labels)} predictions for {len(ref)} reference segments"
            )

        for (gt_label, ipa_label, language), pred_label in zip(ref, pred_labels, strict=True):
            gt.append(gt_label)
            ipa.append(ipa_label)
            lang.append(utt.language)
            pred.append(pred_label)

    gt = np.array(gt, dtype=object)
    ipa = np.array(ipa, dtype=object)
    lang = np.array(lang, dtype=object)
    pred = np.array(pred, dtype=object)

    is_none = np.fromiter((p is None for p in pred), dtype=bool, count=len(pred))
    scored = np.ones(len(pred), dtype=bool) if none_is == "incorrect" else ~is_none
    if not scored.any():
        raise ValueError("no scorable segments (all silence or None predictions)")

    correct = scored & ~is_none & (gt == pred)

    # Per-language (correct, total) via bincount over the scored subset.
    sel = np.flatnonzero(scored)
    langs, inv = np.unique(lang[sel], return_inverse=True)
    totals = np.bincount(inv, minlength=langs.size)
    corrects = np.bincount(inv[correct[sel]], minlength=langs.size)
    per_language = {
        str(lg): (int(c), int(t)) for lg, c, t in zip(langs, corrects, totals)
    }
    return OracleAccuracy(
        correct=int(corrects.sum()),
        total=int(totals.sum()),
        per_language=per_language,
    )
