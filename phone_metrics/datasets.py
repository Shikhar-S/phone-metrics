"""Ground-truth segmentation loaders for TIMIT and VoxAngeles.

These read the datasets *as distributed* — TIMIT ``.phn`` files and VoxAngeles
``.TextGrid`` files — rather than any pre-processed CSV copy. Each loader
returns a list of :class:`Utterance`, whose ``boundaries`` give the
ground-truth boundary times (seconds) for raw segmentation scoring.

Boundary extraction strips a single outer silence segment at each end, then
takes the inner segment starts plus the final inner end (identical to the
notebook's ``_boundary_secs``). TIMIT non-speech (``h#``/``pau``/``epi``) and
VoxAngeles empty/``sp``/``sil`` intervals are normalized to the silence token.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import panphon

from .timit import SILENCE, Seg, merge_adjacent_silence, timit_segments

# VoxAngeles TextGrid labels treated as silence.
VOX_SILENCE_LABELS = frozenset({"", "sp", "sil", "sil+"})

# Glyph confusables seen in the distributed transcriptions: not IPA, but a
# look-alike of a representable phone. ASCII "g" for IPA script "ɡ"; the Cyrillic
# ligature for Latin "æ". TODO: this belongs in an upstream IPA-normalization /
# confusables table (panphon), not hand-maintained here.
_HOMOGLYPHS = {"g": "ɡ", "ӕ": "æ"}


@lru_cache(maxsize=1)
def _feature_table() -> panphon.FeatureTable:
    return panphon.FeatureTable()


def tokenize_ipa(label: str) -> list[str]:
    """Split an IPA string into its component phones (``"aɪ" -> ["a", "ɪ"]``)."""
    return _feature_table().ipa_segs(label)


def canonical_ipa(label: str | None) -> str | None:
    """Map a ground-truth IPA label to its canonical, scorable form.

    Silence/``None`` pass through. A label panphon already knows is returned
    unchanged. Otherwise we apply the homoglyph fixes and re-tokenize: if the
    label reduces to exactly one known phone we return it (recovering diacritic
    noise like ``k̚``->``k`` and confusables like ``g``->``ɡ``), and otherwise we
    keep the label as-is so it stays unknown -- a genuine feature-space gap
    (``ʡ``) or a multi-target segment (``aɪ``, ``w͡a``) that single-phone
    recognition cannot produce, and which therefore counts as an error in the
    exact-match metrics.

    TODO: the single-phone recovery is lossy -- panphon drops contrastive marks
    it cannot represent (``k'``->``k``, ``d̪̤ʱ``->``d̪̤``). Fixing that properly
    belongs upstream in panphon's feature inventory, not here.
    """
    if label is None or label == SILENCE:
        return label
    ft = _feature_table()
    if ft.seg_known(label):
        return label
    label = "".join(_HOMOGLYPHS.get(c, c) for c in label)
    segs = ft.ipa_segs(label)
    if len(segs) == 1 and ft.seg_known(segs[0]):
        return segs[0]
    return label


@dataclass
class Utterance:
    """A single utterance's ground-truth segmentation."""

    audio_path: str
    language: str
    split: str
    segments: list[Seg]

    @property
    def boundaries(self) -> np.ndarray:
        """Ground-truth boundary times (seconds), outer silence stripped."""
        return boundary_secs(self.segments)


def boundary_secs(segments: list[Seg]) -> np.ndarray:
    """Inner segment starts + final inner end, after stripping outer silence."""
    lo, hi = 0, len(segments)
    while lo < hi and segments[lo].ipa_label == SILENCE:
        lo += 1
    while hi > lo and segments[hi - 1].ipa_label == SILENCE:
        hi -= 1
    inner = segments[lo:hi]
    if not inner:
        return np.array([], dtype=float)
    return np.array([s.start for s in inner] + [inner[-1].end], dtype=float)


def load_timit(
    root: str | Path, split: str = "test", merge_closures: bool = True
) -> list[Utterance]:
    """Load TIMIT ground-truth segmentation from the distributed tree.

    ``root`` is any directory above the ``TRAIN``/``TEST`` (or lowercase
    ``train``/``test``) split dirs — e.g. ``data/TIMIT``. ``split`` is
    ``"train"`` or ``"test"``. ``merge_closures`` is forwarded to
    :func:`phone_metrics.timit.timit_segments`.
    """
    root = Path(root)
    tag = split.upper()
    wav_paths = sorted(root.glob(f"**/{tag}/**/*.WAV")) or sorted(
        root.glob(f"**/{tag.lower()}/**/*.wav")
    )
    if not wav_paths:
        raise FileNotFoundError(f"No TIMIT {split} .wav files found under {root}")

    utts = []
    for wav_path in wav_paths:
        phn_path = next(
            (
                p
                for p in (wav_path.with_suffix(".PHN"), wav_path.with_suffix(".phn"))
                if p.exists()
            ),
            None,
        )
        if phn_path is None:
            raise FileNotFoundError(f"No .phn sibling for {wav_path}")
        segs = timit_segments(phn_path, merge_closures=merge_closures)
        for seg in segs:
            seg.ipa_label = canonical_ipa(seg.ipa_label)
        utts.append(Utterance(str(wav_path), "eng", split, segs))
    return utts


def load_voxangeles(root: str | Path) -> list[Utterance]:
    """Load VoxAngeles ground-truth segmentation from the distributed repo.

    ``root`` is the VoxAngeles repo root (e.g. ``data/voxangeles``); TextGrids
    are read from ``data/audited_aligned/**/*.TextGrid``. The per-utterance
    language is taken from the containing directory name.
    """
    import praatio.textgrid

    root = Path(root)
    grid_paths = sorted((root / "data/audited_aligned").glob("**/*.TextGrid"))
    if not grid_paths:
        raise FileNotFoundError(
            f"No VoxAngeles .TextGrid files found under {root}/data/audited_aligned"
        )

    utts = []
    for grid_path in grid_paths:
        grid = praatio.textgrid.openTextgrid(str(grid_path), includeEmptyIntervals=True)
        tier_name = next(x for x in grid.tierNames if x in ("phone", "phones", "Narrow"))
        language = grid_path.parent.name
        segs = []
        for entry in grid.getTier(tier_name).entries:
            label = (entry.label or "").strip()
            ipa = SILENCE if label in VOX_SILENCE_LABELS else canonical_ipa(label)
            segs.append(Seg(float(entry.start), float(entry.end), label, ipa))
        segs = merge_adjacent_silence(segs)
        utts.append(Utterance(str(grid_path.with_suffix(".wav")), language, "test", segs))
    return utts
