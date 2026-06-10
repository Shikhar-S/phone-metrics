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
from pathlib import Path

import numpy as np

from .timit import SILENCE, Seg, merge_adjacent_silence, timit_segments

# VoxAngeles TextGrid labels treated as silence.
VOX_SILENCE_LABELS = frozenset({"", "sp", "sil", "sil+"})


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
            ipa = SILENCE if label in VOX_SILENCE_LABELS else label
            segs.append(Seg(float(entry.start), float(entry.end), label, ipa))
        segs = merge_adjacent_silence(segs)
        utts.append(Utterance(str(grid_path.with_suffix(".wav")), language, "test", segs))
    return utts
