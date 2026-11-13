"""Small, immutable records shared by the audit and split interfaces."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageRecord:
    """A dataset-relative POSIX path, folder label and SHA-1 content group."""

    path: str
    label: str
    group_id: str


@dataclass(frozen=True)
class SplitIndices:
    """Disjoint index selections into an input sequence of ImageRecord objects."""

    train: tuple[int, ...]
    dev: tuple[int, ...]


@dataclass(frozen=True)
class ManifestRow:
    """Portable CSV row; split is train, dev or test."""

    path: str
    label: str
    group_id: str
    split: str
