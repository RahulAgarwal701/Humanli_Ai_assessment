"""Read-only image inventory, full decode checks and file-content hashing."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
from typing import Any, Sequence
import warnings

from PIL import Image, ImageOps

from data.records import ImageRecord


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuditResult:
    """Valid records plus a serializable report, including any audit failures."""

    train: tuple[ImageRecord, ...]
    test: tuple[ImageRecord, ...]
    report: dict[str, Any]


def file_sha1(path: Path) -> str:
    """Hash file bytes in bounded chunks; hashes identify exact duplicate files."""
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(
    root: Path,
    train_folder: str,
    test_folder: str,
    class_names: Sequence[str],
) -> AuditResult:
    """Inspect both sources without modifying files or using test data for splitting.

    Verify headers, reopen and fully decode each image, then record EXIF-adjusted
    dimensions. Hidden metadata files are explicitly listed and ignored. Unexpected
    visible files, invalid layouts and mixed-label content groups fail the audit.
    """
    pools: dict[str, list[ImageRecord]] = {train_folder: [], test_folder: []}
    report: dict[str, Any] = {
        "hash_algorithm": "sha1",
        "sources": {},
        "errors": [],
        "ignored_hidden_entries": [],
    }
    all_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for source in pools:
        source_dir = root / source
        counts: Counter[str] = Counter()
        valid_counts: Counter[str] = Counter()
        formats: Counter[str] = Counter()
        sizes: Counter[str] = Counter()
        modes: Counter[str] = Counter()
        report["sources"][source] = {
            "class_counts": counts,
            "readable_class_counts": valid_counts,
            "formats": formats,
            "sizes": sizes,
            "modes": modes,
        }
        if not source_dir.is_dir():
            report["errors"].append({"path": source, "error": "Source folder is missing"})
            continue
        for entry in sorted(source_dir.iterdir()):
            relative = entry.relative_to(root).as_posix()
            if entry.name.startswith("."):
                report["ignored_hidden_entries"].append(relative)
            elif not entry.is_dir() or entry.name not in class_names:
                report["errors"].append({"path": relative, "error": "Unexpected source entry"})
        for label in class_names:
            class_dir = source_dir / label
            counts[label] = valid_counts[label] = 0
            if not class_dir.is_dir():
                report["errors"].append({"path": f"{source}/{label}", "error": "Class folder is missing"})
                continue
            for path in sorted(class_dir.rglob("*")):
                relative = path.relative_to(root).as_posix()
                if any(part.startswith(".") for part in path.relative_to(class_dir).parts):
                    if path.is_file():
                        report["ignored_hidden_entries"].append(relative)
                    continue
                if not path.is_file():
                    continue
                counts[label] += 1
                try:
                    group_id = file_sha1(path)
                    all_groups[group_id].append({"path": relative, "label": label, "source": source})
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", Image.DecompressionBombWarning)
                        with Image.open(path) as image:
                            image.verify()
                        with Image.open(path) as image:
                            image.load()
                            image_format = image.format or "UNKNOWN"
                            oriented = ImageOps.exif_transpose(image)
                            formats[image_format] += 1
                            sizes[f"{oriented.width}x{oriented.height}"] += 1
                            modes[oriented.mode] += 1
                    pools[source].append(ImageRecord(relative, label, group_id))
                    valid_counts[label] += 1
                except (OSError, ValueError, SyntaxError, Image.DecompressionBombError,
                        Image.DecompressionBombWarning) as exc:
                    report["errors"].append({"path": relative, "error": f"{type(exc).__name__}: {exc}"})
            if counts[label] == 0:
                report["errors"].append({"path": f"{source}/{label}", "error": "Class folder is empty"})
        LOGGER.info("Audited %s: %s", source, dict(valid_counts))
    report["duplicate_groups"] = [
        {"group_id": group_id, "files": members}
        for group_id, members in sorted(all_groups.items()) if len(members) > 1
    ]
    report["extra_duplicate_copies"] = sum(len(g["files"]) - 1 for g in report["duplicate_groups"])
    report["conflicting_label_groups"] = [
        group for group in report["duplicate_groups"]
        if len({member["label"] for member in group["files"]}) > 1
    ]
    train_hashes = {record.group_id for record in pools[train_folder]}
    test_hashes = {record.group_id for record in pools[test_folder]}
    report["train_test_overlap_hashes"] = sorted(train_hashes & test_hashes)
    report["passed"] = not (
        report["errors"] or report["conflicting_label_groups"] or report["train_test_overlap_hashes"]
    )
    return AuditResult(tuple(pools[train_folder]), tuple(pools[test_folder]), report)
