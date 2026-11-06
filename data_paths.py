"""Shared nuScenes location and portable paths stored in annotation files."""
import os


DEFAULT_DATA_ROOT = os.environ.get(
    "NUSCENES_DATA_ROOT", "/mnt/workspace/dataSet/nuscene"
)


def resolve_path(root, path):
    """Resolve nuScenes sensor/map paths under the selected dataset root.

    Annotation files may contain relative paths, data/nuscenes prefixes, or
    absolute paths copied from another server. Canonical nuScenes directories
    follow root in every case; unrelated absolute paths keep their meaning.
    """
    normalized = os.fspath(path).replace("\\", "/")
    parts = normalized.split("/")
    for index, part in enumerate(parts):
        if part in ("samples", "sweeps", "maps"):
            return os.path.join(os.fspath(root), *parts[index:])
    if os.path.isabs(normalized):
        return normalized
    for prefix in ("./data/nuscenes/", "data/nuscenes/"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    return os.path.join(os.fspath(root), normalized.lstrip("./"))
