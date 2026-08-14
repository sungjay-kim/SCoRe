"""Path resolution for the two dataset layouts this code can read.

The working tree keeps each fruit as a directory holding its own subfolders::

    <root>/<fruit_id>/orbbec/pointclouds/<frame_id>_preprocessed.ply
    <root>/<fruit_id>/laser/fruit.ply

The public release groups by kind instead, and names files by session date::

    <root>/processed/<sample_id>/<YYYYMMDD>.ply
    <root>/gt/<sample_id>/pointcloud.ply

Both are addressed through the helpers here rather than by counting path components, so
adding or moving a level cannot silently produce the wrong fruit id.
"""
import os
import re

RELEASE_PARTIAL_DIR = 'processed'
RELEASE_GT_DIR = 'gt'
RELEASE_GT_FILE = 'pointcloud.ply'

WORKING_PARTIAL_SUBDIR = os.path.join('orbbec', 'pointclouds')
WORKING_GT_SUBPATH = os.path.join('laser', 'fruit.ply')

# Working ids look like S12, P101, P3-1; release ids are A1, B7, C80.
FRUIT_ID_RE = re.compile(r'^[A-Z]+\d+(-\d+)?(_\d+)?$')


def is_release_layout(root):
    """True when root holds a published dataset rather than a working tree."""
    return os.path.isdir(os.path.join(root, RELEASE_PARTIAL_DIR)) and \
        os.path.isdir(os.path.join(root, RELEASE_GT_DIR))


def fruit_dirs(root):
    """Fruit directories available under root, whichever layout it uses."""
    base = os.path.join(root, RELEASE_PARTIAL_DIR) if is_release_layout(root) else root
    if not os.path.isdir(base):
        return []
    return sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))


def has_fruit(root, fruit_id):
    base = os.path.join(root, RELEASE_PARTIAL_DIR) if is_release_layout(root) else root
    return os.path.isdir(os.path.join(base, fruit_id))


def partial_dir(root, fruit_id):
    """Directory holding one fruit's per-session partial clouds."""
    if is_release_layout(root):
        return os.path.join(root, RELEASE_PARTIAL_DIR, fruit_id)
    return os.path.join(root, fruit_id, WORKING_PARTIAL_SUBDIR)


def gt_cloud(root, fruit_id):
    """The fruit's post-harvest reference cloud."""
    if is_release_layout(root):
        return os.path.join(root, RELEASE_GT_DIR, fruit_id, RELEASE_GT_FILE)
    return os.path.join(root, fruit_id, WORKING_GT_SUBPATH)


def sdf_samples(root, fruit_id):
    """DeepSDF sample archive for the fruit.

    The release does not ship these; they are regenerated from the reference cloud by
    data_preparation/deepsdf/prepare_deepsdf_training_data.py, which writes them beside it.
    """
    if is_release_layout(root):
        return os.path.join(root, RELEASE_GT_DIR, fruit_id, 'samples.npz')
    return os.path.join(root, fruit_id, 'laser', 'samples.npz')


def bounding_box(root, fruit_id):
    if is_release_layout(root):
        return os.path.join(root, RELEASE_GT_DIR, fruit_id, 'bounding_box.npz')
    return os.path.join(root, fruit_id, 'tf', 'bounding_box.npz')


SESSIONS_CSV = os.path.join('metadata', 'sessions.csv')


def sessions_csv(root):
    """The release's single metadata table, one row per fruit-session."""
    return os.path.join(root, SESSIONS_CSV)


def has_sessions(root):
    return os.path.isfile(sessions_csv(root))


def read_sessions(root):
    """Rows of the release metadata table, as dicts.

    The working tree splits this information across several files (temporal_metadata.csv,
    field_size_data_v2.csv, ground_truth.csv); the release carries one table, and the
    loaders below read whichever is present.
    """
    import csv as _csv
    with open(sessions_csv(root), newline='') as fh:
        return list(_csv.DictReader(fh))


def partial_suffix(root, working_suffix='_preprocessed'):
    """Filename suffix carried by partial clouds.

    The working tree tags them (`<frame>_preprocessed.ply`); the release names files by
    session date alone, so the suffix is empty there and every strip becomes a no-op.
    """
    return '' if is_release_layout(root) else working_suffix


def fruit_id_from_path(path, root=None):
    """Fruit id owning a partial-cloud path.

    With root given the id is read off the relative path, which is exact. Without it the
    path is scanned for the first component that looks like a fruit id and is followed by
    a known layout marker, so callers that never received a root still resolve correctly.
    """
    parts = os.path.normpath(path).split(os.sep)
    if root:
        try:
            rel = os.path.relpath(path, root)
            if not rel.startswith(os.pardir):
                rp = rel.split(os.sep)
                if rp[0] in (RELEASE_PARTIAL_DIR, RELEASE_GT_DIR):
                    return rp[1] if len(rp) > 1 else ''
                return rp[0]
        except ValueError:
            pass
    for i, p in enumerate(parts[:-1]):
        if p in (RELEASE_PARTIAL_DIR, RELEASE_GT_DIR) and i + 1 < len(parts) - 1:
            return parts[i + 1]
        if p == 'orbbec' or p == 'phenobox' or p == 'laser':
            return parts[i - 1] if i else ''
    return parts[-4] if len(parts) >= 4 else ''


def frame_id_from_path(path, suffix='_preprocessed'):
    """Frame id for a partial cloud, with the working tree's suffix removed."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem[:-len(suffix)] if suffix and stem.endswith(suffix) else stem


def looks_like_fruit_id(value):
    return isinstance(value, str) and FRUIT_ID_RE.match(value) is not None
