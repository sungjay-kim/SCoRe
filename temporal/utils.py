import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from dataloaders import dataset_layout as layout


@dataclass
class FrameRecord:
    fruit_id: str
    frame_id: str
    dataset_index: int
    timestamp: Optional[float]


def _parse_timestamp(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none"}:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y%m%d", "%Y%m%d_%H%M%S"):
        try:
            return datetime.strptime(value, fmt).timestamp()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


class TemporalMetadataIndex:
    """Builds chronological sequences for the temporal encoder."""

    def __init__(
        self,
        files: Sequence[str],
        timestamp_csv: Optional[str],
        min_sequence_length: int = 2,
        orbbec_suffix: str = "_preprocessed",
        data_root: Optional[str] = None,
    ) -> None:
        self.orbbec_suffix = orbbec_suffix
        self.metadata: Dict[Tuple[str, str], float] = {}
        if data_root and layout.has_sessions(data_root):
            # Public release: session dates come from the single metadata table, and the
            # date doubles as the frame id because files are named by session. Dates are
            # parsed as dates rather than through _parse_timestamp, whose numeric fast
            # path would read a bare YYYYMMDD as the integer 20260311 and collapse every
            # inter-session interval to zero.
            for row in layout.read_sessions(data_root):
                try:
                    ts = datetime.strptime(row["session_date"].strip(), "%Y%m%d").timestamp()
                except (ValueError, KeyError, AttributeError):
                    continue
                self.metadata[(row["sample_id"], row["session_date"])] = ts
        elif timestamp_csv and os.path.isfile(timestamp_csv):
            df = pd.read_csv(timestamp_csv)
            required_cols = {"fruit_id", "frame_id", "timestamp"}
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                raise ValueError(f"Missing columns {missing} in {timestamp_csv}")
            for _, row in df.iterrows():
                ts = _parse_timestamp(row["timestamp"])
                if ts is not None:
                    key = (str(row["fruit_id"]).strip(), str(row["frame_id"]).strip())
                    self.metadata[key] = ts
        else:
            if timestamp_csv:
                print(f"[temporal] Timestamp CSV not found at {timestamp_csv}; default ordering will be used.")

        grouped: Dict[str, List[FrameRecord]] = {}
        for idx, path in enumerate(files):
            fruit_id = layout.fruit_id_from_path(path)
            filename = os.path.splitext(os.path.basename(path))[0]
            if filename.endswith(orbbec_suffix):
                frame_id = filename.replace(orbbec_suffix, '')
            else:
                frame_id = filename
            # Metadata CSV uses date-only frame_id (e.g. "20250523"), but
            # the file-derived frame_id may be "20250523_masked_color_points".
            # Try full frame_id first, then fall back to date prefix.
            ts = self.metadata.get((fruit_id, frame_id))
            if ts is None:
                date_prefix = frame_id.split("_")[0]
                ts = self.metadata.get((fruit_id, date_prefix))
            grouped.setdefault(fruit_id, []).append(FrameRecord(fruit_id, frame_id, idx, ts))

        # The last frame of each sequence is the final (harvest) frame — no truncation needed.
        # Each sample's last orbbec capture date corresponds to the harvest/laser scan date.
        self.sequences: List[List[FrameRecord]] = []
        for fruit_id, records in grouped.items():
            records.sort(key=lambda r: (float('inf') if r.timestamp is None else r.timestamp, r.frame_id))
            if len(records) >= min_sequence_length:
                self.sequences.append(records)

        self.min_sequence_length = min_sequence_length

    def iter_sequences(self, shuffle: bool = True) -> Sequence[List[FrameRecord]]:
        sequences = list(self.sequences)
        if shuffle:
            random.shuffle(sequences)
        return sequences

    def configure_balanced(
        self,
        vol_map_csv: str,
        bin_edges: Sequence[float],
        target_count_per_bin: int,
        size_inverse_alpha: float = 1.0,
        size_inverse_vol_ref: float = 300.0,
    ) -> None:
        """Build per-bin sequence pools weighted by size_inverse for balanced sampling.

        Each fruit/sequence is binned by its LAST-DATE ellipsoid volume
        (computed from height/width in vol_map_csv). Within each bin,
        sequences are weighted by ``(vol_ref / vol)^alpha`` so small fruits
        get higher sampling probability among the natural set.
        """
        if not os.path.isfile(vol_map_csv):
            raise FileNotFoundError(f"vol_map_csv not found: {vol_map_csv}")
        df = pd.read_csv(vol_map_csv)
        required = {"fruit_id", "date", "height", "width"}
        if not required.issubset(df.columns):
            raise ValueError(f"vol_map_csv missing cols {required - set(df.columns)}")
        # ellipsoid volume in ml: (4/3)π·(w/2)²·(h/2) / 1000 (assuming h,w in mm; consistent with cameralaser_w_masks.py)
        df = df.copy()
        df["ellipsoid_ml"] = (4.0 / 3.0) * 3.141592653589793 * (df["width"] / 2.0) ** 2 * (df["height"] / 2.0) / 1000.0
        # last-date per fruit
        df["date"] = df["date"].astype(str)
        last_vol = df.sort_values("date").groupby("fruit_id")["ellipsoid_ml"].last().to_dict()

        self._bin_edges = list(bin_edges)
        self._bin_target = int(target_count_per_bin)
        n_bins = len(self._bin_edges) - 1
        self._sequences_by_bin: Dict[int, List[Tuple[List[FrameRecord], float]]] = {b: [] for b in range(n_bins)}

        n_assigned = 0
        n_missing = 0
        for seq in self.sequences:
            fid = seq[0].fruit_id
            if fid not in last_vol or last_vol[fid] <= 0:
                n_missing += 1
                continue
            vol = float(last_vol[fid])
            bn = None
            for b in range(n_bins):
                if self._bin_edges[b] <= vol < self._bin_edges[b + 1]:
                    bn = b
                    break
            if bn is None:
                continue
            w = (size_inverse_vol_ref / vol) ** size_inverse_alpha
            self._sequences_by_bin[bn].append((seq, float(w)))
            n_assigned += 1

        self._balanced_configured = True
        print(f"[temporal-balanced] assigned {n_assigned} sequences ({n_missing} missing/zero vol). "
              f"target/bin={self._bin_target}, alpha={size_inverse_alpha}, vol_ref={size_inverse_vol_ref}")
        for b in range(n_bins):
            n = len(self._sequences_by_bin[b])
            tag = "  ⚠️ EMPTY" if n == 0 else ""
            print(f"  bin {b} [{self._bin_edges[b]:>4}-{self._bin_edges[b+1]:>4}]: {n:>3} sequences{tag}")

    def iter_sequences_balanced(self, shuffle: bool = True) -> Sequence[List[FrameRecord]]:
        """Sample ``bin_target`` sequences per bin with size_inverse weights (with replacement)."""
        if not getattr(self, "_balanced_configured", False):
            raise RuntimeError("Call configure_balanced() before iter_sequences_balanced().")
        out: List[List[FrameRecord]] = []
        for b, pool in self._sequences_by_bin.items():
            if not pool:
                continue
            seqs, weights = zip(*pool)
            sampled = random.choices(seqs, weights=weights, k=self._bin_target)
            out.extend(sampled)
        if shuffle:
            random.shuffle(out)
        return out

    def compute_delta(self, prev_ts: Optional[float], curr_ts: Optional[float], unit: float = 1.0) -> float:
        if prev_ts is None or curr_ts is None:
            return 0.0
        delta = max(curr_ts - prev_ts, 0.0)
        if unit <= 0:
            return delta
        return delta / unit
