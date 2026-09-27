"""
Data loading and ingestion module for Business Entity Resolution.
Supports streaming and memory-efficient batch processing for large-scale TSV files.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple
from .normalizer import (
    normalize_business_name,
    normalize_address,
    get_spaceless_name,
    extract_core_tokens,
    extract_address_numbers,
    extract_address_keys
)


@dataclass(slots=True)
class EntityRecord:
    entity_id: str
    business_name: str
    business_address: str
    country: str
    norm_name: str
    norm_address: str
    spaceless_name: str
    name_tokens: Tuple[str, ...]
    address_numbers: Tuple[str, ...]
    address_keys: Tuple[str, ...]


def load_tsv_records(
    filepath: Path,
    max_records: Optional[int] = None,
    pre_normalize: bool = True
) -> Dict[str, EntityRecord]:
    """
    Load a source TSV file into a dictionary of EntityRecords keyed by entity_id.
    """
    records = {}
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header:
            return records

        for i, row in enumerate(reader):
            if max_records and i >= max_records:
                break
            if len(row) < 4:
                eid = row[0].strip() if len(row) > 0 else f"UNKNOWN-{i}"
                name = row[1].strip() if len(row) > 1 else ""
                addr = row[2].strip() if len(row) > 2 else ""
                cntry = row[3].strip() if len(row) > 3 else ""
            else:
                eid, name, addr, cntry = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()

            if pre_normalize:
                norm_name = normalize_business_name(name)
                norm_addr = normalize_address(addr)
                spaceless = get_spaceless_name(name)
                tokens = tuple(extract_core_tokens(name))
                numbers = tuple(extract_address_numbers(addr))
                addr_keys = tuple(extract_address_keys(addr))
            else:
                norm_name = norm_addr = spaceless = ""
                tokens = numbers = addr_keys = ()

            records[eid] = EntityRecord(
                entity_id=eid,
                business_name=name,
                business_address=addr,
                country=cntry,
                norm_name=norm_name,
                norm_address=norm_addr,
                spaceless_name=spaceless,
                name_tokens=tokens,
                address_numbers=numbers,
                address_keys=addr_keys
            )

    return records


def load_ground_truth(filepath: Path) -> Dict[str, Set[str]]:
    """
    Load ground truth matching labels mapping source1_entity_id -> set of matched entity_ids.
    """
    gt = {}
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # Skip header
        for row in reader:
            if not row or not row[0].strip():
                continue
            s1_id = row[0].strip()
            if len(row) > 1 and row[1].strip():
                matches = {m.strip() for m in row[1].strip().split(",") if m.strip()}
                gt[s1_id] = matches
            else:
                gt[s1_id] = set()
    return gt


def write_submission_file(
    filepath: Path,
    mapping: Dict[str, List[str]],
    header_col1: str,
    header_col2: str
) -> None:
    """
    Write a tab-separated submission file strictly formatted according to challenge rules.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow([header_col1, header_col2])
        for s1_id, id_list in mapping.items():
            seen = set()
            unique_ids = []
            for tid in id_list:
                if tid not in seen and tid != s1_id and (tid.startswith("S2-") or tid.startswith("S3-")):
                    seen.add(tid)
                    unique_ids.append(tid)
            joined = ",".join(unique_ids)
            writer.writerow([s1_id, joined])
