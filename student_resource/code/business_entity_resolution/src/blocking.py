"""
Candidate Generation and Multi-Tier Blocking Engine.
Indexes Source 2 and Source 3 records to generate high-recall candidate sets for Source 1 entities.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from .data_loader import EntityRecord
from .config import BLOCKING_CONFIG


class BlockingIndex:
    """
    Multi-tier inverted index for scalable candidate generation across multi-million entity pools.
    """

    def __init__(self, max_token_freq: int = 3000, min_token_len: int = 3):
        self.max_token_freq = max_token_freq
        self.min_token_len = min_token_len

        # Inverted index mappings
        # Key: (country, exact_norm_name) -> list of target_ids
        self.exact_name_index = defaultdict(list)
        
        # Key: (country, spaceless_name) -> list of target_ids
        self.spaceless_name_index = defaultdict(list)

        # Key: (country, token) -> list of target_ids
        self.token_index = defaultdict(list)
        self.token_doc_counts = defaultdict(int)

        # Key: (country, address_key) -> list of target_ids
        self.address_key_index = defaultdict(list)
        self.address_key_counts = defaultdict(int)

        # Key: (country, prefix_6) -> list of target_ids
        self.prefix_index = defaultdict(list)

    def build_index(self, target_records: Dict[str, EntityRecord]) -> None:
        """
        Build inverted indices from Source 2 and Source 3 target records.
        """
        # First pass: compute token and address key frequencies
        for tid, rec in target_records.items():
            cntry = rec.country.lower()
            if rec.norm_name:
                self.exact_name_index[(cntry, rec.norm_name)].append(tid)
                if len(rec.norm_name) >= 6:
                    self.prefix_index[(cntry, rec.norm_name[:6])].append(tid)

            if rec.spaceless_name:
                self.spaceless_name_index[(cntry, rec.spaceless_name)].append(tid)

            # Unique name tokens
            seen_tokens = set()
            for t in rec.name_tokens:
                if len(t) >= self.min_token_len:
                    seen_tokens.add(t)

            for t in seen_tokens:
                self.token_doc_counts[(cntry, t)] += 1

            # Unique address keys
            seen_addr_keys = set(rec.address_keys)
            for ak in seen_addr_keys:
                self.address_key_counts[(cntry, ak)] += 1

        # Second pass: populate token and address indices only for selective keys
        for tid, rec in target_records.items():
            cntry = rec.country.lower()
            seen_tokens = set()
            for t in rec.name_tokens:
                if len(t) >= self.min_token_len:
                    seen_tokens.add(t)

            for t in seen_tokens:
                if self.token_doc_counts[(cntry, t)] <= self.max_token_freq:
                    self.token_index[(cntry, t)].append(tid)

            for ak in set(rec.address_keys):
                if self.address_key_counts[(cntry, ak)] <= 800:
                    self.address_key_index[(cntry, ak)].append(tid)

    def get_candidates_for_s1(
        self,
        s1_rec: EntityRecord,
        max_candidates: int = 25
    ) -> List[str]:
        """
        Retrieve and rank candidate S2/S3 entity IDs for a given Source 1 record.
        """
        cntry = s1_rec.country.lower()
        candidate_scores = defaultdict(float)

        # 1. Exact normalized name match
        if s1_rec.norm_name:
            exact_matches = self.exact_name_index.get((cntry, s1_rec.norm_name), [])
            for tid in exact_matches:
                candidate_scores[tid] += 6.0

            # Prefix 6-gram match
            if len(s1_rec.norm_name) >= 6:
                prefix_matches = self.prefix_index.get((cntry, s1_rec.norm_name[:6]), [])
                if len(prefix_matches) <= 200:
                    for tid in prefix_matches:
                        candidate_scores[tid] += 1.8

        # 2. Spaceless name match (handles concatenated words, @handles, website names)
        if s1_rec.spaceless_name:
            spaceless_matches = self.spaceless_name_index.get((cntry, s1_rec.spaceless_name), [])
            for tid in spaceless_matches:
                candidate_scores[tid] += 5.5

        # 3. Rare & Informative Token Overlap
        s1_tokens = set(s1_rec.name_tokens)
        for t in s1_tokens:
            if len(t) >= self.min_token_len:
                freq = self.token_doc_counts.get((cntry, t), 0)
                if 0 < freq <= self.max_token_freq:
                    idf_weight = 2.5 / (1.0 + (freq ** 0.5) * 0.1)
                    matched_tids = self.token_index.get((cntry, t), [])
                    for tid in matched_tids:
                        candidate_scores[tid] += idf_weight

        # 4. Address Anchor Keys (Building number + Street token, Postal codes)
        for ak in set(s1_rec.address_keys):
            freq = self.address_key_counts.get((cntry, ak), 0)
            if 0 < freq <= 800:
                addr_weight = 2.0 if "_" in ak else 0.8
                matched_tids = self.address_key_index.get((cntry, ak), [])
                for tid in matched_tids:
                    candidate_scores[tid] += addr_weight

        if not candidate_scores:
            return []

        # Sort candidates by accumulated score descending
        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [tid for tid, score in sorted_candidates[:max_candidates]]


def generate_candidates_for_dataset(
    s1_records: Dict[str, EntityRecord],
    s2_records: Dict[str, EntityRecord],
    s3_records: Dict[str, EntityRecord],
    max_candidates: int = 25
) -> Dict[str, List[str]]:
    """
    Generate candidate pairs for all Source 1 records against Source 2 and Source 3.
    """
    target_records = {}
    target_records.update(s2_records)
    target_records.update(s3_records)

    index = BlockingIndex(
        max_token_freq=BLOCKING_CONFIG["max_block_size"],
        min_token_len=BLOCKING_CONFIG["min_token_len"]
    )
    index.build_index(target_records)

    candidate_mapping = {}
    for s1_id, s1_rec in s1_records.items():
        candidates = index.get_candidates_for_s1(s1_rec, max_candidates=max_candidates)
        candidate_mapping[s1_id] = candidates

    return candidate_mapping
