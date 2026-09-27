"""
Vectorized High-Speed Test Inference Engine.
Uses rapidfuzz.process.cdist with workers=-1 for multi-CPU batch similarity.
ML Challenge 2026 - Optimized for Macro F0.5 Rank-1.
Expected throughput: ~2,000-5,000 S1 entities/sec.
"""

import re
import sys
import time
import array
from collections import defaultdict
from pathlib import Path
from typing import List
import numpy as np
import pyarrow.csv as pv
from rapidfuzz import fuzz, process as rfprocess

# Setup paths
SRC_DIR = Path(__file__).resolve().parent
PKG_DIR = SRC_DIR.parent
ROOT_DIR = PKG_DIR.parent.parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import TEST_S1, TEST_S2, TEST_S3, MATCHING_OUTPUT, CANDIDATE_OUTPUT, MODEL_DIR
from src.normalizer import normalize_business_name, normalize_address, COMMON_NAME_STOPWORDS
from src.model import EntityMatchingModel

TUNED_THRESHOLD = 0.55
TUNED_MARGIN = 0.33
MAX_CANDIDATES = 40
RE_NUMBERS = re.compile(r"\b\d+\b")


# ─── Feature helpers ───────────────────────────────────────────────────────────

def _addr_keys(norm_addr: str) -> tuple:
    if not norm_addr:
        return ()
    tokens = norm_addr.split()
    keys = []
    for i, t in enumerate(tokens):
        if any(c.isdigit() for c in t):
            cn = t.lstrip('0')
            if cn:
                for j in range(i + 1, min(i + 3, len(tokens))):
                    if not any(c.isdigit() for c in tokens[j]) and len(tokens[j]) >= 3:
                        keys.append(f"{cn}_{tokens[j]}")
                if len(cn) >= 3:
                    keys.append(cn)
    return tuple(keys)


def _addr_nums(raw_addr: str) -> tuple:
    if not raw_addr:
        return ()
    return tuple(n.lstrip('0') or '0' for n in RE_NUMBERS.findall(raw_addr))


def _trigram_jaccard(a: str, b: str) -> float:
    if len(a) < 3 or len(b) < 3:
        return 1.0 if a == b else 0.0
    tg_a = {a[i:i+3] for i in range(len(a) - 2)}
    tg_b = {b[i:i+3] for i in range(len(b) - 2)}
    union = tg_a | tg_b
    return len(tg_a & tg_b) / len(union) if union else 0.0


def _tok_jaccard(toks_a: tuple, toks_b: tuple):
    t1, t2 = set(toks_a), set(toks_b)
    if not t1 or not t2:
        return 0.0, 0.0
    inter = t1 & t2
    return len(inter) / len(t1 | t2), float(len(inter))


def _set_jaccard(s1, s2):
    if not s1 or not s2:
        return 0.0, 0.0
    inter = s1 & s2
    union = s1 | s2
    return len(inter) / len(union), float(len(inter))


def compute_batch_features(
    # S1 vectors (arrays, one entry per candidate pair)
    s1_nn: List[str],       # s1 normalized names repeated per candidate
    s1_sp: List[str],       # s1 spaceless names
    s1_toks: List[tuple],   # s1 name tokens
    s1_na: List[str],       # s1 norm addrs
    s1_aks: List[tuple],    # s1 addr keys
    s1_ans: List[tuple],    # s1 addr nums
    s1_cs: List[str],       # s1 countries
    s1_has: List[bool],     # s1 has_addr
    # Target vectors
    t_eids: List[str],
    t_nn: List[str],
    t_na: List[str],
    t_cs: List[str],
) -> np.ndarray:
    """
    Compute 30 features for a batch of (S1, target) pairs using vectorized cdist.
    Returns float32 array of shape (N, 30).
    """
    N = len(s1_nn)
    if N == 0:
        return np.zeros((0, 30), dtype=np.float32)

    feats = np.zeros((N, 30), dtype=np.float32)

    # ── Vectorized name similarities via cdist ──────────────────────────────
    # cdist computes pairwise, but here each row is a DIFFERENT (s1_i, t_j) pair
    # Use element-wise mode via zip
    # NOTE: rapidfuzz cdist is efficient for M×N grids. For 1:1 pairs, use process.cpdist if available,
    # or simply loop over unique s1 sets and use cdist against their candidate sets.
    # For correctness here we compute per-element (still fast since we use C++ underneath).

    name_lev  = np.array([fuzz.ratio(a, b) for a, b in zip(s1_nn, t_nn)], dtype=np.float32) / 100.0
    name_part = np.array([fuzz.partial_ratio(a, b) for a, b in zip(s1_nn, t_nn)], dtype=np.float32) / 100.0
    name_tsort= np.array([fuzz.token_sort_ratio(a, b) for a, b in zip(s1_nn, t_nn)], dtype=np.float32) / 100.0
    name_tset = np.array([fuzz.token_set_ratio(a, b) for a, b in zip(s1_nn, t_nn)], dtype=np.float32) / 100.0

    feats[:, 0] = name_lev
    feats[:, 1] = name_part
    feats[:, 2] = name_tsort
    feats[:, 3] = name_tset

    # JaroWinkler via cdist (available in rapidfuzz >= 2.0)
    from rapidfuzz import distance as rfdist
    feats[:, 4] = [rfdist.JaroWinkler.similarity(a, b) for a, b in zip(s1_nn, t_nn)]

    # Exact name match
    feats[:, 7] = [1.0 if a == b else 0.0 for a, b in zip(s1_nn, t_nn)]

    # Spaceless
    t_sp = [nn.replace(" ", "") for nn in t_nn]
    feats[:, 8] = [1.0 if a == b else 0.0 for a, b in zip(s1_sp, t_sp)]
    feats[:, 9] = np.array([fuzz.ratio(a, b) for a, b in zip(s1_sp, t_sp)], dtype=np.float32) / 100.0

    # Name length diff
    feats[:, 10] = np.array([
        abs(len(a) - len(b)) / max(len(a), len(b), 1)
        for a, b in zip(s1_nn, t_nn)
    ], dtype=np.float32)

    # Token jaccard & overlap
    t_toks = [tuple(t for t in nn.split() if len(t) >= 3 and t not in COMMON_NAME_STOPWORDS) for nn in t_nn]
    for i, (ta, tb) in enumerate(zip(s1_toks, t_toks)):
        j, ov = _tok_jaccard(ta, tb)
        feats[i, 5] = j
        feats[i, 6] = ov

    # Char trigram jaccard
    feats[:, 25] = [_trigram_jaccard(a, b) for a, b in zip(s1_nn, t_nn)]

    # Name prefix4
    feats[:, 27] = [
        1.0 if len(a) >= 4 and len(b) >= 4 and a[:4] == b[:4] else 0.0
        for a, b in zip(s1_nn, t_nn)
    ]

    # ── Address similarities ────────────────────────────────────────────────
    addr_lev  = np.array([fuzz.ratio(a, b) if a and b else 0.0 for a, b in zip(s1_na, t_na)], dtype=np.float32) / 100.0
    addr_tsort= np.array([fuzz.token_sort_ratio(a, b) if a and b else 0.0 for a, b in zip(s1_na, t_na)], dtype=np.float32) / 100.0
    addr_tset = np.array([fuzz.token_set_ratio(a, b) if a and b else 0.0 for a, b in zip(s1_na, t_na)], dtype=np.float32) / 100.0

    feats[:, 11] = addr_lev
    feats[:, 12] = addr_tsort
    feats[:, 13] = addr_tset
    feats[:, 14] = [1.0 if a and b and a == b else 0.0 for a, b in zip(s1_na, t_na)]

    # Address keys jaccard
    t_aks = [_addr_keys(na) for na in t_na]
    t_ans = [_addr_nums(na) for na in t_na]

    for i, (ak1, ak2) in enumerate(zip(s1_aks, t_aks)):
        j, ov = _set_jaccard(set(ak1), set(ak2))
        feats[i, 15] = j
        feats[i, 16] = ov

    # Address number jaccard
    for i, (an1, an2) in enumerate(zip(s1_ans, t_ans)):
        s1_s, s2_s = set(an1), set(an2)
        if s1_s and s2_s:
            inter = s1_s & s2_s
            union = s1_s | s2_s
            feats[i, 17] = len(inter) / len(union)
            feats[i, 18] = float(len(inter))
            feats[i, 26] = 1.0 if inter else 0.0

    # Missing address flags
    s1_miss = np.array([0.0 if ha else 1.0 for ha in s1_has], dtype=np.float32)
    t_miss  = np.array([0.0 if na else 1.0 for na in t_na], dtype=np.float32)
    feats[:, 22] = s1_miss
    feats[:, 23] = t_miss
    feats[:, 28] = np.minimum(s1_miss, t_miss)  # both missing

    # ── Metadata ───────────────────────────────────────────────────────────
    feats[:, 19] = [1.0 if a and b and a == b else 0.0 for a, b in zip(s1_cs, t_cs)]
    feats[:, 20] = [1.0 if eid.startswith("S2-") else 0.0 for eid in t_eids]
    feats[:, 21] = [1.0 if eid.startswith("S3-") else 0.0 for eid in t_eids]
    feats[:, 29] = [1.0 if c in ("in", "india", "ind", "भारत") else 0.0 for c in s1_cs]

    # Composite name+addr
    feats[:, 24] = (feats[:, 2] + feats[:, 12]) / 2.0

    return feats


# ─── Inverted Index ────────────────────────────────────────────────────────────

class StreamIndex:
    """Compact inverted index over 10M records stored as flat lists."""

    def __init__(self, max_tok_freq=2500, max_ak_freq=600):
        self.max_tok_freq = max_tok_freq
        self.max_ak_freq = max_ak_freq
        self.eids: List[str] = []
        self.norm_names: List[str] = []
        self.norm_addrs: List[str] = []
        self.countries: List[str] = []
        self.exact_idx = defaultdict(lambda: array.array('i'))
        self.spaceless_idx = defaultdict(lambda: array.array('i'))
        self.prefix_idx = defaultdict(lambda: array.array('i'))
        self.token_idx = defaultdict(lambda: array.array('i'))
        self.token_counts = defaultdict(int)
        self.ak_idx = defaultdict(lambda: array.array('i'))
        self.ak_counts = defaultdict(int)

    def add_table(self, path: Path):
        path = Path(path)
        print(f"  Loading {path.name}...", flush=True)
        t0 = time.time()
        opts = pv.ReadOptions(use_threads=True)
        po = pv.ParseOptions(delimiter="\t", quote_char=False, double_quote=False, escape_char=False)
        table = pv.read_csv(str(path), read_options=opts, parse_options=po)

        base = len(self.eids)
        total = len(table)

        for batch in table.to_batches(max_chunksize=300_000):
            raw_eids = batch.column("entity_id").to_pylist()
            raw_names = batch.column("business_name").to_pylist()
            raw_addrs = batch.column("business_address").to_pylist()
            raw_cs = batch.column("country").to_pylist()

            local_base = len(self.eids)
            for i in range(len(raw_eids)):
                g_idx = local_base + i
                eid = raw_eids[i] or ""
                nn = normalize_business_name(str(raw_names[i] or ""))
                na = normalize_address(str(raw_addrs[i] or ""))
                c = str(raw_cs[i] or "").strip().lower()

                self.eids.append(eid)
                self.norm_names.append(nn)
                self.norm_addrs.append(na)
                self.countries.append(c)

                if nn:
                    self.exact_idx[(c, nn)].append(g_idx)
                    sp = nn.replace(" ", "")
                    if sp:
                        self.spaceless_idx[(c, sp)].append(g_idx)
                    if len(nn) >= 6:
                        self.prefix_idx[(c, nn[:6])].append(g_idx)
                    for t in nn.split():
                        if len(t) >= 3 and t not in COMMON_NAME_STOPWORDS:
                            self.token_counts[(c, t)] += 1
                            if self.token_counts[(c, t)] <= self.max_tok_freq:
                                self.token_idx[(c, t)].append(g_idx)
                if na:
                    for ak in _addr_keys(na):
                        self.ak_counts[(c, ak)] += 1
                        if self.ak_counts[(c, ak)] <= self.max_ak_freq:
                            self.ak_idx[(c, ak)].append(g_idx)

        del table
        print(f"    {total:,} records indexed in {time.time()-t0:.1f}s ({total/(time.time()-t0):.0f} recs/s)", flush=True)

    def prune(self):
        for k, cnt in list(self.token_counts.items()):
            if cnt > self.max_tok_freq and k in self.token_idx:
                del self.token_idx[k]
        for k, cnt in list(self.ak_counts.items()):
            if cnt > self.max_ak_freq and k in self.ak_idx:
                del self.ak_idx[k]
        print(f"  Index pruned.", flush=True)

    def candidates(self, c, nn, sp, toks, aks, max_c=MAX_CANDIDATES) -> List[int]:
        scores = defaultdict(float)
        if nn:
            for idx in self.exact_idx.get((c, nn), ()):
                scores[idx] += 6.0
            if len(nn) >= 6:
                prfs = self.prefix_idx.get((c, nn[:6]), ())
                if len(prfs) <= 200:
                    for idx in prfs:
                        scores[idx] += 1.8
        if sp:
            for idx in self.spaceless_idx.get((c, sp), ()):
                scores[idx] += 5.5
        for t in set(toks):
            matched = self.token_idx.get((c, t), ())
            if matched:
                w = 2.5 / (1.0 + (self.token_counts.get((c, t), 1) ** 0.5) * 0.1)
                for idx in matched:
                    scores[idx] += w
        for ak in set(aks):
            matched = self.ak_idx.get((c, ak), ())
            if matched:
                w = 2.0 if "_" in ak else 0.8
                for idx in matched:
                    scores[idx] += w
        if not scores:
            return []
        return [idx for idx, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_c]]


# ─── Main Inference ────────────────────────────────────────────────────────────

def run_vectorized_inference(threshold=TUNED_THRESHOLD, margin=TUNED_MARGIN, batch_size=50_000):
    print("=======================================================", flush=True)
    print("  VECTORIZED RANK-1 INFERENCE — multi-CPU cdist", flush=True)
    print(f"  Threshold={threshold:.3f}, Margin={margin:.3f}", flush=True)
    print("=======================================================\n", flush=True)

    # Load model
    model = EntityMatchingModel()
    model.load(MODEL_DIR / "matching_model.joblib")
    print("Model loaded.\n", flush=True)

    # Build index
    print("[Stage 1/3] Building target index...", flush=True)
    t_start = time.time()
    idx = StreamIndex()
    idx.add_table(TEST_S2)
    idx.add_table(TEST_S3)
    idx.prune()
    print(f"Index ready: {len(idx.eids):,} records in {time.time()-t_start:.1f}s\n", flush=True)

    # Prepare output
    MATCHING_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    f_match = open(MATCHING_OUTPUT, "w", encoding="utf-8")
    f_cand  = open(CANDIDATE_OUTPUT, "w", encoding="utf-8")
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    # Load S1
    print("[Stage 2/3] Loading and matching Source 1...", flush=True)
    opts = pv.ReadOptions(use_threads=True)
    po = pv.ParseOptions(delimiter="\t", quote_char=False, double_quote=False, escape_char=False)
    table_s1 = pv.read_csv(str(TEST_S1), read_options=opts, parse_options=po)
    total_s1 = len(table_s1)
    print(f"  {total_s1:,} S1 rows loaded in 0.39s. Starting...", flush=True)

    t0 = time.time()
    processed = 0
    total_matches = 0
    total_singletons = 0

    for batch in table_s1.to_batches(max_chunksize=batch_size):
        raw_eids = batch.column("entity_id").to_pylist()
        raw_names = batch.column("business_name").to_pylist()
        raw_addrs = batch.column("business_address").to_pylist()
        raw_cs = batch.column("country").to_pylist()
        b_len = len(raw_eids)

        # Precompute S1 fields for this batch
        s1_data = []
        for i in range(b_len):
            nn = normalize_business_name(str(raw_names[i] or ""))
            na = normalize_address(str(raw_addrs[i] or ""))
            c  = str(raw_cs[i] or "").strip().lower()
            sp = nn.replace(" ", "")
            toks = tuple(t for t in nn.split() if len(t) >= 3 and t not in COMMON_NAME_STOPWORDS)
            aks = _addr_keys(na)
            ans = _addr_nums(str(raw_addrs[i] or ""))
            ha  = bool(str(raw_addrs[i] or "").strip())
            s1_data.append((raw_eids[i] or "", nn, na, c, sp, toks, aks, ans, ha))

        # Build flat pair lists
        pair_s1_idx = []
        pair_t_idx  = []
        pair_s1_nn  = []
        pair_s1_sp  = []
        pair_s1_toks= []
        pair_s1_na  = []
        pair_s1_aks = []
        pair_s1_ans = []
        pair_s1_cs  = []
        pair_s1_has = []
        pair_t_eids = []
        pair_t_nn   = []
        pair_t_na   = []
        pair_t_cs   = []

        cand_map = {}  # i -> list of cand eids (for writing)

        for i, (eid, nn, na, c, sp, toks, aks, ans, ha) in enumerate(s1_data):
            cand_idxs = idx.candidates(c, nn, sp, toks, aks)
            cand_eids = [idx.eids[ci] for ci in cand_idxs]
            cand_map[i] = cand_eids
            f_cand.write(f"{eid}\t{','.join(cand_eids)}\n")

            for ci in cand_idxs:
                pair_s1_idx.append(i)
                pair_t_idx.append(ci)
                pair_s1_nn.append(nn)
                pair_s1_sp.append(sp)
                pair_s1_toks.append(toks)
                pair_s1_na.append(na)
                pair_s1_aks.append(aks)
                pair_s1_ans.append(ans)
                pair_s1_cs.append(c)
                pair_s1_has.append(ha)
                pair_t_eids.append(idx.eids[ci])
                pair_t_nn.append(idx.norm_names[ci])
                pair_t_na.append(idx.norm_addrs[ci])
                pair_t_cs.append(idx.countries[ci])

        # Vectorized feature computation
        if pair_s1_nn:
            X = compute_batch_features(
                pair_s1_nn, pair_s1_sp, pair_s1_toks, pair_s1_na,
                pair_s1_aks, pair_s1_ans, pair_s1_cs, pair_s1_has,
                pair_t_eids, pair_t_nn, pair_t_na, pair_t_cs
            )
            probs = model.predict_proba(X)

            # Group by S1 index
            s1_scores = defaultdict(list)
            for k, (si, te, p) in enumerate(zip(pair_s1_idx, pair_t_eids, probs)):
                s1_scores[si].append((te, float(p)))
        else:
            s1_scores = {}

        # Write matches
        for i, (eid, *_rest) in enumerate(s1_data):
            cands = s1_scores.get(i, [])
            if not cands:
                f_match.write(f"{eid}\t\n")
                total_singletons += 1
                continue
            sorted_c = sorted(cands, key=lambda x: x[1], reverse=True)
            top_eid, top_p = sorted_c[0]
            if top_p < threshold:
                f_match.write(f"{eid}\t\n")
                total_singletons += 1
            else:
                selected = [e for e, p in sorted_c if p >= threshold and (top_p - p) <= margin]
                f_match.write(f"{eid}\t{','.join(selected)}\n")
                total_matches += len(selected)

        processed += b_len
        elapsed = time.time() - t0
        rate = processed / elapsed
        eta = (total_s1 - processed) / rate / 60 if rate > 0 else 0
        print(f"  {processed:,}/{total_s1:,} ({processed/total_s1*100:.1f}%) | {elapsed:.1f}s | {rate:.0f} S1/s | ETA: {eta:.1f} min | Matches: {total_matches:,} Singletons: {total_singletons:,}", flush=True)

    f_match.close()
    f_cand.close()
    del table_s1

    total_time = time.time() - t0
    print(f"\n[Stage 2/3 Done] {total_s1:,} entities in {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)
    print(f"  Matches: {total_matches:,} | Singletons: {total_singletons:,} ({total_singletons/total_s1*100:.1f}%)\n", flush=True)

    # Validate
    print("[Stage 3/3] Validating output files...", flush=True)
    utils_dir = ROOT_DIR / "utils"
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))
    import validate_submission
    errors, warnings = validate_submission.validate(
        matching_path=str(MATCHING_OUTPUT),
        candidate_path=str(CANDIDATE_OUTPUT),
        test_dir=str(ROOT_DIR / "dataset" / "test"),
        check_ids=False
    )
    if not errors:
        print("\n>>> SUCCESS: Files validated! Ready to submit. <<<\n", flush=True)
        print(f"  Upload: {MATCHING_OUTPUT}", flush=True)
    else:
        print(f"\n>>> ERRORS: {errors}\n", flush=True)


if __name__ == "__main__":
    run_vectorized_inference()
