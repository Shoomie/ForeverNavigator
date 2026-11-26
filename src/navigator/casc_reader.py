"""Pure-Python local WoW CASC archive reader (supports WoW: Forever / Classic / Retail)."""

from __future__ import annotations

import bisect
import os
import re
import struct
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class CASCError(RuntimeError):
    pass


def decompress_blte(data: bytes) -> bytes:
    """Decompress a Blizzard BLTE container."""
    if len(data) < 8 or data[:4] != b"BLTE":
        raise CASCError("Invalid BLTE magic")

    header_size = struct.unpack(">I", data[4:8])[0]
    if header_size == 0:
        return _decode_blte_chunk(data[8:])

    if len(data) < header_size:
        raise CASCError("Truncated BLTE header")

    # flags (1 byte, 0x0F) + chunk_count (3 bytes big-endian)
    chunk_count = (data[9] << 16) | (data[10] << 8) | data[11]
    chunks_meta: List[Tuple[int, int]] = []
    pos = 12
    for _ in range(chunk_count):
        comp_size, decomp_size = struct.unpack(">II", data[pos : pos + 8])
        pos += 24  # 4 + 4 + 16 (md5)
        chunks_meta.append((comp_size, decomp_size))

    out = bytearray()
    data_pos = header_size
    for comp_size, _decomp_size in chunks_meta:
        chunk = data[data_pos : data_pos + comp_size]
        data_pos += comp_size
        out.extend(_decode_blte_chunk(chunk))
    return bytes(out)


def _decode_blte_chunk(chunk: bytes) -> bytes:
    if not chunk:
        return b""
    mode = chunk[0:1]
    payload = chunk[1:]
    if mode == b"N":
        return payload
    if mode == b"Z":
        return zlib.decompress(payload)
    if mode == b"F":
        return decompress_blte(payload)
    if mode == b"E":
        raise CASCError("Encrypted BLTE chunk (TACT key required)")
    raise CASCError(f"Unsupported BLTE chunk mode: {mode!r}")


class LocalCASC:
    """Reads files by FileDataID or EKey directly from a local World of Warcraft folder."""

    def __init__(self, wow_dir: str | Path, product: str = "wowf"):
        self.wow_dir = Path(wow_dir)
        self.data_dir = self.wow_dir / "Data"
        self.product = product

        self.build_key = self._read_build_key(product)
        self.build_config = self._read_config(self.build_key)

        self.root_ckey = bytes.fromhex(self.build_config["root"].split()[0])
        enc_parts = self.build_config["encoding"].split()
        self.enc_ckey = bytes.fromhex(enc_parts[0])
        self.enc_ekey = bytes.fromhex(enc_parts[1])

        self._idx: Dict[bytes, Tuple[int, int, int]] = {}
        self._load_indices()

        # Lazy-loaded encoding & root tables
        self._encoding_data: Optional[bytes] = None
        self._enc_ce_page_size: int = 0
        self._enc_ce_page_count: int = 0
        self._enc_ce_first_keys: List[bytes] = []
        self._enc_ce_pages_offset: int = 0
        self._enc_page_cache: Dict[int, Dict[bytes, bytes]] = {}

        self._fdid_to_ckey: Dict[int, bytes] = {}
        self._open_archives: Dict[int, object] = {}

    def close(self) -> None:
        for f in self._open_archives.values():
            try:
                f.close()
            except Exception:
                pass
        self._open_archives.clear()

    def __enter__(self) -> "LocalCASC":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _read_build_key(self, product: str) -> str:
        build_info = self.wow_dir / ".build.info"
        if not build_info.exists():
            raise CASCError(f".build.info not found in {self.wow_dir}")
        lines = build_info.read_text(encoding="utf-8").splitlines()
        headers = [h.split("!")[0] for h in lines[0].split("|")]
        prod_i = headers.index("Product")
        bkey_i = headers.index("Build Key")
        for line in lines[1:]:
            if not line.strip():
                continue
            cols = line.split("|")
            if cols[prod_i] == product:
                return cols[bkey_i]
        raise CASCError(f"Product {product!r} not found in .build.info")

    def _read_config(self, key: str) -> Dict[str, str]:
        cfg_path = self.data_dir / "config" / key[:2] / key[2:4] / key
        if not cfg_path.exists():
            raise CASCError(f"Config file not found: {cfg_path}")
        result: Dict[str, str] = {}
        for line in cfg_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            result[k.strip()] = v.strip()
        return result

    def _load_indices(self) -> None:
        idx_dir = self.data_dir / "data"
        # Group .idx files by 2-hex-digit bucket and pick the highest version
        latest_per_bucket: Dict[str, Path] = {}
        for p in idx_dir.glob("*.idx"):
            m = re.match(r"^([0-9a-fA-F]{2})([0-9a-fA-F]{8})\.idx$", p.name)
            if not m:
                continue
            bucket = m.group(1).lower()
            ver = int(m.group(2), 16)
            if bucket not in latest_per_bucket or ver > int(latest_per_bucket[bucket].stem[2:], 16):
                latest_per_bucket[bucket] = p

        for p in sorted(latest_per_bucket.values()):
            self._parse_idx_file(p)

    def _parse_idx_file(self, path: Path) -> None:
        raw = path.read_bytes()
        header_size, _header_hash = struct.unpack_from("<II", raw, 0)
        version, _bucket, _extra, size_bytes, offset_bytes, key_bytes, offset_bits = struct.unpack_from(
            "<HBBBBBB", raw, 8
        )
        if version != 7:
            raise CASCError(f"Unsupported .idx version {version} in {path.name}")

        # Entries block starts after header (8 + header_size), aligned up to 16 bytes
        pos = 8 + header_size
        if pos % 16 != 0:
            pos += 16 - (pos % 16)
        entries_size, _entries_hash = struct.unpack_from("<II", raw, pos)
        pos += 8

        entry_len = key_bytes + offset_bytes + size_bytes
        end = pos + entries_size
        offset_mask = (1 << offset_bits) - 1

        idx = self._idx
        while pos + entry_len <= end:
            ekey9 = raw[pos : pos + key_bytes]
            # 5-byte big-endian offset
            off_val = int.from_bytes(raw[pos + key_bytes : pos + key_bytes + offset_bytes], "big")
            size_val = int.from_bytes(
                raw[pos + key_bytes + offset_bytes : pos + entry_len], "little"
            )
            pos += entry_len
            if size_val == 0:
                continue
            archive_id = off_val >> offset_bits
            archive_offset = off_val & offset_mask
            idx[ekey9] = (archive_id, archive_offset, size_val)

    def read_by_ekey(self, ekey: bytes) -> bytes:
        """Read and BLTE-decompress a file given its 9+ byte EKey."""
        loc = self._idx.get(ekey[:9])
        if loc is None:
            raise FileNotFoundError(f"EKey {ekey[:9].hex()} not found in local CASC indices")
        archive_id, archive_offset, size_val = loc
        f = self._open_archives.get(archive_id)
        if f is None:
            arch_path = self.data_dir / "data" / f"data.{archive_id:03d}"
            f = open(arch_path, "rb")
            self._open_archives[archive_id] = f
        f.seek(archive_offset)
        block = f.read(size_val)
        if len(block) >= 8 and block[:4] == b"BLTE":
            return decompress_blte(block)
        # Skip 30-byte local archive header
        if len(block) < 34:
            raise CASCError(f"Truncated archive block at data.{archive_id:03d}+{archive_offset}")
        return decompress_blte(block[30:])

    def _ensure_encoding(self) -> None:
        if self._encoding_data is not None:
            return
        data = self.read_by_ekey(self.enc_ekey)
        if data[:2] != b"EN":
            raise CASCError("Invalid encoding file magic")
        _ver, ckey_sz, ekey_sz, ce_kb, _es_kb, ce_cnt, _es_cnt, _unk, espec_sz = struct.unpack_from(
            ">BBBHHIHBI", data, 2
        )
        # Wait: exact format of EN header:
        # 2s magic, B ver, B hash_size_ckey, B hash_size_ekey, H ce_page_kb, H es_page_kb,
        # I ce_page_count, I es_page_count, B unk, I espec_size
        _ver, ckey_sz, ekey_sz, ce_kb, _es_kb, ce_cnt, _es_cnt, _unk, espec_sz = struct.unpack_from(
            ">BBBHHIIBI", data, 2
        )
        if ckey_sz != 16 or ekey_sz != 16:
            raise CASCError(f"Unexpected encoding key sizes: {ckey_sz}, {ekey_sz}")
        pos = 22 + espec_sz
        first_keys: List[bytes] = []
        for _ in range(ce_cnt):
            first_keys.append(data[pos : pos + 16])
            pos += 32  # 16 ckey + 16 page_md5
        self._encoding_data = data
        self._enc_ce_page_size = ce_kb * 1024
        self._enc_ce_page_count = ce_cnt
        self._enc_ce_first_keys = first_keys
        self._enc_ce_pages_offset = pos

    def ckey_to_ekey(self, ckey: bytes) -> Optional[bytes]:
        """Resolve a 16-byte CKey to its 16-byte EKey via binary search on encoding pages."""
        self._ensure_encoding()
        idx = bisect.bisect_right(self._enc_ce_first_keys, ckey) - 1
        if idx < 0:
            return None
        page_map = self._enc_page_cache.get(idx)
        if page_map is None:
            page_map = {}
            data = self._encoding_data
            assert data is not None
            start = self._enc_ce_pages_offset + idx * self._enc_ce_page_size
            end = start + self._enc_ce_page_size
            p = start
            while p + 22 <= end:
                key_count = data[p]
                if key_count == 0:
                    break
                # 1 byte key_count + 5 bytes file_size + 16 bytes ckey + key_count * 16 bytes ekeys
                entry_ckey = data[p + 6 : p + 22]
                first_ekey = data[p + 22 : p + 38]
                page_map[entry_ckey] = first_ekey
                p += 22 + key_count * 16
            self._enc_page_cache[idx] = page_map
        return page_map.get(ckey)

    def read_by_ckey(self, ckey: bytes) -> bytes:
        ekey = self.ckey_to_ekey(ckey)
        if ekey is None:
            raise FileNotFoundError(f"CKey {ckey.hex()} not found in encoding table")
        return self.read_by_ekey(ekey)

    def _ensure_root(self) -> None:
        if self._fdid_to_ckey:
            return
        root_data = self.read_by_ckey(self.root_ckey)
        pos = 0
        version = 0
        if root_data[:4] == b"TSFM":
            val1, val2 = struct.unpack_from("<II", root_data, 4)
            # Root v2 (10.1.7+ / Classic 1.15+ / Forever):
            # magic(4), header_size(4) [20 or 24], version(4), total_files(4), named_files(4), ...
            if 16 <= val1 <= 64 and val2 in (1, 2):
                version = val2
                pos = val1
            else:
                # Root v1 (8.2+): magic(4), total_files(4), named_files(4)
                version = 1
                pos = 12

        total_len = len(root_data)
        fdid_map = self._fdid_to_ckey
        # Prefer enUS (0x02) or All (0xFFFFFFFF) locale, non-encrypted/standard
        preferred_locale = 0x2  # enUS
        fdid_locale_score: Dict[int, int] = {}

        while pos + 12 <= total_len:
            if version == 2:
                if pos + 17 > total_len:
                    break
                num_records, locale_flags, cf1, cf2, cf3 = struct.unpack_from("<IIIIB", root_data, pos)
                pos += 17
                content_flags = cf1 | cf2 | (cf3 << 17)
            else:
                num_records, content_flags, locale_flags = struct.unpack_from("<III", root_data, pos)
                pos += 12

            if num_records == 0:
                continue

            has_name_hashes = (content_flags & 0x10000000) == 0  # NoNameHash flag
            deltas_bytes = num_records * 4
            ckeys_bytes = num_records * 16
            hashes_bytes = (num_records * 8) if has_name_hashes else 0
            if pos + deltas_bytes + ckeys_bytes + hashes_bytes > total_len:
                break

            deltas = struct.unpack_from(f"<{num_records}i", root_data, pos)
            pos += deltas_bytes
            ckeys_start = pos
            pos += ckeys_bytes + hashes_bytes

            # Score block: prefer enUS or universal locale, prefer non-low-violence
            score = 0
            if locale_flags & preferred_locale:
                score += 10
            if locale_flags == 0xFFFFFFFF:
                score += 5
            if (content_flags & 0x80) == 0:  # not low violence
                score += 2

            fdid = -1
            for i, d in enumerate(deltas):
                fdid = fdid + 1 + d
                prev = fdid_locale_score.get(fdid, -1)
                if score > prev:
                    ck_off = ckeys_start + i * 16
                    fdid_map[fdid] = root_data[ck_off : ck_off + 16]
                    fdid_locale_score[fdid] = score

    def has_file(self, fdid: int) -> bool:
        self._ensure_root()
        return fdid in self._fdid_to_ckey

    def read_file(self, fdid: int) -> bytes:
        self._ensure_root()
        ckey = self._fdid_to_ckey.get(fdid)
        if ckey is None:
            raise FileNotFoundError(f"FileDataID {fdid} not found in root")
        return self.read_by_ckey(ckey)


def load_listfile(listfile_path: str | Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Load community-listfile-withcapitals.csv into (id->path, lower_path->id)."""
    id_to_path: Dict[int, str] = {}
    path_to_id: Dict[str, int] = {}
    with open(listfile_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or ";" not in line:
                continue
            fid_s, path = line.split(";", 1)
            try:
                fid = int(fid_s)
            except ValueError:
                continue
            norm = path.replace("\\", "/")
            id_to_path[fid] = norm
            path_to_id[norm.lower()] = fid
    return id_to_path, path_to_id
