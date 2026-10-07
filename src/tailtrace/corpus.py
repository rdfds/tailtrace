"""Immutable byte-token shards with memory-mapped token and fixed-width index files."""

from __future__ import annotations

import codecs
import hashlib
import json
import mmap
import struct
from pathlib import Path

from tailtrace.evidence import atomic_json

ENTRY = struct.Struct("<QII")  # byte offset, sequence length, source document


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def content_hash(data):
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_corpus(inputs, out, max_length=128, min_length=2):
    if not 2 <= min_length <= max_length <= 1_000_000:
        raise ValueError("require 2 <= min_length <= max_length <= 1000000")
    files = set()
    for value in inputs:
        path = Path(value).resolve()
        if path.is_dir():
            files.update(
                p.resolve() for p in path.rglob("*") if p.is_file() and p.suffix in {".md", ".txt"}
            )
        elif path.is_file():
            files.add(path)
        else:
            raise FileNotFoundError(path)
    if not files:
        raise ValueError("no text documents found")
    files = sorted(files)
    import os

    root = Path(os.path.commonpath([str(p.parent) for p in files]))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    samples, targets, offset = 0, 0, 0
    documents = []
    with (out / "tokens.bin").open("xb") as tokens, (out / "index.bin").open("xb") as index:

        def emit(sequence, doc):
            nonlocal samples, targets, offset
            tokens.write(sequence)
            index.write(ENTRY.pack(offset, len(sequence), doc))
            offset += len(sequence)
            samples += 1
            targets += len(sequence) - 1

        for doc, path in enumerate(files):
            digest, size, pending = hashlib.sha256(), 0, b""
            decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
            before = samples
            with path.open("rb") as source:
                for block in iter(lambda: source.read(65536), b""):
                    decoder.decode(block)
                    digest.update(block)
                    size += len(block)
                    pending += block
                    while len(pending) >= max_length:
                        emit(pending[:max_length], doc)
                        pending = pending[max_length - 1 :]
                decoder.decode(b"", final=True)
            dropped = 0
            if len(pending) >= min_length:
                emit(pending, doc)
            else:
                dropped = max(0, len(pending) - 1)
            documents.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": digest.hexdigest(),
                    "bytes": size,
                    "samples": samples - before,
                    "dropped_targets": dropped,
                }
            )
    if not samples:
        raise ValueError("corpus contains no usable sequences; no manifest was committed")
    data = {
        "schema_version": 1,
        "codec": "utf8_bytes",
        "vocab_size": 256,
        "max_length": max_length,
        "min_length": min_length,
        "sample_count": samples,
        "valid_tokens": targets,
        "documents": documents,
        "tokens_bytes": offset,
        "tokens_sha256": file_hash(out / "tokens.bin"),
        "index_sha256": file_hash(out / "index.bin"),
    }
    data["sha256"] = content_hash(data)
    atomic_json(out / "manifest.json", data)
    return data


class TokenCorpus:
    def __init__(self, root):
        self.root = Path(root)
        self._files, self._maps = [], []
        try:
            self.manifest = json.loads((self.root / "manifest.json").read_text())
            data = self.manifest
            if (
                data.get("schema_version") != 1
                or data.get("codec") != "utf8_bytes"
                or data.get("vocab_size") != 256
            ):
                raise ValueError("unsupported corpus schema or codec")
            if data.get("sha256") != content_hash({k: v for k, v in data.items() if k != "sha256"}):
                raise ValueError("corpus manifest content changed")
            for name in ("tokens", "index"):
                path = self.root / f"{name}.bin"
                if file_hash(path) != data[f"{name}_sha256"]:
                    raise ValueError(f"corpus {name} content changed")
                stream = path.open("rb")
                self._files.append(stream)
                self._maps.append(mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ))
            self.tokens, self.index = self._maps
            if (
                len(self.index) != data["sample_count"] * ENTRY.size
                or len(self.tokens) != data["tokens_bytes"]
            ):
                raise ValueError("corpus file size mismatch")
            expected, targets = 0, 0
            for i in range(len(self)):
                offset, length, doc = self.entry(i)
                if (
                    offset != expected
                    or not data["min_length"] <= length <= data["max_length"]
                    or not 0 <= doc < len(data["documents"])
                ):
                    raise ValueError("invalid corpus index extent")
                expected += length
                targets += length - 1
            if expected != len(self.tokens) or targets != data["valid_tokens"]:
                raise ValueError("corpus index/token accounting mismatch")
            import os

            self._stats = [
                (s.st_size, s.st_mtime_ns) for s in (os.fstat(f.fileno()) for f in self._files)
            ]
        except BaseException:
            self.close()
            raise

    @property
    def sha256(self):
        return self.manifest["sha256"]

    def __len__(self):
        return self.manifest["sample_count"]

    def entry(self, i):
        if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(self):
            raise IndexError(i)
        return ENTRY.unpack_from(self.index, i * ENTRY.size)

    def __getitem__(self, i):
        import os

        if [
            (s.st_size, s.st_mtime_ns) for s in (os.fstat(f.fileno()) for f in self._files)
        ] != self._stats:
            raise ValueError("open corpus changed during training")
        offset, length, _ = self.entry(i)
        return self.tokens[offset : offset + length]

    def close(self):
        for mapping in self._maps:
            mapping.close()
        for stream in self._files:
            stream.close()
        self._maps, self._files = [], []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
