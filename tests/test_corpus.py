import json

import pytest

from tailtrace.corpus import TokenCorpus, build_corpus


def test_overlapping_windows_preserve_every_document_next_byte_target(tmp_path):
    path = tmp_path / "unicode.txt"
    original = ("CUDA → GPU\n" * 10000).encode()
    path.write_bytes(original)
    out = tmp_path / "shard"
    manifest = build_corpus([path], out, max_length=31)
    with TokenCorpus(out) as corpus:
        reconstructed = corpus[0] + b"".join(corpus[i][1:] for i in range(1, len(corpus)))
        assert reconstructed == original
        assert manifest["valid_tokens"] == len(original) - 1
        assert manifest["documents"][0]["dropped_targets"] == 0


def test_determinism_and_explicit_short_tail_accounting(tmp_path):
    path = tmp_path / "text.txt"
    path.write_bytes(b"abcdefghi")
    a = build_corpus([path], tmp_path / "a", 8, 4)
    b = build_corpus([path, path], tmp_path / "b", 8, 4)
    assert a == b
    assert a["valid_tokens"] == 7
    assert a["documents"][0]["dropped_targets"] == 1
    with pytest.raises(FileExistsError):
        build_corpus([path], tmp_path / "a")


def test_rejects_payload_and_manifest_tampering(tmp_path):
    path = tmp_path / "text.txt"
    path.write_text("abcdefghij")
    out = tmp_path / "corpus"
    build_corpus([path], out)
    payload = out / "tokens.bin"
    payload.write_bytes(b"x" + payload.read_bytes()[1:])
    with pytest.raises(ValueError, match="tokens content changed"):
        TokenCorpus(out)
    data = json.loads((out / "manifest.json").read_text())
    data["sample_count"] += 1
    (out / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="manifest content changed"):
        TokenCorpus(out)
