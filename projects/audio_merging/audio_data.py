"""Data utilities for audio-classification model merging with AST (ViT for audio).

We evaluate on six diverse audio classification tasks spanning environmental
sound, urban sound, music genre, speech emotion and keyword spotting:
  * ESC-50          - environmental sound (50 classes)
  * UrbanSound8K     - urban sound (10 classes)
  * GTZAN            - music genre (10 classes)
  * RAVDESS          - speech emotion (8 classes)
  * CREMA-D          - speech emotion (6 classes)
  * Speech Commands  - keyword spotting (~35 classes)

Audio is decoded and resampled to 16 kHz and turned into log-mel spectrograms by
the AST feature extractor (``input_values`` of shape (1024, 128)).

Decoding uses scipy/stdlib WAV readers. We do **not** use HuggingFace
audio decode (needs torchcodec/FFmpeg) or ``torchaudio.load`` (same backend).
"""

import io
import os
from math import gcd

import numpy as np
import torch
from datasets import Audio, load_dataset
from transformers import AutoFeatureExtractor

# task_name -> dict(repo, config, label_col, split)
# split strategy:
#   "native"  -> dataset already provides train/test
#   "fold:N"  -> column 'fold' present; fold == N is the test set
#   "ratio"   -> single split; deterministic 80/20 split
AUDIO_TASKS = {
    "esc50": dict(repo="ashraq/esc50", config=None, label_col="target", split="fold:5"),
    "urbansound8k": dict(
        repo="danavery/urbansound8K", config=None, label_col="classID", split="fold:10"
    ),
    "gtzan": dict(
        repo="sanchit-gandhi/gtzan", config=None, label_col="genre", split="ratio"
    ),
    "ravdess": dict(
        repo="xbgoose/ravdess", config=None, label_col="emotion", split="ratio"
    ),
    "cremad": dict(
        repo="AbstractTTS/CREMA-D", config=None, label_col="major_emotion", split="ratio"
    ),
    "speech_commands": dict(
        # google/speech_commands still ships a loading script, which current
        # `datasets` rejects. beeneptune is a parquet drop-in of v0.02.
        repo="beeneptune/speech_commands",
        config="v0.02",
        label_col="label",
        split="native",
        fallback_repos=(("pollen-robotics/speech-commands-v0.02", None),),
    ),
}

ALL_TASKS = list(AUDIO_TASKS.keys())

TARGET_SR = 16000


def _load_full(spec):
    attempts = [(spec["repo"], spec.get("config"))]
    attempts.extend(spec.get("fallback_repos") or ())
    last_exc = None
    for repo, config in attempts:
        try:
            if config:
                return load_dataset(repo, config)
            return load_dataset(repo)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(
        f"Failed to load audio dataset {spec.get('repo')}: {last_exc}"
    ) from last_exc


def _to_mono(wav):
    if torch.is_tensor(wav):
        if wav.ndim == 2:
            wav = wav.mean(dim=0)
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        # (channels, time) vs (time, channels)
        if wav.shape[0] <= 8 and wav.shape[0] < wav.shape[-1]:
            wav = wav.mean(axis=0)
        else:
            wav = wav.mean(axis=-1)
    return np.asarray(wav, dtype=np.float32)


def _pcm_to_float(data):
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float32, copy=False)
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    if data.dtype == np.int32:
        return data.astype(np.float32) / 2147483648.0
    if data.dtype == np.uint8:
        return (data.astype(np.float32) - 128.0) / 128.0
    if data.dtype == np.int8:
        return data.astype(np.float32) / 128.0
    peak = float(np.max(np.abs(data))) if data.size else 1.0
    return data.astype(np.float32) / max(peak, 1.0)


def _resample(wav, sr, target_sr):
    """Resample with scipy (preferred) or numpy; never import torchaudio/torchcodec."""
    sr = int(sr)
    if sr == target_sr:
        return wav.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly

        g = gcd(sr, target_sr)
        out = resample_poly(wav, target_sr // g, sr // g)
        return np.asarray(out, dtype=np.float32)
    except Exception:
        n_out = max(1, int(round(len(wav) * float(target_sr) / sr)))
        x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, wav).astype(np.float32)


def _as_readable(source):
    if isinstance(source, (bytes, bytearray, memoryview)):
        return io.BytesIO(source)
    if isinstance(source, io.BytesIO):
        source.seek(0)
        return source
    return source


def _read_wav_scipy(source):
    from scipy.io import wavfile

    sr, data = wavfile.read(_as_readable(source))
    return _to_mono(_pcm_to_float(data)), int(sr)


def _read_wav_stdlib(source):
    import wave

    src = _as_readable(source)
    closer = None
    if isinstance(src, (str, os.PathLike)):
        handle = wave.open(str(src), "rb")
        closer = handle
    else:
        handle = wave.open(src, "rb")
    try:
        sr = handle.getframerate()
        nch = handle.getnchannels()
        width = handle.getsampwidth()
        raw = handle.readframes(handle.getnframes())
    finally:
        if closer is not None:
            closer.close()
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2")
    elif width == 1:
        data = np.frombuffer(raw, dtype=np.uint8)
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4")
    else:
        raise ValueError(f"unsupported WAV sample width {width}")
    if nch > 1:
        data = data.reshape(-1, nch)
    return _to_mono(_pcm_to_float(data)), int(sr)


def _source_bytes(source) -> bytes:
    src = _as_readable(source)
    if isinstance(src, (str, os.PathLike)):
        with open(src, "rb") as handle:
            return handle.read()
    return src.read()


def _riff_chunks(blob: bytes):
    if blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos = 12
    while pos + 8 <= len(blob):
        cid = blob[pos : pos + 4]
        size = int.from_bytes(blob[pos + 4 : pos + 8], "little")
        start = pos + 8
        yield cid, blob[start : start + size]
        pos = start + size + (size & 1)


# Microsoft ADPCM (WAV format 0x0002), used by some UrbanSound8K files.
_MS_ADPCM_ADAPT = (
    230, 230, 230, 230, 307, 409, 512, 614,
    768, 614, 512, 409, 307, 230, 230, 230,
)
_MS_ADPCM_COEFFS = (
    (256, 0),
    (512, -256),
    (0, 0),
    (192, 64),
    (240, 0),
    (460, -208),
    (392, -232),
)
_IMA_STEP = (
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
    50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230,
    253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796, 876, 963,
    1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327,
    3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442,
    11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794,
    32767,
)
_IMA_INDEX = (-1, -1, -1, -1, 2, 4, 6, 8)


def _clip_i16(x: int) -> int:
    return -32768 if x < -32768 else 32767 if x > 32767 else x


def _decode_ms_adpcm_block(block: bytes, nch: int, samples_per_block: int, coeffs) -> np.ndarray:
    if nch == 1:
        return _decode_ms_adpcm_mono_block(block, samples_per_block, coeffs)
    header = 7 * nch
    if len(block) < header:
        raise ValueError("truncated MS-ADPCM block")
    pred = [block[i] for i in range(nch)]
    delta = [
        int.from_bytes(block[nch + 2 * i : nch + 2 * i + 2], "little", signed=True)
        for i in range(nch)
    ]
    samp1 = [
        int.from_bytes(
            block[nch + 2 * nch + 2 * i : nch + 2 * nch + 2 * i + 2],
            "little",
            signed=True,
        )
        for i in range(nch)
    ]
    samp2 = [
        int.from_bytes(
            block[nch + 4 * nch + 2 * i : nch + 4 * nch + 2 * i + 2],
            "little",
            signed=True,
        )
        for i in range(nch)
    ]
    out = np.empty((samples_per_block, nch), dtype=np.int16)
    for ch in range(nch):
        out[0, ch] = samp2[ch]
        out[1, ch] = samp1[ch]
    si = 2
    for byte in block[header:]:
        if si >= samples_per_block:
            break
        for ch, nibble in enumerate(((byte >> 4) & 0xF, byte & 0xF)[:nch]):
            signed = nibble - 16 if nibble & 0x8 else nibble
            c1, c2 = coeffs[pred[ch] % len(coeffs)]
            pred_s = (samp1[ch] * c1 + samp2[ch] * c2) >> 8
            sample = _clip_i16(pred_s + signed * delta[ch])
            delta[ch] = max((delta[ch] * _MS_ADPCM_ADAPT[nibble]) >> 8, 16)
            samp2[ch], samp1[ch] = samp1[ch], sample
            out[si, ch] = sample
        si += 1
    return out


def _decode_ms_adpcm_mono_block(block: bytes, samples_per_block: int, coeffs) -> np.ndarray:
    pred = block[0]
    delta = int.from_bytes(block[1:3], "little", signed=True)
    samp1 = int.from_bytes(block[3:5], "little", signed=True)
    samp2 = int.from_bytes(block[5:7], "little", signed=True)
    out = np.empty(samples_per_block, dtype=np.int16)
    out[0] = samp2
    out[1] = samp1
    c1, c2 = coeffs[pred % len(coeffs)]
    payload = block[7:]
    si = 2
    for byte in payload:
        for shift in (4, 0):
            if si >= samples_per_block:
                return out
            nibble = (byte >> shift) & 0xF
            signed = nibble - 16 if nibble & 0x8 else nibble
            pred_s = (samp1 * c1 + samp2 * c2) >> 8
            sample = _clip_i16(pred_s + signed * delta)
            delta = max((delta * _MS_ADPCM_ADAPT[nibble]) >> 8, 16)
            samp2, samp1 = samp1, sample
            out[si] = sample
            si += 1
    return out


def _decode_ima_adpcm_mono_block(block: bytes, samples_per_block: int) -> np.ndarray:
    predictor = int.from_bytes(block[0:2], "little", signed=True)
    index = block[2]
    index = 0 if index > 88 else index
    out = np.empty(samples_per_block, dtype=np.int16)
    out[0] = predictor
    si = 1
    for byte in block[4:]:
        for nibble in (byte & 0xF, (byte >> 4) & 0xF):
            if si >= samples_per_block:
                return out
            step = _IMA_STEP[index]
            diff = step >> 3
            if nibble & 1:
                diff += step >> 2
            if nibble & 2:
                diff += step >> 1
            if nibble & 4:
                diff += step
            predictor = _clip_i16(predictor + (-diff if nibble & 8 else diff))
            index = min(max(index + _IMA_INDEX[nibble & 7], 0), 88)
            out[si] = predictor
            si += 1
    return out


def _read_wav_compressed(source):
    """Decode MS-ADPCM / IMA-ADPCM WAVE files that scipy and stdlib wave reject."""
    blob = _source_bytes(source)
    fmt = data = None
    for cid, payload in _riff_chunks(blob):
        if cid == b"fmt ":
            fmt = payload
        elif cid == b"data":
            data = payload
    if fmt is None or data is None:
        raise ValueError("WAVE missing fmt/data chunk")
    audio_format = int.from_bytes(fmt[0:2], "little")
    nch = int.from_bytes(fmt[2:4], "little")
    sr = int.from_bytes(fmt[4:8], "little")
    block_align = int.from_bytes(fmt[12:14], "little")
    extra = fmt[18:] if len(fmt) >= 18 else b""
    samples_per_block = int.from_bytes(extra[0:2], "little") if len(extra) >= 2 else 0

    if audio_format == 2:
        coeffs = list(_MS_ADPCM_COEFFS)
        if len(extra) >= 4:
            ncoef = int.from_bytes(extra[2:4], "little")
            custom = []
            pos = 4
            for _ in range(ncoef):
                if pos + 4 > len(extra):
                    break
                c1 = int.from_bytes(extra[pos : pos + 2], "little", signed=True)
                c2 = int.from_bytes(extra[pos + 2 : pos + 4], "little", signed=True)
                custom.append((c1, c2))
                pos += 4
            if custom:
                coeffs = custom
        if samples_per_block <= 0:
            samples_per_block = ((block_align - 7 * nch) * 2) // max(nch, 1) + 2
        frames = []
        for off in range(0, len(data), block_align):
            block = data[off : off + block_align]
            if len(block) < 7 * nch:
                break
            if nch == 1:
                frames.append(_decode_ms_adpcm_mono_block(block, samples_per_block, coeffs))
            else:
                frames.append(_decode_ms_adpcm_block(block, nch, samples_per_block, coeffs))
        if not frames:
            raise ValueError("empty MS-ADPCM data")
        pcm = np.concatenate(frames, axis=0)
        return _to_mono(_pcm_to_float(pcm)), int(sr)

    if audio_format == 0x0011:
        if samples_per_block <= 0:
            samples_per_block = (block_align - 4 * nch) * 2 // max(nch, 1) + 1
        if nch != 1:
            raise ValueError("IMA-ADPCM stereo not implemented")
        frames = [
            _decode_ima_adpcm_mono_block(data[off : off + block_align], samples_per_block)
            for off in range(0, len(data), block_align)
            if off + 4 <= len(data)
        ]
        pcm = np.concatenate(frames, axis=0)
        return _to_mono(_pcm_to_float(pcm)), int(sr)

    raise ValueError(f"unsupported WAVE format {audio_format}")


def load_mono_wav(audio, target_sr=TARGET_SR):
    """Decode an HF audio example without torchcodec/torchaudio/soundfile."""
    if isinstance(audio, dict) and audio.get("array") is not None:
        wav = _to_mono(audio["array"])
        sr = audio.get("sampling_rate") or target_sr
        return _resample(wav, sr, target_sr)

    source = None
    if isinstance(audio, dict):
        if audio.get("bytes"):
            source = io.BytesIO(audio["bytes"])
        elif audio.get("path"):
            source = audio["path"]
    else:
        source = audio
    if source is None:
        raise ValueError(f"Cannot decode audio payload: {type(audio)}")

    errors = []
    for reader, name in (
        (_read_wav_scipy, "scipy"),
        (_read_wav_stdlib, "wave"),
        (_read_wav_compressed, "adpcm"),
    ):
        try:
            wav, sr = reader(source)
            return _resample(wav, sr, target_sr)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            if isinstance(source, io.BytesIO):
                source.seek(0)

    raise RuntimeError("Failed to decode audio (" + "; ".join(errors) + ")")


def _normalize_label(value):
    """Return a hashable canonical label (str for text labels, int for ints)."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    return str(value)


def _train_test_split(spec, dataset, seed):
    """Resolve the (train_hf, test_hf) HF datasets given the split strategy."""
    strategy = spec["split"]
    if strategy == "native":
        test_key = "test" if "test" in dataset else "validation"
        return dataset["train"], dataset[test_key]

    # single split datasets are stored under "train"
    base = dataset["train"] if "train" in dataset else dataset[list(dataset.keys())[0]]

    if strategy.startswith("fold:"):
        test_fold = int(strategy.split(":")[1])
        folds = np.array(base["fold"])
        test_idx = np.where(folds == test_fold)[0].tolist()
        train_idx = np.where(folds != test_fold)[0].tolist()
        return base.select(train_idx), base.select(test_idx)

    if strategy == "ratio":
        splits = base.train_test_split(test_size=0.2, seed=seed)
        return splits["train"], splits["test"]

    raise ValueError(f"Unknown split strategy {strategy}")


class AudioClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, feature_extractor, label_col, label2id):
        self.ds = hf_dataset
        self.fe = feature_extractor
        self.label_col = label_col
        self.label2id = label2id

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example = self.ds[idx]
        wav = load_mono_wav(example["audio"], target_sr=TARGET_SR)
        features = self.fe(
            wav, sampling_rate=TARGET_SR, return_tensors="pt"
        )["input_values"][0]
        label = self.label2id[_normalize_label(example[self.label_col])]
        return {"input_values": features, "labels": label}


def collate_fn(batch):
    return {
        "input_values": torch.stack([b["input_values"] for b in batch]),
        "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
    }


def _dataloader_kwargs(num_workers: int) -> dict:
    """Avoid fork-after-CUDA deadlocks with HuggingFace datasets.

    ``num_workers=0`` (default) loads in the main process. If workers are
    requested, use ``spawn`` instead of the Linux default ``fork``.
    """
    kwargs = {"num_workers": max(int(num_workers), 0)}
    if kwargs["num_workers"] > 0:
        kwargs["multiprocessing_context"] = "spawn"
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def get_audio_dataloaders(
    task_name: str,
    model_name: str,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    num_workers: int = 0,
    subsample_train: int = -1,
    subsample_test: int = -1,
    seed: int = 42,
):
    """Returns (train_loader, test_loader, num_classes, label2id)."""
    if task_name not in AUDIO_TASKS:
        raise ValueError(f"Unknown task {task_name}. Available: {ALL_TASKS}")
    spec = AUDIO_TASKS[task_name]

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    dataset = _load_full(spec)
    # Keep encoded bytes/paths; decode in ``load_mono_wav`` (no torchcodec).
    for split in list(dataset.keys()):
        dataset[split] = dataset[split].cast_column(
            "audio", Audio(sampling_rate=TARGET_SR, decode=False)
        )

    train_hf, test_hf = _train_test_split(spec, dataset, seed)

    # build a label -> id map that is consistent across train and test
    label_col = spec["label_col"]
    labels = sorted(
        {_normalize_label(v) for v in train_hf[label_col]}
        | {_normalize_label(v) for v in test_hf[label_col]},
        key=lambda x: (isinstance(x, str), x),
    )
    label2id = {lab: i for i, lab in enumerate(labels)}
    num_classes = len(label2id)

    if subsample_train and subsample_train > 0:
        n = min(subsample_train, len(train_hf))
        train_hf = train_hf.shuffle(seed=seed).select(range(n))
    if subsample_test and subsample_test > 0:
        n = min(subsample_test, len(test_hf))
        test_hf = test_hf.shuffle(seed=seed).select(range(n))

    train_loader = torch.utils.data.DataLoader(
        AudioClassificationDataset(train_hf, feature_extractor, label_col, label2id),
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        **_dataloader_kwargs(num_workers),
    )
    test_loader = torch.utils.data.DataLoader(
        AudioClassificationDataset(test_hf, feature_extractor, label_col, label2id),
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **_dataloader_kwargs(num_workers),
    )
    return train_loader, test_loader, num_classes, label2id
