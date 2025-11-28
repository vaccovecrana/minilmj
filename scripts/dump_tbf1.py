import struct, torch
from transformers import AutoModel

DTYPE_MAP = {
    torch.float32: 1,
    torch.float16: 2,
    torch.float64: 3,
    torch.int64: 4,
    torch.int32: 5,
    torch.uint8: 6,
}
DTYPE_ITEMSIZE = {1: 4, 2: 2, 3: 8, 4: 8, 5: 4, 6: 1}


def dump_tbf1(model, path):
    entries, blobs = [], []
    offset = 0

    for name, p in model.named_parameters():
        arr = p.detach().cpu().contiguous().numpy()
        dt = DTYPE_MAP.get(p.dtype)
        if dt is None:
            raise TypeError(f"Unsupported dtype {p.dtype} for {name}")
        nbytes = arr.size * DTYPE_ITEMSIZE[dt]
        entries.append(
            {
                "name": name,
                "dtype": dt,
                "shape": arr.shape,
                "offset": offset,
                "nbytes": nbytes,
                "arr": arr.tobytes(order="C"),
            }
        )
        blobs.append(entries[-1]["arr"])
        offset += nbytes

    # Compute absolute start of data section: magic(4) + count(8) + per-entry headers
    data_start = 4 + 8
    for e in entries:
        name_len = len(e["name"].encode("utf-8"))
        rank = len(e["shape"])
        # name_len(2) + name + dtype(1) + rank(1) + dims(4*rank) + offset(8) + nbytes(8)
        data_start += 2 + name_len + 1 + 1 + 4 * rank + 8 + 8

    # backpatch the offsets
    for e in entries:
        e["offset"] = data_start + e["offset"]

    with open(path, "wb") as f:
        f.write(b"TBF1")  # magic + version
        f.write(struct.pack("<Q", len(entries)))  # tensor count

        for e in entries:
            name_bytes = e["name"].encode("utf-8")
            f.write(struct.pack("<H", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<B", e["dtype"]))
            rank = len(e["shape"])
            f.write(struct.pack("<B", rank))
            for d in e["shape"]:
                f.write(struct.pack("<I", d))
            f.write(struct.pack("<Q", e["offset"]))
            f.write(struct.pack("<Q", e["nbytes"]))

        for b, e in zip(blobs, entries):
            print(f"{e['name']:50} | offset={e['offset']:10} | nbytes={e['nbytes']:10}")
            f.write(b)


if __name__ == "__main__":
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    dump_tbf1(model, "bert_weights.tbf")

# Example usage:
# dump_tbf1(model, "bert_weights.tbf")
