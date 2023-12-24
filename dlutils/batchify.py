
def batchify(iterable, bsz, *, drop_last=False):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == bsz:
            yield batch
            batch = []
    if len(batch) > 0 and not drop_last:
        yield batch
