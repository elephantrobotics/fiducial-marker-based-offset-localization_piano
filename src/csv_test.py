import typing as T


def read_offset_table(filename: str):
    res: T.Dict[str, T.Tuple[float, float, float]] = {}
    with open(filename, "r") as f:
        t = f.read()
        t = t.strip()
        lines = t.split("\n")
        content = lines[1:]
        for line in content:
            items = line.split(",")
            name, x, y, z = items
            res[name] = (float(x), float(y), float(z))
    return res
