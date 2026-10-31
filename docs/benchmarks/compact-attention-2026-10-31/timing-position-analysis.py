"""Position-versus-version attribution of the FLOAT32 warmed timing pairs.

Usage: python3 timing-position-analysis.py float32-results.json
"""
import json
import statistics as st
import sys

FAMILIES = ("prism", "nova", "loop")
REVERSED_ROUND = 1


def measured(process):
    return [x["seconds"] for x in process["timings"] if x["phase"] != "warmup"]


def median_ms(process):
    return st.median(measured(process)) * 1000


def order(round_index):
    return ("candidate", "baseline") if round_index == REVERSED_ROUND else ("baseline", "candidate")


def pick(timings, family, round_index, version):
    return next(p for p in timings if p["family"] == family
                and p["round"] == round_index and p["version"] == version)


def load(path):
    return [p for p in json.load(open(path))["processes"] if p["mode"] == "timing"]


def pair_effects(timings):
    for fam in FAMILIES:
        for r in range(3):
            first, second = order(r)
            a, b = median_ms(pick(timings, fam, r, first)), median_ms(pick(timings, fam, r, second))
            position = (b / a - 1) * 100
            yield fam, r, position, position if r != REVERSED_ROUND else -position


def print_pairs(timings):
    print("round family  2nd-vs-1st%  new-vs-old%")
    positions, versions = [], []
    for fam, r, position, version in pair_effects(timings):
        positions.append(position)
        versions.append(version)
        print(f"{r}     {fam:6}  {position:+10.3f}  {version:+10.3f}")
    print(f"\nmean 2nd-vs-1st {st.mean(positions):+.3f}%  mean new-vs-old {st.mean(versions):+.3f}%")
    print(f"positive with position {sum(x > 0 for x in positions)}/9, with version {sum(x > 0 for x in versions)}/9")


def print_drift(timings):
    print("\nfamily rnd pos version   first10_ms last10_ms  drift%")
    for fam in FAMILIES:
        for r in range(3):
            for pos, ver in zip(("1st", "2nd"), order(r)):
                m = measured(pick(timings, fam, r, ver))
                a, b = st.median(m[:10]) * 1000, st.median(m[-10:]) * 1000
                print(f"{fam:6} {r}   {pos} {ver:9} {a:9.3f} {b:9.3f}  {(b / a - 1) * 100:+.3f}")


if __name__ == "__main__":
    data = load(sys.argv[1])
    print_pairs(data)
    print_drift(data)

