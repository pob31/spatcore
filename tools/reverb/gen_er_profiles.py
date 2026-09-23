"""Early-reflection profiles for effects/modules/reverb/EarlyReflections.h.

An image-source model of four shoebox rooms. For each room: every image
source up to third order, its delay RELATIVE TO THE DIRECT SOUND (the dry
path is the direct sound, so a reflection is how much later it arrives) and
its gain (spherical spreading against the direct path times a reflection
coefficient per bounce - the floor has its own, since an audience or pews
absorb what a bare wall returns).

Images arriving within MIN_MS of the direct sound are dropped: they fuse
with it rather than being heard as reflections, and in a mono effect they
only comb-filter it. So are images later than the room's window: past it a
reflection belongs to the tail the reverb model already makes. Inside the
window the first-order images come first, then the STRONGEST higher-order
ones, up to the room's tap count - skipping any within SPACING_MS of a tap
already chosen (the floor-bounce twin of a wall reflection, half a
millisecond behind it, would fuse with it and comb) - and the set is sorted
by arrival. Gains are normalised so the taps' energy is 1/4: after
the reverb's fixed wet make-up of 2, reflections at ER Level 0 dB carry the
dry's energy.

Run it and paste the output over the tables in
effects/modules/reverb/EarlyReflections.h:

    python tools/reverb/gen_er_profiles.py

The numbers are a starting point for listening, not a measurement of any
real room.
"""
import itertools
import math

C = 343.0          # m/s
MAX_ORDER = 3
MIN_MS = 2.0
SPACING_MS = 1.0

# name, dimensions (m), source, listener, walls, floor, taps, window (ms)
ROOMS = [
    ("Room",      (6.2, 4.8, 3.0),    (2.0, 1.6, 1.4),   (4.3, 3.1, 1.2),   0.78, 0.70, 16,  30.0),
    ("Chamber",   (9.5, 6.5, 4.2),    (2.3, 1.9, 1.6),   (6.8, 4.1, 1.3),   0.88, 0.85, 18,  40.0),
    ("Hall",      (36.0, 24.0, 16.0), (6.0, 12.5, 1.8),  (22.0, 9.0, 1.2),  0.82, 0.30, 20, 120.0),
    ("Cathedral", (72.0, 28.0, 30.0), (12.0, 14.0, 2.0), (45.0, 10.0, 1.5), 0.90, 0.45, 24, 200.0),
]


def images(dims, src):
    """(position, order, floor bounces) of every image up to MAX_ORDER.

    Along one axis an image is (1 - 2p) s + 2 m L, bouncing |m - p| times
    off the wall at L and |m| times off the wall at 0; on the vertical axis
    the wall at 0 is the floor."""
    axes = []
    for L, s in zip(dims, src):
        per_axis = []
        for m in range(-MAX_ORDER, MAX_ORDER + 1):
            for p in (0, 1):
                far, near = abs(m - p), abs(m)
                if far + near <= MAX_ORDER:
                    per_axis.append(((1 - 2 * p) * s + 2 * m * L, far + near, near))
        axes.append(per_axis)
    for (x, ox, _), (y, oy, _), (z, oz, floor) in itertools.product(*axes):
        order = ox + oy + oz
        if 1 <= order <= MAX_ORDER:
            yield (x, y, z), order, floor


def profile(dims, src, lis, walls, floor, taps, window):
    direct = math.dist(src, lis)
    found = []
    for pos, order, floors in images(dims, src):
        r = math.dist(pos, lis)
        ms = (r - direct) / C * 1000.0
        if MIN_MS <= ms <= window:
            g = (direct / r) * walls ** (order - floors) * floor ** floors
            found.append((ms, g, order))
    chosen = []
    for t in sorted(found, key=lambda t: (t[2] > 1, -t[1])):   # first order first, then by strength
        if len(chosen) == taps:
            break
        if all(abs(t[0] - c[0]) >= SPACING_MS for c in chosen):
            chosen.append(t)
    assert len(chosen) == taps
    chosen.sort()
    norm = math.sqrt(0.25 / sum(g * g for _, g, _ in chosen))
    return [(ms, g * norm, order) for ms, g, order in chosen]


def main():
    for name, dims, src, lis, walls, floor, taps, window in ROOMS:
        rows = profile(dims, src, lis, walls, floor, taps, window)
        print("// %s: %.1f x %.1f x %.1f m, walls %.2f, floor %.2f"
              % (name, dims[0], dims[1], dims[2], walls, floor))
        print("inline constexpr ErTapSpec kEr%s[] =" % name)
        print("{")
        for ms, g, order in rows:
            print("    { %8.3ff, %.5ff, %d }," % (ms, g, order))
        print("};")
        print()


if __name__ == "__main__":
    main()
