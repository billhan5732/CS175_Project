"""
ice_track_testing.py

The caller (MalmoBoatEnv) injects the returned XML string into the
mission XML template via {PLACEHOLDER_TRACK_XML}, just like the maze
example injects {PLACEHOLDER_MAZESEED}.
"""

import math
import random

RESET_BLOCK_TYPE = "lava"
MAX_DRAW_BLOCKS  = 18_000


# -----------------------------------------------------------------------------
# Shape generation
# -----------------------------------------------------------------------------

def generate_star_polygon(num_points=8, inner_radius=12, outer_radius=22,
                           center=(50, 50)):
    vertices     = []
    vertex_types = []
    for i in range(num_points):
        angle  = (2 * math.pi * i) / num_points
        radius = (outer_radius if i % 2 == 0 else inner_radius) + random.uniform(-1, 1)
        vtype  = 'outer' if i % 2 == 0 else 'inner'
        x = center[0] + radius * math.cos(angle)
        z = center[1] + radius * math.sin(angle)
        vertices.append((int(x), int(z)))
        vertex_types.append(vtype)
    return vertices, vertex_types


def generate_bridge_connections(vertices, vertex_types, start_idx,
                                 bridge_probability=0.3):
    n       = len(vertices)
    bridges = []
    visited = set()
    current = start_idx
    for _ in range(n):
        visited.add(current)
        nxt    = (current + 1) % n
        bridge = (current + 2) % n
        if (vertex_types[current] == vertex_types[bridge]
                and bridge not in visited
                and nxt not in visited
                and random.random() < bridge_probability):
            bridges.append((current, bridge))
            current = bridge
        else:
            current = nxt
    return bridges


def get_skipped(bridges, n):
    skipped_edges = set()
    skipped_verts = set()
    for s, e in bridges:
        if (e - s) % n == 2:
            mid = (s + 1) % n
            skipped_edges.add(tuple(sorted([s, mid])))
            skipped_edges.add(tuple(sorted([mid, e])))
            skipped_verts.add(mid)
    return skipped_edges, skipped_verts


# -----------------------------------------------------------------------------
# Block interpolation
# -----------------------------------------------------------------------------

def interpolate_segment(start, end, width=5):
    """
    Interpolate ice blocks along a segment.
    round() instead of int() prevents diagonal gaps.
    One step per block distance — no 2x oversampling.
    """
    x0, z0 = start
    x1, z1 = end
    dist   = math.sqrt((x1 - x0) ** 2 + (z1 - z0) ** 2)
    steps  = int(dist) + 1
    length = dist
    if length == 0:
        return set()
    perp_x = -(z1 - z0) / length
    perp_z =  (x1 - x0) / length
    blocks = set()
    for i in range(steps):
        t  = i / (steps - 1) if steps > 1 else 0
        cx = x0 + t * (x1 - x0)
        cz = z0 + t * (z1 - z0)
        for w in range(-width // 2, width // 2 + 1):
            blocks.add((round(cx + w * perp_x), round(cz + w * perp_z)))
    return blocks


def vertex_circle(vertex, radius):
    x, z   = vertex
    blocks = set()
    for dx in range(-radius - 1, radius + 2):
        for dz in range(-radius - 1, radius + 2):
            if math.sqrt(dx * dx + dz * dz) <= radius:
                blocks.add((x + dx, z + dz))
    return blocks


# -----------------------------------------------------------------------------
# Single track builder — returns (draw_xml_str, checkpoints, spawn)
# -----------------------------------------------------------------------------

def build_track(num_points=8, min_width=3, max_width=6,
                bridge_probability=0.3, offset_x=0):
    """
    Generate one star-shaped track.
    Returns:
        draw_xml  : str   — DrawBlock XML lines for this track
        checkpoints: list — [(x,z), ...] checkpoint world positions
        spawn     : tuple — (x, z) spawn position
    """
    vertices, vtypes = generate_star_polygon(num_points)
    vertices = [(x + offset_x, z) for x, z in vertices]
    start    = 0
    bridges  = generate_bridge_connections(vertices, vtypes, start,
                                           bridge_probability)
    skipped_edges, skipped_verts = get_skipped(bridges, len(vertices))

    all_blocks         = set()
    vertex_edge_widths = {i: [] for i in range(len(vertices))}

    # Perimeter segments
    for i in range(len(vertices)):
        end_idx = (i + 1) % len(vertices)
        edge    = tuple(sorted([i, end_idx]))
        if edge in skipped_edges:
            continue
        w = random.randint(min_width, max_width)
        vertex_edge_widths[i].append(w)
        vertex_edge_widths[end_idx].append(w)
        all_blocks.update(interpolate_segment(vertices[i], vertices[end_idx], w))

    # Bridge segments
    for s, e in bridges:
        w = random.randint(min_width, max_width)
        vertex_edge_widths[s].append(w)
        vertex_edge_widths[e].append(w)
        all_blocks.update(interpolate_segment(vertices[s], vertices[e], w))

    # Junction circles
    for i, v in enumerate(vertices):
        if vertex_edge_widths[i]:
            r = min(vertex_edge_widths[i]) // 2
            all_blocks.update(vertex_circle(v, r))

    # Build XML lines
    lines = []
    for x, z in all_blocks:
        lines.append(f'<DrawBlock x="{x}" y="226" z="{z}" type="packed_ice"/>')
        lines.append(f'<DrawBlock x="{x}" y="227" z="{z}" type="air"/>')
        lines.append(f'<DrawBlock x="{x}" y="228" z="{z}" type="air"/>')

    # Checkpoints at vertices
    checkpoints = []
    for i, (x, z) in enumerate(vertices):
        if i in skipped_verts:
            continue
        checkpoints.append((x, z))
        btype = "emerald_block" if i == start else "gold_block"
        for dx in range(-1, 2):
            for dz in range(-1, 2):
                lines.append(f'<DrawBlock x="{x+dx}" y="229" z="{z+dz}" type="{btype}"/>')
                lines.append(f'<DrawBlock x="{x+dx}" y="230" z="{z+dz}" type="{btype}"/>')

    spawn_x, spawn_z = vertices[start]
    return "\n".join(lines), checkpoints, (spawn_x, spawn_z)


# -----------------------------------------------------------------------------
# Multi-track builder — the main entry point
# -----------------------------------------------------------------------------

def generate_tracks(num_tracks=5, track_x_spacing=120):
    """
    Generate all tracks and return everything the env needs.

    Returns dict:
        draw_xml      : str   — combined DrawBlock XML for ALL tracks
        lava_xml      : str   — lava floor + air clear cuboids
        tracks        : list  — [{checkpoints, spawn_point}, ...]
        num_tracks    : int
        first_spawn   : tuple — (x, z) of track 0 spawn
    """
    max_x = num_tracks * track_x_spacing + 100

    # Clear area + lava floor — two cuboids, instant regardless of size
    lava_xml = (
        f'<DrawCuboid x1="-50" y1="225" z1="-150" '
        f'x2="{max_x}" y2="255" z2="150" type="air"/>\n'
        f'<DrawCuboid x1="-50" y1="225" z1="-150" '
        f'x2="{max_x}" y2="225" z2="150" type="{RESET_BLOCK_TYPE}"/>'
    )

    all_track_xml = []
    tracks        = []

    for i in range(num_tracks):
        offset_x    = i * track_x_spacing
        num_points  = random.choice([6, 8])
        min_width   = random.randint(3, 4)
        max_width   = random.randint(5, 6)
        bridge_prob = random.uniform(0.2, 0.4)

        xml, checkpoints, spawn = build_track(
            num_points        = num_points,
            min_width         = min_width,
            max_width         = max_width,
            bridge_probability= bridge_prob,
            offset_x          = offset_x,
        )
        all_track_xml.append(xml)
        tracks.append({
            'checkpoints': checkpoints,
            'spawn_point': spawn,
        })

    combined_xml  = lava_xml + "\n" + "\n".join(all_track_xml)
    block_count   = combined_xml.count('<DrawBlock')
    print(f"[Track Gen] Tracks: {num_tracks} | DrawBlocks: {block_count:,}")

    assert block_count <= MAX_DRAW_BLOCKS, (
        f"DrawBlock count {block_count:,} exceeds limit {MAX_DRAW_BLOCKS:,}. "
        f"Tighten track params."
    )

    return {
        'draw_xml':   combined_xml,
        'tracks':     tracks,
        'num_tracks': num_tracks,
        'first_spawn': tracks[0]['spawn_point'],
    }


if __name__ == "__main__":
    # Sanity check without Minecraft
    for trial in range(5):
        data  = generate_tracks(num_tracks=5)
        count = data['draw_xml'].count('<DrawBlock')
        print(f"Trial {trial+1}: {count:,} DrawBlocks")