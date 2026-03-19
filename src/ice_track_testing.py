"""
ice_track_testing.py  —  track generation for the ice boat racer.

Generates star-polygon tracks with:
  - Fence borders around track edges (cobblestone_wall)
  - Checkpoint pillars at star vertices (emerald = start, gold = rest)
  - Bridge shortcuts (low probability by default)
  - generate_tracks() wrapper compatible with MalmoBoatEnv
"""

import math
import random

RESET_BLOCK_TYPE = "lava"


def generate_star_polygon(num_points=12, inner_radius=35, outer_radius=48, center=(50, 50)):
    vertices, vertex_types = [], []
    for i in range(num_points):
        angle  = (2 * math.pi * i) / num_points
        radius = (outer_radius if i % 2 == 0 else inner_radius) + random.uniform(-3, 3)
        x = center[0] + radius * math.cos(angle)
        z = center[1] + radius * math.sin(angle)
        vertices.append((int(x), int(z)))
        vertex_types.append('outer' if i % 2 == 0 else 'inner')
    return vertices, vertex_types


def generate_bridge_connections(vertices, vertex_types, start_vertex_idx, bridge_probability=0.4):
    num_vertices = len(vertices)
    bridges, visited = [], set()
    current_idx = start_vertex_idx
    for step in range(num_vertices):
        visited.add(current_idx)
        if step == num_vertices - 1:
            break
        normal_next = (current_idx + 1) % num_vertices
        bridge_next = (current_idx + 2) % num_vertices
        middle_vertex = normal_next
        can_bridge = (
            vertex_types[current_idx] == vertex_types[bridge_next] and
            bridge_next not in visited and
            middle_vertex not in visited and
            random.random() < bridge_probability
        )
        if can_bridge:
            bridges.append((current_idx, bridge_next))
            current_idx = bridge_next
        else:
            current_idx = normal_next
    return bridges


def get_skipped_edges_and_verts(bridges, num_vertices):
    skipped_edges, skipped_verts = set(), set()
    for start_idx, end_idx in bridges:
        if (end_idx - start_idx) % num_vertices == 2:
            middle_idx = (start_idx + 1) % num_vertices
            skipped_edges.add(tuple(sorted([start_idx, middle_idx])))
            skipped_edges.add(tuple(sorted([middle_idx, end_idx])))
            skipped_verts.add(middle_idx)
    return skipped_edges, skipped_verts


def interpolate_track_segment(start, end, track_width=8):
    x0, z0 = start
    x1, z1 = end
    blocks  = []
    distance = math.sqrt((x1-x0)**2 + (z1-z0)**2)
    steps    = int(distance * 2) + 2
    for i in range(steps):
        t  = i / (steps - 1) if steps > 1 else 0
        x  = x0 + t * (x1 - x0)
        z  = z0 + t * (z1 - z0)
        dx = x1 - x0
        dz = z1 - z0
        length = math.sqrt(dx*dx + dz*dz)
        if length > 0:
            perp_x = -dz / length
            perp_z =  dx / length
            for w in range(-track_width//2, track_width//2 + 1):
                bx = int(x + w * perp_x)
                bz = int(z + w * perp_z)
                blocks.extend([(bx, bz), (bx+1, bz), (bx-1, bz),
                                (bx, bz+1), (bx, bz-1)])
    return blocks


def generate_vertex_circle(vertex, radius):
    x, z = vertex
    return [(x+dx, z+dz)
            for dx in range(-radius-1, radius+2)
            for dz in range(-radius-1, radius+2)
            if math.sqrt(dx*dx + dz*dz) <= radius]


def generate_fence_border(track_positions, layers=1):
    track_positions = set(track_positions)
    frontier        = set(track_positions)
    border_positions = set()
    neighbor_offsets = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    for _ in range(layers):
        next_frontier = set()
        for x, z in frontier:
            for dx, dz in neighbor_offsets:
                pos = (x+dx, z+dz)
                if pos not in track_positions and pos not in border_positions:
                    border_positions.add(pos)
                    next_frontier.add(pos)
        frontier = next_frontier
    return border_positions


def generate_star_race_track_with_offset(num_points=6, min_width=6, max_width=16,
                                          bridge_probability=0.05, offset_x=0):
    vertices, vertex_types = generate_star_polygon(num_points)
    vertices = [(x + offset_x, z) for x, z in vertices]

    start_vertex_idx = 0
    bridges = generate_bridge_connections(vertices, vertex_types,
                                          start_vertex_idx, bridge_probability)
    skipped_edges, skipped_verts = get_skipped_edges_and_verts(bridges, len(vertices))

    xml_blocks          = []
    all_track_positions = set()
    vertex_edge_widths  = {i: [] for i in range(len(vertices))}

    # Perimeter segments
    for i in range(len(vertices)):
        j    = (i + 1) % len(vertices)
        edge = tuple(sorted([i, j]))
        if edge in skipped_edges:
            continue
        w = random.randint(min_width, max_width)
        vertex_edge_widths[i].append(w)
        vertex_edge_widths[j].append(w)
        all_track_positions.update(interpolate_track_segment(vertices[i], vertices[j], w))

    # Bridge segments
    for si, ei in bridges:
        w = random.randint(min_width, max_width)
        vertex_edge_widths[si].append(w)
        vertex_edge_widths[ei].append(w)
        all_track_positions.update(interpolate_track_segment(vertices[si], vertices[ei], w))

    # Vertex junction circles
    for i, v in enumerate(vertices):
        if vertex_edge_widths[i]:
            r = min(vertex_edge_widths[i]) // 2
            all_track_positions.update(generate_vertex_circle(v, r))

    # Ice + air blocks
    for x, z in all_track_positions:
        xml_blocks.append(f'<DrawBlock x="{x}" y="226" z="{z}" type="packed_ice"/>')
        xml_blocks.append(f'<DrawBlock x="{x}" y="227" z="{z}" type="air"/>')
        xml_blocks.append(f'<DrawBlock x="{x}" y="228" z="{z}" type="air"/>')

    # Fence border
    fence_positions = generate_fence_border(all_track_positions, layers=1)
    for x, z in fence_positions:
        xml_blocks.append(f'<DrawBlock x="{x}" y="227" z="{z}" type="cobblestone_wall"/>')

    # Checkpoint pillars at vertices
    checkpoint_positions = []
    for i, (x, z) in enumerate(vertices):
        if i in skipped_verts:
            continue
        checkpoint_positions.append((x, z))
        bt = "emerald_block" if i == start_vertex_idx else "gold_block"
        for dx in range(-1, 2):
            for dz in range(-1, 2):
                xml_blocks.append(f'<DrawBlock x="{x+dx}" y="229" z="{z+dz}" type="{bt}"/>')
                xml_blocks.append(f'<DrawBlock x="{x+dx}" y="230" z="{z+dz}" type="{bt}"/>')

    spawn_x, spawn_z = vertices[start_vertex_idx]
    return "\n".join(xml_blocks), checkpoint_positions, (spawn_x, spawn_z)


def create_combined_tracks_mission(num_tracks=5, track_x_spacing=200):
    all_tracks_drawing = []
    tracks_data        = []

    for i in range(num_tracks):
        offset_x    = i * track_x_spacing
        num_points  = random.choice([6])
        min_width   = random.randint(5, 8)
        max_width   = random.randint(12, 20)
        bridge_prob = random.uniform(0.0, 0.1)

        track_xml, cp_pos, spawn = generate_star_race_track_with_offset(
            num_points        = num_points,
            min_width         = min_width,
            max_width         = max_width,
            bridge_probability= bridge_prob,
            offset_x          = offset_x,
        )
        all_tracks_drawing.append(track_xml)
        tracks_data.append({
            'checkpoints': cp_pos,
            'spawn_point': spawn,
            'track_xml':   track_xml,
        })

    return {
        'tracks':     tracks_data,
        'num_tracks': num_tracks,
        'draw_xml':   "\n".join(all_tracks_drawing),
    }


def generate_tracks(num_tracks=5, seed=None):
    """
    Entry point used by MalmoBoatEnv._generate_new_tracks().
    Returns {'draw_xml': str, 'tracks': [...], 'num_tracks': int}
    Each track has 'checkpoints' and 'spawn_point'.
    """
    if seed is not None:
        random.seed(seed)

    result  = create_combined_tracks_mission(num_tracks=num_tracks)
    max_x   = num_tracks * 200 + 60
    clear   = (
        f'<DrawCuboid x1="-10" y1="225" z1="-10" x2="{max_x}" y2="235" z2="110" type="air"/>\n'
        f'<DrawCuboid x1="-10" y1="225" z1="-10" x2="{max_x}" y2="225" z2="110" type="lava"/>'
    )
    return {
        'draw_xml':   clear + "\n" + result['draw_xml'],
        'tracks':     result['tracks'],
        'num_tracks': result['num_tracks'],
    }


if __name__ == "__main__":
    data = generate_tracks(num_tracks=3, seed=42)
    print(f"Tracks: {data['num_tracks']}")
    for i, t in enumerate(data['tracks']):
        print(f"  Track {i}: spawn={t['spawn_point']}  checkpoints={len(t['checkpoints'])}")
    print(f"DrawBlocks: {data['draw_xml'].count('<DrawBlock'):,}")