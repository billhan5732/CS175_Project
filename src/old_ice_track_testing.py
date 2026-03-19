import MalmoPython
import json
import random
import math
import time

RESET_BLOCK_TYPE = "lava"

# Hard cap — if generation exceeds this, params are too aggressive
MAX_DRAW_BLOCKS = 18_000


def generate_star_polygon(num_points=8, inner_radius=12, outer_radius=22, center=(50, 50)):
    """
    Smaller default radii (was 15/30) keep segment lengths short.
    Shorter segments = fewer steps in interpolate_track_segment = fewer blocks.
    num_points default reduced from 12 to 8.
    """
    vertices = []
    vertex_types = []

    for i in range(num_points):
        angle = (2 * math.pi * i) / num_points

        if i % 2 == 0:
            radius = outer_radius + random.uniform(-1, 1)
            vertex_types.append('outer')
        else:
            radius = inner_radius + random.uniform(-1, 1)
            vertex_types.append('inner')

        x = center[0] + radius * math.cos(angle)
        z = center[1] + radius * math.sin(angle)
        vertices.append((int(x), int(z)))

    return vertices, vertex_types


def generate_bridge_connections(vertices, vertex_types, start_vertex_idx, bridge_probability=0.4):
    num_vertices = len(vertices)
    bridges = []
    visited = set()
    current_idx = start_vertex_idx

    for step in range(num_vertices):
        visited.add(current_idx)
        if step == num_vertices - 1:
            break

        normal_next   = (current_idx + 1) % num_vertices
        bridge_next   = (current_idx + 2) % num_vertices
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
    skipped_edges = set()
    skipped_verts = set()

    for start_idx, end_idx in bridges:
        if (end_idx - start_idx) % num_vertices == 2:
            middle_idx = (start_idx + 1) % num_vertices
            skipped_edges.add(tuple(sorted([start_idx, middle_idx])))
            skipped_edges.add(tuple(sorted([middle_idx, end_idx])))
            skipped_verts.add(middle_idx)

    return skipped_edges, skipped_verts


def interpolate_track_segment(start, end, track_width=5):
    """
    Uses round() instead of int() to avoid diagonal gaps.
    One step per block of distance (no 2x oversampling).
    No adjacency padding — set() handles deduplication.
    track_width default reduced from 6 to 5.
    """
    x0, z0 = start
    x1, z1 = end

    blocks = set()

    distance = math.sqrt((x1 - x0) ** 2 + (z1 - z0) ** 2)
    steps = int(distance) + 1

    dx = x1 - x0
    dz = z1 - z0
    length = math.sqrt(dx * dx + dz * dz)

    if length == 0:
        return list(blocks)

    perp_x = -dz / length
    perp_z =  dx / length

    for i in range(steps):
        t = i / (steps - 1) if steps > 1 else 0
        cx = x0 + t * (x1 - x0)
        cz = z0 + t * (z1 - z0)

        for w in range(-track_width // 2, track_width // 2 + 1):
            bx = round(cx + w * perp_x)
            bz = round(cz + w * perp_z)
            blocks.add((bx, bz))

    return list(blocks)


def generate_vertex_circle(vertex, radius):
    x, z = vertex
    blocks = set()
    for dx in range(-radius - 1, radius + 2):
        for dz in range(-radius - 1, radius + 2):
            if math.sqrt(dx * dx + dz * dz) <= radius:
                blocks.add((x + dx, z + dz))
    return list(blocks)


def _build_track_xml(vertices, vertex_types, start_vertex_idx,
                     min_width=3, max_width=6, bridge_probability=0.3):
    """Shared core for all track generation variants."""
    bridges = generate_bridge_connections(
        vertices, vertex_types, start_vertex_idx, bridge_probability
    )
    skipped_edges, skipped_verts = get_skipped_edges_and_verts(bridges, len(vertices))

    all_track_positions = set()
    segment_widths      = []
    vertex_edge_widths  = {i: [] for i in range(len(vertices))}

    # Perimeter segments
    for i in range(len(vertices)):
        end_idx = (i + 1) % len(vertices)
        edge    = tuple(sorted([i, end_idx]))
        if edge in skipped_edges:
            continue
        w = random.randint(min_width, max_width)
        segment_widths.append(w)
        vertex_edge_widths[i].append(w)
        vertex_edge_widths[end_idx].append(w)
        all_track_positions.update(
            interpolate_track_segment(vertices[i], vertices[end_idx], w)
        )

    # Bridge segments
    for start_idx, end_idx in bridges:
        w = random.randint(min_width, max_width)
        vertex_edge_widths[start_idx].append(w)
        vertex_edge_widths[end_idx].append(w)
        all_track_positions.update(
            interpolate_track_segment(vertices[start_idx], vertices[end_idx], w)
        )

    # Junction circles at each vertex
    for i, vertex in enumerate(vertices):
        if vertex_edge_widths[i]:
            r = min(vertex_edge_widths[i]) // 2
            all_track_positions.update(generate_vertex_circle(vertex, r))

    xml_blocks = []

    # 3 DrawBlocks per unique (x,z): ice floor + 2 air layers
    for x, z in all_track_positions:
        xml_blocks.append(f'<DrawBlock x="{x}" y="226" z="{z}" type="packed_ice"/>')
        xml_blocks.append(f'<DrawBlock x="{x}" y="227" z="{z}" type="air"/>')
        xml_blocks.append(f'<DrawBlock x="{x}" y="228" z="{z}" type="air"/>')

    # Checkpoint markers
    checkpoint_positions = []
    for i, (x, z) in enumerate(vertices):
        if i in skipped_verts:
            continue
        checkpoint_positions.append((x, z))
        block_type = "emerald_block" if i == start_vertex_idx else "gold_block"
        for dx in range(-1, 2):
            for dz in range(-1, 2):
                xml_blocks.append(
                    f'<DrawBlock x="{x+dx}" y="229" z="{z+dz}" type="{block_type}"/>'
                )
                xml_blocks.append(
                    f'<DrawBlock x="{x+dx}" y="230" z="{z+dz}" type="{block_type}"/>'
                )

    spawn_x, spawn_z = vertices[start_vertex_idx]
    return (
        "\n".join(xml_blocks),
        checkpoint_positions,
        segment_widths,
        bridges,
        (spawn_x, spawn_z),
    )


def generate_star_race_track(num_points=8, min_width=3, max_width=6, bridge_probability=0.3):
    vertices, vertex_types = generate_star_polygon(num_points)
    start_vertex_idx = 0
    track_xml, cp_pos, seg_widths, bridges, spawn = _build_track_xml(
        vertices, vertex_types, start_vertex_idx, min_width, max_width, bridge_probability
    )
    return track_xml, cp_pos, seg_widths, bridges, spawn, start_vertex_idx


def generate_star_race_track_with_offset(num_points=8, min_width=3, max_width=6,
                                          bridge_probability=0.3, offset_x=0):
    vertices, vertex_types = generate_star_polygon(num_points)
    vertices = [(x + offset_x, z) for x, z in vertices]
    start_vertex_idx = 0
    track_xml, cp_pos, seg_widths, bridges, spawn = _build_track_xml(
        vertices, vertex_types, start_vertex_idx, min_width, max_width, bridge_probability
    )
    return track_xml, cp_pos, seg_widths, bridges, spawn, start_vertex_idx


def create_combined_tracks_mission(num_tracks=5, track_x_spacing=120):
    """
    Generate multiple tracks in one mission.

    Parameter ranges are deliberately conservative so that even the
    worst-case random roll stays well under MAX_DRAW_BLOCKS.

    Worst-case estimate per track:
      num_points=10, max_width=6, outer_radius=22
      ~10 segments * ~35 blocks long * 7 wide * 3 layers = ~7,350 blocks
      5 tracks * 7,350 = ~36,750  → still hits ceiling
      With num_points capped at 8:
      ~8 segments * ~30 blocks long * 7 wide * 3 layers = ~5,040 blocks
      5 tracks * 5,040 = ~25,200  → fine with deduplication reducing it further

    The assert below will catch any unlucky roll before Minecraft is touched.
    """
    all_tracks_drawing = []
    tracks_data = []

    for i in range(num_tracks):
        offset_x = i * track_x_spacing

        # Tightly capped ranges — do NOT increase max_width above 6
        # or num_points above 8 without re-verifying block counts
        num_points  = random.choice([6, 8])       # was up to 12
        min_width   = random.randint(3, 4)         # was up to 6
        max_width   = random.randint(5, 6)         # was up to 12
        bridge_prob = random.uniform(0.2, 0.4)

        track_xml, cp_pos, seg_widths, bridges, spawn, start_idx = \
            generate_star_race_track_with_offset(
                num_points=num_points,
                min_width=min_width,
                max_width=max_width,
                bridge_probability=bridge_prob,
                offset_x=offset_x,
            )

        all_tracks_drawing.append(track_xml)
        tracks_data.append({
            'checkpoints':      cp_pos,
            'spawn_point':      spawn,
            'segment_widths':   seg_widths,
            'bridges':          bridges,
            'start_vertex_idx': start_idx,
            'offset_x':         offset_x,
            'difficulty': {
                'num_points':  num_points,
                'min_width':   min_width,
                'max_width':   max_width,
                'num_bridges': len(bridges),
            },
        })

    first_spawn_x, first_spawn_z = tracks_data[0]['spawn_point']
    max_x = num_tracks * track_x_spacing + 100
    combined_track_xml = "\n".join(all_tracks_drawing)

    draw_block_count = combined_track_xml.count('<DrawBlock')
    print(f"[Track Gen] Tracks: {num_tracks} | DrawBlocks: {draw_block_count:,}")

    # Hard stop — better to crash here than hang Minecraft for 2 minutes
    assert draw_block_count <= MAX_DRAW_BLOCKS, (
        f"DrawBlock count {draw_block_count:,} exceeds limit {MAX_DRAW_BLOCKS:,}. "
        f"Tighten track params before launching."
    )

    mission_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <About>
            <Summary>Ice Boat Racing Training - All Tracks Combined</Summary>
        </About>

        <ServerSection>
            <ServerInitialConditions>
                <Time>
                    <StartTime>6000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
                <AllowSpawning>false</AllowSpawning>
            </ServerInitialConditions>
            <ServerHandlers>
                <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1"/>
                <DrawingDecorator>
                    <DrawCuboid x1="-50" y1="225" z1="-150" x2="{max_x}" y2="255" z2="150" type="air"/>
                    <DrawCuboid x1="-50" y1="225" z1="-150" x2="{max_x}" y2="225" z2="150" type="{RESET_BLOCK_TYPE}"/>
                    {combined_track_xml}
                </DrawingDecorator>
            </ServerHandlers>
        </ServerSection>

        <AgentSection mode="Creative">
            <Name>IceBoatRacer</Name>
            <AgentStart>
                <Placement x="{first_spawn_x}" y="227" z="{first_spawn_z}" pitch="0" yaw="0"/>
            </AgentStart>
            <AgentHandlers>
                <ObservationFromFullStats/>
                <ObservationFromNearbyEntities>
                    <Range name="entities" xrange="10" yrange="2" zrange="10"/>
                </ObservationFromNearbyEntities>
                <ObservationFromGrid>
                    <Grid name="nearby_blocks">
                        <min x="-3" y="-1" z="-3"/>
                        <max x="3" y="1" z="3"/>
                    </Grid>
                </ObservationFromGrid>
                <HumanLevelCommands/>
                <ChatCommands/>
                <AbsoluteMovementCommands>
                    <ModifierList type="allow-list">
                        <command>tp</command>
                        <command>setYaw</command>
                        <command>setPitch</command>
                    </ModifierList>
                </AbsoluteMovementCommands>
                <AgentQuitFromReachingCommandQuota total="0"/>
            </AgentHandlers>
        </AgentSection>
    </Mission>'''

    return {
        'mission_xml':   mission_xml,
        'tracks':        tracks_data,
        'num_tracks':    num_tracks,
        'track_spacing': track_x_spacing,
    }


def create_mission_xml(track_xml, spawn_point, seed=None):
    if seed is None:
        seed = random.randint(0, 999999)
    spawn_x, spawn_z = spawn_point

    return f'''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <About>
            <Summary>Ice Boat Racing Training Environment</Summary>
        </About>
        <ServerSection>
            <ServerInitialConditions>
                <Time><StartTime>6000</StartTime><AllowPassageOfTime>false</AllowPassageOfTime></Time>
                <Weather>clear</Weather>
                <AllowSpawning>false</AllowSpawning>
            </ServerInitialConditions>
            <ServerHandlers>
                <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1"/>
                <DrawingDecorator>
                    <DrawCuboid x1="-50" y1="225" z1="-50" x2="150" y2="255" z2="150" type="air"/>
                    <DrawCuboid x1="-50" y1="225" z1="-50" x2="150" y2="225" z2="150" type="lava"/>
                    {track_xml}
                    <DrawEntity x="{spawn_x}" y="227" z="{spawn_z}" type="Boat"/>
                </DrawingDecorator>
            </ServerHandlers>
        </ServerSection>
        <AgentSection mode="Creative">
            <Name>IceBoatRacer</Name>
            <AgentStart>
                <Placement x="{spawn_x}" y="227" z="{spawn_z}" pitch="90" yaw="0"/>
            </AgentStart>
            <AgentHandlers>
                <ObservationFromFullStats/>
                <ObservationFromNearbyEntities>
                    <Range name="entities" xrange="10" yrange="2" zrange="10"/>
                </ObservationFromNearbyEntities>
                <ObservationFromGrid>
                    <Grid name="nearby_blocks">
                        <min x="-3" y="-1" z="-3"/>
                        <max x="3" y="1" z="3"/>
                    </Grid>
                </ObservationFromGrid>
                <HumanLevelCommands/>
                <ChatCommands/>
                <AbsoluteMovementCommands>
                    <ModifierList type="allow-list">
                        <command>tp</command>
                        <command>setYaw</command>
                        <command>setPitch</command>
                    </ModifierList>
                </AbsoluteMovementCommands>
                <MissionQuitCommands/>
                <AgentQuitFromReachingCommandQuota total="0"/>
            </AgentHandlers>
        </AgentSection>
    </Mission>'''


def create_varied_environments(num_envs=10):
    environments = []
    for i in range(num_envs):
        num_points  = random.choice([6, 8])
        min_width   = random.randint(3, 4)
        max_width   = random.randint(5, 6)
        bridge_prob = random.uniform(0.2, 0.4)

        track_xml, cp_pos, seg_widths, bridges, spawn, start_idx = generate_star_race_track(
            num_points=num_points,
            min_width=min_width,
            max_width=max_width,
            bridge_probability=bridge_prob,
        )
        mission_xml = create_mission_xml(track_xml, spawn, seed=i)
        environments.append({
            'mission_xml':      mission_xml,
            'checkpoints':      cp_pos,
            'segment_widths':   seg_widths,
            'bridges':          bridges,
            'spawn_point':      spawn,
            'start_vertex_idx': start_idx,
            'difficulty': {
                'num_points':  num_points,
                'min_width':   min_width,
                'max_width':   max_width,
                'num_bridges': len(bridges),
            },
        })
    return environments


if __name__ == "__main__":
    # Sanity check — run this without Minecraft to verify block counts
    for trial in range(5):
        data  = create_combined_tracks_mission(num_tracks=5)
        count = data['mission_xml'].count('<DrawBlock')
        print(f"Trial {trial+1}: {count:,} DrawBlocks — {'OK' if count <= MAX_DRAW_BLOCKS else 'TOO HIGH'}")