from malmo import MalmoPython
import json
import random
import math
import time
import numpy as np

from stable_baselines3 import PPO
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("PPO imported successfully!")

RESET_BLOCK_TYPE = "lava"


def generate_star_polygon(num_points=12, inner_radius=35, outer_radius=48, center=(50, 50)):
    """
    Generate a star polygon with alternating inner and outer vertices
    Returns vertices and their types (inner/outer)
    """
    vertices = []
    vertex_types = []

    for i in range(num_points):
        angle = (2 * math.pi * i) / num_points

        if i % 2 == 0:
            radius = outer_radius + random.uniform(-3, 3)
            vertex_types.append('outer')
        else:
            radius = inner_radius + random.uniform(-3, 3)
            vertex_types.append('inner')

        x = center[0] + radius * math.cos(angle)
        z = center[1] + radius * math.sin(angle)

        vertices.append((int(x), int(z)))

    return vertices, vertex_types


def generate_bridge_connections(vertices, vertex_types, start_vertex_idx, bridge_probability=0.4):
    """
    Generate bridge connections by walking along the polygon sequentially.
    At each vertex, decide whether to continue normally or create a bridge shortcut.
    This ensures the path remains continuous and all vertices are reachable.
    """
    num_vertices = len(vertices)
    bridges = []
    visited = set()
    path = []
    current_idx = start_vertex_idx

    for step in range(num_vertices):
        visited.add(current_idx)
        path.append(current_idx)

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
    """
    Determine which edges should be skipped because a bridge replaces them
    Returns a set of edge tuples to skip
    """
    skipped_edges = set()
    skipped_verts = set()

    for start_idx, end_idx in bridges:
        if (end_idx - start_idx) % num_vertices == 2:
            middle_idx = (start_idx + 1) % num_vertices

            edge1 = tuple(sorted([start_idx, middle_idx]))
            edge2 = tuple(sorted([middle_idx, end_idx]))

            skipped_edges.add(edge1)
            skipped_edges.add(edge2)

            skipped_verts.add(middle_idx)

    return skipped_edges, skipped_verts


def interpolate_track_segment(start, end, track_width=8):
    """
    Create ice blocks between two vertices to form a track segment
    Uses denser sampling to avoid holes
    """
    x0, z0 = start
    x1, z1 = end

    blocks = []

    distance = math.sqrt((x1 - x0) ** 2 + (z1 - z0) ** 2)
    steps = int(distance * 2) + 2

    for i in range(steps):
        t = i / (steps - 1) if steps > 1 else 0
        x = x0 + t * (x1 - x0)
        z = z0 + t * (z1 - z0)

        dx = x1 - x0
        dz = z1 - z0
        length = math.sqrt(dx * dx + dz * dz)

        if length > 0:
            perp_x = -dz / length
            perp_z = dx / length

            for w in range(-track_width // 2 - 1, track_width // 2 + 2):
                block_x = int(x + w * perp_x)
                block_z = int(z + w * perp_z)
                blocks.append((block_x, block_z))

                blocks.append((block_x + 1, block_z))
                blocks.append((block_x - 1, block_z))
                blocks.append((block_x, block_z + 1))
                blocks.append((block_x, block_z - 1))

    return blocks


def generate_vertex_circle(vertex, radius):
    """
    Generate a filled circle of ice blocks around a vertex
    """
    x, z = vertex
    blocks = []

    for dx in range(-radius - 1, radius + 2):
        for dz in range(-radius - 1, radius + 2):
            distance = math.sqrt(dx * dx + dz * dz)
            if distance <= radius:
                blocks.append((x + dx, z + dz))

    return blocks


def generate_fence_border(track_positions, layers=1):
    """
    Generate fence positions around the outside edge of the track.
    track_positions: set of (x, z) positions occupied by packed ice
    layers: thickness of the fence border in blocks
    """
    track_positions = set(track_positions)
    frontier = set(track_positions)
    border_positions = set()

    neighbor_offsets = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    for _ in range(layers):
        next_frontier = set()
        for x, z in frontier:
            for dx, dz in neighbor_offsets:
                nx, nz = x + dx, z + dz
                pos = (nx, nz)
                if pos not in track_positions and pos not in border_positions:
                    border_positions.add(pos)
                    next_frontier.add(pos)
        frontier = next_frontier

    return border_positions


def generate_star_race_track(num_points=12, min_width=6, max_width=16, bridge_probability=0.3):
    """
    Generate a star-shaped race track with random bridge shortcuts
    Uses a sequential walk algorithm to ensure continuous path
    """

    vertices, vertex_types = generate_star_polygon(num_points)

    start_vertex_idx = 0

    bridges = generate_bridge_connections(vertices, vertex_types, start_vertex_idx, bridge_probability)

    skipped_edges, skipped_verts = get_skipped_edges_and_verts(bridges, len(vertices))

    xml_blocks = []
    all_track_positions = set()
    segment_widths = []

    vertex_edge_widths = {i: [] for i in range(len(vertices))}

    for i in range(len(vertices)):
        start_idx = i
        end_idx = (i + 1) % len(vertices)

        edge = tuple(sorted([start_idx, end_idx]))
        if edge in skipped_edges:
            continue

        start = vertices[start_idx]
        end = vertices[end_idx]

        segment_width = random.randint(min_width, max_width)
        segment_widths.append(segment_width)

        vertex_edge_widths[start_idx].append(segment_width)
        vertex_edge_widths[end_idx].append(segment_width)

        segment_blocks = interpolate_track_segment(start, end, segment_width)
        all_track_positions.update(segment_blocks)

    bridge_widths = []
    for start_idx, end_idx in bridges:
        start = vertices[start_idx]
        end = vertices[end_idx]

        bridge_width = random.randint(min_width, max_width)
        bridge_widths.append(bridge_width)

        vertex_edge_widths[start_idx].append(bridge_width)
        vertex_edge_widths[end_idx].append(bridge_width)

        bridge_blocks = interpolate_track_segment(start, end, bridge_width)
        all_track_positions.update(bridge_blocks)

    for i, vertex in enumerate(vertices):
        if vertex_edge_widths[i]:
            min_edge_width = min(vertex_edge_widths[i])
            circle_radius = min_edge_width // 2

            circle_blocks = generate_vertex_circle(vertex, circle_radius)
            all_track_positions.update(circle_blocks)

    fence_positions = generate_fence_border(all_track_positions, layers=1)

    for x, z in all_track_positions:
        xml_blocks.append(f'<DrawBlock x="{x}" y="226" z="{z}" type="packed_ice"/>')
        xml_blocks.append(f'<DrawBlock x="{x}" y="227" z="{z}" type="air"/>')
        xml_blocks.append(f'<DrawBlock x="{x}" y="228" z="{z}" type="air"/>')

    for x, z in fence_positions:
        xml_blocks.append(f'<DrawBlock x="{x}" y="227" z="{z}" type="fence"/>')
        xml_blocks.append(f'<DrawBlock x="{x}" y="228" z="{z}" type="fence"/>')

    checkpoint_positions = []
    for i, (x, z) in enumerate(vertices):
        if i in skipped_verts:
            continue
        checkpoint_positions.append((x, z))

        block_type = "emerald_block" if i == start_vertex_idx else "gold_block"

        for dx in range(-1, 2):
            for dz in range(-1, 2):
                xml_blocks.append(f'<DrawBlock x="{x + dx}" y="229" z="{z + dz}" type="{block_type}"/>')
                xml_blocks.append(f'<DrawBlock x="{x + dx}" y="230" z="{z + dz}" type="{block_type}"/>')

    spawn_x, spawn_z = vertices[start_vertex_idx]

    return "\n".join(xml_blocks), checkpoint_positions, segment_widths, bridges, (spawn_x, spawn_z), start_vertex_idx, list(fence_positions)


def generate_star_race_track_with_offset(num_points=12, min_width=6, max_width=16, bridge_probability=0.3, offset_x=0):
    """
    Generate a star-shaped race track with optional X offset applied during generation
    Much faster than parsing XML after the fact
    """

    vertices, vertex_types = generate_star_polygon(num_points)

    vertices = [(x + offset_x, z) for x, z in vertices]

    start_vertex_idx = 0

    bridges = generate_bridge_connections(vertices, vertex_types, start_vertex_idx, bridge_probability)

    skipped_edges, skipped_verts = get_skipped_edges_and_verts(bridges, len(vertices))

    xml_blocks = []
    all_track_positions = set()
    segment_widths = []

    vertex_edge_widths = {i: [] for i in range(len(vertices))}

    for i in range(len(vertices)):
        start_idx = i
        end_idx = (i + 1) % len(vertices)

        edge = tuple(sorted([start_idx, end_idx]))
        if edge in skipped_edges:
            continue

        start = vertices[start_idx]
        end = vertices[end_idx]

        segment_width = random.randint(min_width, max_width)
        segment_widths.append(segment_width)

        vertex_edge_widths[start_idx].append(segment_width)
        vertex_edge_widths[end_idx].append(segment_width)

        segment_blocks = interpolate_track_segment(start, end, segment_width)
        all_track_positions.update(segment_blocks)

    bridge_widths = []
    for start_idx, end_idx in bridges:
        start = vertices[start_idx]
        end = vertices[end_idx]

        bridge_width = random.randint(min_width, max_width)
        bridge_widths.append(bridge_width)

        vertex_edge_widths[start_idx].append(bridge_width)
        vertex_edge_widths[end_idx].append(bridge_width)

        bridge_blocks = interpolate_track_segment(start, end, bridge_width)
        all_track_positions.update(bridge_blocks)

    for i, vertex in enumerate(vertices):
        if vertex_edge_widths[i]:
            min_edge_width = min(vertex_edge_widths[i])
            circle_radius = min_edge_width // 2
            circle_blocks = generate_vertex_circle(vertex, circle_radius)
            all_track_positions.update(circle_blocks)

    fence_positions = generate_fence_border(all_track_positions, layers=1)

    for x, z in all_track_positions:
        xml_blocks.append(f'<DrawBlock x="{x}" y="226" z="{z}" type="packed_ice"/>')
        xml_blocks.append(f'<DrawBlock x="{x}" y="227" z="{z}" type="air"/>')
        xml_blocks.append(f'<DrawBlock x="{x}" y="228" z="{z}" type="air"/>')

    for x, z in fence_positions:
        xml_blocks.append(f'<DrawBlock x="{x}" y="227" z="{z}" type="cobblestone_wall"/>')

    checkpoint_positions = []
    for i, (x, z) in enumerate(vertices):
        if i in skipped_verts:
            continue
        checkpoint_positions.append((x, z))

        block_type = "emerald_block" if i == start_vertex_idx else "gold_block"

        for dx in range(-1, 2):
            for dz in range(-1, 2):
                xml_blocks.append(f'<DrawBlock x="{x + dx}" y="229" z="{z + dz}" type="{block_type}"/>')
                xml_blocks.append(f'<DrawBlock x="{x + dx}" y="230" z="{z + dz}" type="{block_type}"/>')

    spawn_x, spawn_z = vertices[start_vertex_idx]

    return "\n".join(xml_blocks), checkpoint_positions, segment_widths, bridges, (spawn_x, spawn_z), start_vertex_idx, list(fence_positions)


def create_combined_tracks_mission(num_tracks=5, track_x_spacing=200):
    """
    Generate multiple tracks at different X positions in a single mission.
    Much faster version - applies offset during generation instead of parsing XML
    """
    all_tracks_drawing = []
    tracks_data = []

    for i in range(num_tracks):
        offset_x = i * track_x_spacing

        num_points = random.choice([6])
        min_width = random.randint(5, 8)
        max_width = random.randint(12, 20)
        bridge_prob = random.uniform(0.0, 0.1)

        track_xml, cp_pos, seg_widths, bridges, spawn, start_idx, wall_positions = generate_star_race_track_with_offset(
            num_points=num_points,
            min_width=min_width,
            max_width=max_width,
            bridge_probability=bridge_prob,
            offset_x=offset_x
        )

        all_tracks_drawing.append(track_xml)

        tracks_data.append({
            'checkpoints': cp_pos,
            'spawn_point': spawn,
            'segment_widths': seg_widths,
            'bridges': bridges,
            'start_vertex_idx': start_idx,
            'offset_x': offset_x,
            'track_xml': track_xml,
            'wall_positions': wall_positions,
            'wall_positions': wall_positions,
            'difficulty': {
                'num_points': num_points,
                'min_width': min_width,
                'max_width': max_width,
                'num_bridges': len(bridges)
            }
        })

    first_spawn_x, first_spawn_z = tracks_data[0]['spawn_point']

    max_x = num_tracks * track_x_spacing + 150

    combined_track_xml = "\n".join(all_tracks_drawing)

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
        'mission_xml': mission_xml,
        'tracks': tracks_data,
        'num_tracks': num_tracks,
        'track_spacing': track_x_spacing
    }


def create_mission_xml(track_xml, spawn_point, seed=None):
    """
    Create the full mission XML with the generated track
    """
    if seed is None:
        seed = random.randint(0, 999999)

    spawn_x, spawn_z = spawn_point

    return f'''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <About>
            <Summary>Ice Boat Racing Training Environment - Star Track with Bridges</Summary>
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
                    <DrawCuboid x1="-50" y1="225" z1="-50" x2="150" y2="255" z2="150" type="air"/>
                    <DrawCuboid x1="-50" y1="225" z1="-50" x2="150" y2="224" z2="150" type="lava"/>

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
                    <Range name="entities" xrange="10" yrange="2" zrange="10" />
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
    """
    Generate multiple varied star track environments with bridges
    """
    environments = []

    for i in range(num_envs):
        num_points = 6
        min_width = random.randint(10, 12)
        max_width = random.randint(13, 15)
        bridge_prob = random.uniform(0.0, 0.1)

        track_xml, cp_pos, seg_widths, bridges, spawn, start_idx, wall_positions = generate_star_race_track(
            num_points=num_points,
            min_width=min_width,
            max_width=max_width,
            bridge_probability=bridge_prob
        )

        mission_xml = create_mission_xml(track_xml, spawn, seed=i)

        environments.append({
            'mission_xml': mission_xml,
            'checkpoints': cp_pos,
            'segment_widths': seg_widths,
            'bridges': bridges,
            'spawn_point': spawn,
            'start_vertex_idx': start_idx,
            'difficulty': {
                'num_points': num_points,
                'min_width': min_width,
                'max_width': max_width,
                'num_bridges': len(bridges)
            }
        })

    return environments


if __name__ == "__main__":
    envs = create_varied_environments(5)

    print(f"Generated star track with {len(envs[0]['checkpoints'])} checkpoints")
    print(f"Starting vertex: {envs[0]['start_vertex_idx']} (marked with EMERALD)")
    print(f"Number of bridge shortcuts: {len(envs[0]['bridges'])}")
    print(f"Bridge connections: {envs[0]['bridges']}")

    agent_host = MalmoPython.AgentHost()

    my_mission = MalmoPython.MissionSpec(envs[0]['mission_xml'], True)
    my_mission_record = MalmoPython.MissionRecordSpec()

    try:
        agent_host.startMission(my_mission, my_mission_record)
    except RuntimeError as e:
        print(f"Error starting mission: {e}")
        exit(1)

    print("Waiting for mission to start...")
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()

    print("Mission started! Star track with bridge shortcuts - CONTINUOUS PATH!")
    print("Camera positioned directly above, looking down.")
    print("EMERALD block = starting checkpoint")
    print("GOLD blocks = other checkpoints")
    print("Press CTRL+C to exit.")