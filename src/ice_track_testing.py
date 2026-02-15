import MalmoPython
import json
import random
import math
import time

from stable_baselines3 import PPO

print("PPO imported successfully!")

RESET_BLOCK_TYPE = "lava"


def generate_star_polygon(num_points=12, inner_radius=25, outer_radius=55, center=(50, 50)):
    """
    Generate a star polygon with alternating inner and outer vertices
    Returns vertices and their types (inner/outer)
    """
    vertices = []
    vertex_types = []  # 'outer' or 'inner'

    for i in range(num_points):
        angle = (2 * math.pi * i) / num_points

        # Alternate between inner and outer radius for star effect
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
    path = []  # Track the actual path we take
    current_idx = start_vertex_idx

    # Walk around the entire polygon
    for step in range(num_vertices):
        visited.add(current_idx)
        path.append(current_idx)

        # If this is the last step, we're done
        if step == num_vertices - 1:
            break

        # Default next vertex (following the polygon)
        normal_next = (current_idx + 1) % num_vertices

        # Check if we can create a bridge (skip to vertex after next)
        bridge_next = (current_idx + 2) % num_vertices
        middle_vertex = normal_next

        # Only consider a bridge if:
        # 1. Same vertex type (outer to outer, or inner to inner)
        # 2. We haven't visited the bridge destination yet
        # 3. We haven't visited the middle vertex yet (important!)
        # 4. Random chance succeeds
        can_bridge = (
                vertex_types[current_idx] == vertex_types[bridge_next] and
                bridge_next not in visited and
                middle_vertex not in visited and  # NEW: Don't skip already visited vertices
                random.random() < bridge_probability
        )

        if can_bridge:
            # Create a bridge shortcut
            bridges.append((current_idx, bridge_next))
            # Move to the bridge destination (skipping the middle vertex)
            current_idx = bridge_next
        else:
            # Follow the normal polygon edge
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
        # When we bridge from vertex i to vertex i+2, we skip edges:
        # - (i, i+1) and (i+1, i+2)

        # Make sure we're handling adjacent same-type vertices
        if (end_idx - start_idx) % num_vertices == 2:
            middle_idx = (start_idx + 1) % num_vertices

            # Add both edges that are being bypassed by the bridge
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

    # Calculate distance and steps - use MORE steps to fill holes
    distance = math.sqrt((x1 - x0) ** 2 + (z1 - z0) ** 2)
    steps = int(distance * 2) + 2  # Double the steps to make it denser

    for i in range(steps):
        t = i / (steps - 1) if steps > 1 else 0
        x = x0 + t * (x1 - x0)
        z = z0 + t * (z1 - z0)

        # Create track width perpendicular to the direction
        dx = x1 - x0
        dz = z1 - z0
        length = math.sqrt(dx * dx + dz * dz)

        if length > 0:
            # Perpendicular direction
            perp_x = -dz / length
            perp_z = dx / length

            # Draw track width - add extra blocks around the edges to fill gaps
            for w in range(-track_width // 2 - 1, track_width // 2 + 2):
                block_x = int(x + w * perp_x)
                block_z = int(z + w * perp_z)
                blocks.append((block_x, block_z))

                # Add adjacent blocks to really fill holes
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

    # Use circle equation to fill the area
    for dx in range(-radius - 1, radius + 2):
        for dz in range(-radius - 1, radius + 2):
            # Check if point is within the circle
            distance = math.sqrt(dx * dx + dz * dz)
            if distance <= radius:
                blocks.append((x + dx, z + dz))

    return blocks

    return blocks


def generate_star_race_track(num_points=12, min_width=6, max_width=16, bridge_probability=0.3):
    """
    Generate a star-shaped race track with random bridge shortcuts
    Uses a sequential walk algorithm to ensure continuous path
    """

    # Generate star vertices
    vertices, vertex_types = generate_star_polygon(num_points)

    # Choose a random starting vertex
    start_vertex_idx = 0

    # Generate bridge connections by walking the polygon
    bridges = generate_bridge_connections(vertices, vertex_types, start_vertex_idx, bridge_probability)

    # Determine which edges to skip
    skipped_edges, skipped_verts = get_skipped_edges_and_verts(bridges, len(vertices))

    xml_blocks = []
    all_track_positions = set()
    segment_widths = []

    # Track the width of edges connected to each vertex
    vertex_edge_widths = {i: [] for i in range(len(vertices))}

    # 1. Generate main perimeter track segments (excluding skipped edges)
    for i in range(len(vertices)):
        start_idx = i
        end_idx = (i + 1) % len(vertices)

        # Check if this edge should be skipped
        edge = tuple(sorted([start_idx, end_idx]))
        if edge in skipped_edges:
            continue  # Skip this edge - it's replaced by a bridge

        start = vertices[start_idx]
        end = vertices[end_idx]

        # Randomize width for this segment
        segment_width = random.randint(min_width, max_width)
        segment_widths.append(segment_width)

        # Track widths for each vertex
        vertex_edge_widths[start_idx].append(segment_width)
        vertex_edge_widths[end_idx].append(segment_width)

        segment_blocks = interpolate_track_segment(start, end, segment_width)
        all_track_positions.update(segment_blocks)

    # 2. Generate bridge segments (shortcuts)
    bridge_widths = []
    for start_idx, end_idx in bridges:
        start = vertices[start_idx]
        end = vertices[end_idx]

        # Bridges can have different widths too
        bridge_width = random.randint(min_width, max_width)
        bridge_widths.append(bridge_width)

        # Track widths for each vertex
        vertex_edge_widths[start_idx].append(bridge_width)
        vertex_edge_widths[end_idx].append(bridge_width)

        bridge_blocks = interpolate_track_segment(start, end, bridge_width)
        all_track_positions.update(bridge_blocks)

    # 3. Add circular ice patches at each vertex
    for i, vertex in enumerate(vertices):
        # Get the minimum width of edges connected to this vertex
        if vertex_edge_widths[i]:
            min_edge_width = min(vertex_edge_widths[i])
            circle_radius = min_edge_width // 2

            # Generate circle of ice around this vertex
            circle_blocks = generate_vertex_circle(vertex, circle_radius)
            all_track_positions.update(circle_blocks)

    # 4. Draw all ice blocks
    for x, z in all_track_positions:
        xml_blocks.append(f'<DrawBlock x="{x}" y="226" z="{z}" type="packed_ice"/>')
        xml_blocks.append(f'<DrawBlock x="{x}" y="227" z="{z}" type="air"/>')
        xml_blocks.append(f'<DrawBlock x="{x}" y="228" z="{z}" type="air"/>')

    # 5. Place checkpoints (goals) at vertices
    checkpoint_positions = []
    for i, (x, z) in enumerate(vertices):

        if i in skipped_verts:
            continue
        checkpoint_positions.append((x, z))

        # Make starting checkpoint a different color (emerald block)
        block_type = "emerald_block" if i == start_vertex_idx else "gold_block"

        # Draw a 3x3 area of blocks at each vertex
        for dx in range(-1, 2):
            for dz in range(-1, 2):
                xml_blocks.append(f'<DrawBlock x="{x + dx}" y="229" z="{z + dz}" type="{block_type}"/>')
                # Make checkpoint taller so it's visible
                xml_blocks.append(f'<DrawBlock x="{x + dx}" y="230" z="{z + dz}" type="{block_type}"/>')

    # Use starting vertex position for spawn
    spawn_x, spawn_z = vertices[start_vertex_idx]

    return "\n".join(xml_blocks), checkpoint_positions, segment_widths, bridges, (spawn_x, spawn_z), start_vertex_idx


def generate_star_race_track_with_offset(num_points=12, min_width=6, max_width=16, bridge_probability=0.3, offset_x=0):
    """
    Generate a star-shaped race track with optional X offset applied during generation
    Much faster than parsing XML after the fact
    """

    # Generate star vertices with offset already applied
    vertices, vertex_types = generate_star_polygon(num_points)

    # Apply offset to vertices immediately
    vertices = [(x + offset_x, z) for x, z in vertices]

    # Choose a random starting vertex
    start_vertex_idx = 0

    # Generate bridge connections by walking the polygon
    bridges = generate_bridge_connections(vertices, vertex_types, start_vertex_idx, bridge_probability)

    # Determine which edges to skip
    skipped_edges, skipped_verts = get_skipped_edges_and_verts(bridges, len(vertices))

    xml_blocks = []
    all_track_positions = set()
    segment_widths = []

    # Track the width of edges connected to each vertex
    vertex_edge_widths = {i: [] for i in range(len(vertices))}

    # 1. Generate main perimeter track segments (excluding skipped edges)
    for i in range(len(vertices)):
        start_idx = i
        end_idx = (i + 1) % len(vertices)

        # Check if this edge should be skipped
        edge = tuple(sorted([start_idx, end_idx]))
        if edge in skipped_edges:
            continue

        start = vertices[start_idx]
        end = vertices[end_idx]

        # Randomize width for this segment
        segment_width = random.randint(min_width, max_width)
        segment_widths.append(segment_width)

        # Track widths for each vertex
        vertex_edge_widths[start_idx].append(segment_width)
        vertex_edge_widths[end_idx].append(segment_width)

        segment_blocks = interpolate_track_segment(start, end, segment_width)
        all_track_positions.update(segment_blocks)

    # 2. Generate bridge segments (shortcuts)
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

    # 3. Add circular ice patches at each vertex
    for i, vertex in enumerate(vertices):
        if vertex_edge_widths[i]:
            min_edge_width = min(vertex_edge_widths[i])
            circle_radius = min_edge_width // 2
            circle_blocks = generate_vertex_circle(vertex, circle_radius)
            all_track_positions.update(circle_blocks)

    # 4. Draw all ice blocks
    for x, z in all_track_positions:
        xml_blocks.append(f'<DrawBlock x="{x}" y="226" z="{z}" type="packed_ice"/>')
        xml_blocks.append(f'<DrawBlock x="{x}" y="227" z="{z}" type="air"/>')
        xml_blocks.append(f'<DrawBlock x="{x}" y="228" z="{z}" type="air"/>')

    # 5. Place checkpoints (goals) at vertices
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

    # Use starting vertex position for spawn
    spawn_x, spawn_z = vertices[start_vertex_idx]

    return "\n".join(xml_blocks), checkpoint_positions, segment_widths, bridges, (spawn_x, spawn_z), start_vertex_idx


def create_combined_tracks_mission(num_tracks=5, track_x_spacing=200):
    """
    Generate multiple tracks at different X positions in a single mission.
    Much faster version - applies offset during generation instead of parsing XML
    """
    all_tracks_drawing = []
    tracks_data = []

    # Generate each track with offset applied during generation
    for i in range(num_tracks):
        offset_x = i * track_x_spacing

        # Generate track with varied parameters
        num_points = random.choice([8, 10, 12, 14, 16])
        min_width = random.randint(5, 8)
        max_width = random.randint(12, 20)
        bridge_prob = random.uniform(0.3, 0.6)

        # Generate track with offset already applied - much faster!
        track_xml, cp_pos, seg_widths, bridges, spawn, start_idx = generate_star_race_track_with_offset(
            num_points=num_points,
            min_width=min_width,
            max_width=max_width,
            bridge_probability=bridge_prob,
            offset_x=offset_x  # Apply offset during generation
        )

        all_tracks_drawing.append(track_xml)

        tracks_data.append({
            'checkpoints': cp_pos,  # Already offsetted
            'spawn_point': spawn,  # Already offsetted
            'segment_widths': seg_widths,
            'bridges': bridges,
            'start_vertex_idx': start_idx,
            'offset_x': offset_x,
            'track_xml': track_xml,
            'difficulty': {
                'num_points': num_points,
                'min_width': min_width,
                'max_width': max_width,
                'num_bridges': len(bridges)
            }
        })

    # Get first spawn for initial placement
    first_spawn_x, first_spawn_z = tracks_data[0]['spawn_point']

    # Calculate world bounds
    max_x = num_tracks * track_x_spacing + 150

    # Build combined mission XML
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
                    <!-- Clear large area for all tracks -->
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
                    <!-- Clear the area first -->
                    <DrawCuboid x1="-50" y1="225" z1="-50" x2="150" y2="255" z2="150" type="air"/>
                    <DrawCuboid x1="-50" y1="225" z1="-50" x2="150" y2="224" z2="150" type="lava"/>

                    {track_xml}
                    <!-- Spawn boat at starting checkpoint -->
                    <DrawEntity x="{spawn_x}" y="227" z="{spawn_z}" type="Boat"/>
                </DrawingDecorator>
                <!-- <ServerQuitFromTimeUp timeLimitMs="120000"/> -->
                <!-- REMOVE OR COMMENT OUT ServerQuitWhenAnyAgentFinishes -->
                <!-- <ServerQuitWhenAnyAgentFinishes/> -->
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
        # Vary difficulty
        num_points = random.choice([8, 10, 12, 14, 16])  # Must be even for star
        min_width = random.randint(5, 8)
        max_width = random.randint(12, 20)
        bridge_prob = random.uniform(0.3, 0.6)  # 30-60% chance of bridges

        track_xml, cp_pos, seg_widths, bridges, spawn, start_idx = generate_star_race_track(
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


# Example usage
if __name__ == "__main__":
    # Generate 5 training environments
    envs = create_varied_environments(5)

    # Print info about the first environment
    print(f"Generated star track with {len(envs[0]['checkpoints'])} checkpoints")
    print(f"Starting vertex: {envs[0]['start_vertex_idx']} (marked with EMERALD)")
    print(f"Number of bridge shortcuts: {len(envs[0]['bridges'])}")
    print(f"Bridge connections: {envs[0]['bridges']}")

    # Start a mission with the first environment
    agent_host = MalmoPython.AgentHost()

    my_mission = MalmoPython.MissionSpec(envs[0]['mission_xml'], True)
    my_mission_record = MalmoPython.MissionRecordSpec()

    # Launch the mission
    try:
        agent_host.startMission(my_mission, my_mission_record)
    except RuntimeError as e:
        print(f"Error starting mission: {e}")
        exit(1)

    # Wait for the mission to start
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

    # Keep the mission running
    # try:
    #    while world_state.is_mission_running:
    #        time.sleep(0.1)
