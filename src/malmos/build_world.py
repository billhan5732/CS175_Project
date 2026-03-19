"""
build_world.py — Run this ONCE before training.

What it does:
  1. Starts a Malmo mission with FlatWorldGenerator + all your DrawBlocks
  2. Waits as long as needed for the server to finish placing blocks
  3. Prints instructions for saving the world in Minecraft

After running this, go into Minecraft and save the world to a known folder.
Then set SAVED_WORLD_PATH in malmo_boat_env_dqn.py to that folder path.
From that point on, training missions load the pre-built world instantly —
zero DrawBlocks, zero timeout risk.

Usage:
    python build_world.py
"""

import MalmoPython
import json
import time

from ice_track_testing import create_combined_tracks_mission, RESET_BLOCK_TYPE

# How many tracks to build — set this to whatever you want for training
NUM_TRACKS    = 5
TRACK_SPACING = 120

# How long to wait for block placement — be generous, this only runs once
BUILD_WAIT_SECONDS = 120


def main():
    combined_data = create_combined_tracks_mission(
        num_tracks=NUM_TRACKS,
        track_x_spacing=TRACK_SPACING,
    )
    mission_xml  = combined_data['mission_xml']
    tracks_data  = combined_data['tracks']

    # Save track metadata to disk so the training env can load it
    # without needing to regenerate (which would give different checkpoint positions)
    import json as _json
    with open("track_data.json", "w") as f:
        _json.dump({
            'num_tracks':    NUM_TRACKS,
            'track_spacing': TRACK_SPACING,
            'tracks': [
                {
                    'checkpoints':      t['checkpoints'],
                    'spawn_point':      list(t['spawn_point']),
                    'start_vertex_idx': t['start_vertex_idx'],
                    'offset_x':         t['offset_x'],
                    'difficulty':       t['difficulty'],
                }
                for t in tracks_data
            ]
        }, f, indent=2)
    print("Track metadata saved to track_data.json")

    # Start the mission — this is the slow part, it only happens once
    agent_host     = MalmoPython.AgentHost()
    mission        = MalmoPython.MissionSpec(mission_xml, True)
    mission_record = MalmoPython.MissionRecordSpec()

    print("Starting world-building mission...")
    print(f"This may take up to {BUILD_WAIT_SECONDS}s — go make a coffee.")

    for retry in range(3):
        try:
            agent_host.startMission(mission, mission_record)
            break
        except RuntimeError as e:
            print(f"Retry {retry + 1}: {e}")
            time.sleep(10)
    else:
        raise RuntimeError("Could not start mission after 3 attempts")

    # Wait for mission to begin — no timeout here, we want to wait as long as needed
    print("Waiting for world to generate and blocks to be placed...")
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        time.sleep(0.5)
        world_state = agent_host.getWorldState()

    print("Mission started! Blocks are being placed...")
    print(f"Waiting {BUILD_WAIT_SECONDS}s for all blocks to finish placing...")

    # Wait generously — this is the only time we ever pay this cost
    for remaining in range(BUILD_WAIT_SECONDS, 0, -10):
        print(f"  {remaining}s remaining...")
        time.sleep(10)

    print("\n" + "="*60)
    print("BLOCK PLACEMENT COMPLETE")
    print("="*60)
    print()
    print("Now do the following IN MINECRAFT:")
    print("  1. Press F3 to confirm the track is visible in the world")
    print("  2. Open the game menu (Esc)")
    print("  3. Click 'Save and Quit to Title'")
    print("  4. On the main menu, click 'Singleplayer'")
    print("  5. Find the world named TEMP_10000_... and click 'Edit'")
    print("  6. Click 'Open World Folder'")
    print("  7. Copy that entire folder somewhere permanent, e.g.:")
    print("       C:/Users/coolb/CS175/CS175_Project/saved_worlds/ice_track/")
    print("  8. Set SAVED_WORLD_PATH in malmo_boat_env_dqn.py to that path")
    print()
    print("After that, training will load this world instantly every run.")

    # Keep mission alive so you can inspect the world before saving
    print("\nMission is still running. Press Ctrl+C when you have saved the world.")
    try:
        while world_state.is_mission_running:
            time.sleep(1)
            world_state = agent_host.getWorldState()
    except KeyboardInterrupt:
        print("Shutting down builder mission...")
        try:
            agent_host.sendCommand("quit")
        except Exception:
            pass

    print("Done.")


if __name__ == "__main__":
    main()