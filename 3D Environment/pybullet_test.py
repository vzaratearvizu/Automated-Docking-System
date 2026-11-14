# pybullet_water_test.py
# Adds simple buoyancy physics to the raft placeholder: water is modeled as a
# flat plane at z=0, with an upward force applied each step based on how much
# of the raft is currently submerged below that plane (a simplified version
# of Archimedes' principle). This replaces pybullet_test.py's plain gravity
# drop -- the raft should now settle and float, not sink to the ground plane.

import math
import pybullet as p
import pybullet_data
import time

physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Pool floor, pushed well BELOW the water surface -- previously this sat at
# z=0, nearly the same height as the water decal, so the two blended into
# one flat-looking surface with no visible depth between them.
POOL_DEPTH = 3.0
ground_id = p.loadURDF("plane.urdf", basePosition=[0, 0, -POOL_DEPTH])

# -- Water setup --
WATER_LEVEL = 0.0        # z-height of the water's surface
WATER_DENSITY = 1000.0   # kg/m^3 -- real water's density
GRAVITY_MAG = 9.8

# A solid-looking blue VOLUME (no collision shape -- purely visual), filling
# the gap from the pool floor up to the water's surface. A thin flat decal
# at just z=0 read as part of the ground; a block with real height reads as
# an actual body of water you can see things sink into.
water_visual_shape = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=[5, 5, POOL_DEPTH / 2],
    rgbaColor=[0.1, 0.4, 0.8, 0.5],
)
water_id = p.createMultiBody(
    baseMass=0,  # mass 0 = static, never affected by physics
    baseVisualShapeIndex=water_visual_shape,
    basePosition=[0, 0, WATER_LEVEL - POOL_DEPTH / 2],  # top face lands exactly at z=0
)

# -- Hull setup --
# A boat-shaped hull instead of a plain rectangular box: flat stern (back),
# straight side walls through the midsection, then two angled panels that
# taper inward to a point at the bow (front) -- the classic simple-boat
# silhouette. PyBullet has no hull/boat primitive, so this is still built
# entirely from flat box pieces -- the bow "point" is just two rotated
# rectangular panels meeting at a line, not a smoothly curved surface.
#
# Bow points toward +X. Overall OUTER bounding size: length 1.5, width 0.7,
# height 0.6 (not counting wall thickness).
OUTER_HALF_LENGTH = 0.75   # x
OUTER_HALF_WIDTH = 0.35    # y
OUTER_HALF_HEIGHT = 0.3    # z
WALL_THICKNESS = 0.04
BOW_START_X = 0.2  # where the straight side walls end and the taper begins

# These are the outer BOUNDING half-extents (as if the hull were a plain
# box) -- used by apply_buoyancy_forces() below. Since the real hull is
# narrower near the bow than this bounding box, buoyancy will slightly
# overestimate force contributed near the bow corners (there's no actual
# hull material there to displace water) -- a known, acceptable
# simplification, same spirit as treating the hollow interior as solid for
# buoyancy purposes.
RAFT_HALF_EXTENTS = [OUTER_HALF_LENGTH, OUTER_HALF_WIDTH, OUTER_HALF_HEIGHT]

IDENTITY_ORIENTATION = [0, 0, 0, 1]  # no rotation

# -- Midsection side walls (straight, from the stern to BOW_START_X) --
side_wall_half_length = (BOW_START_X + OUTER_HALF_LENGTH) / 2
side_wall_center_x = (-OUTER_HALF_LENGTH + BOW_START_X) / 2
side_wall_y = OUTER_HALF_WIDTH - WALL_THICKNESS / 2

# -- Bow panels (angled, converging from BOW_START_X to the tip at the bow) --
# Each panel runs from (BOW_START_X, +-side_wall_y) to (OUTER_HALF_LENGTH, 0).
# panel_angle is the direction of that line, used to rotate the panel so its
# own length axis points along it -- this is what creates the taper.
dx = OUTER_HALF_LENGTH - BOW_START_X
dy = side_wall_y
panel_length = math.hypot(dx, dy)
panel_angle = math.atan2(-dy, dx)  # for the +Y panel; -Y panel mirrors this

panel_center_x = (BOW_START_X + OUTER_HALF_LENGTH) / 2
panel_center_y = side_wall_y / 2

# Six pieces total: floor, stern wall, 2 side walls, 2 bow panels.
shape_types = [p.GEOM_BOX] * 6
half_extents_list = [
    [OUTER_HALF_LENGTH, OUTER_HALF_WIDTH, WALL_THICKNESS / 2],          # floor
    [WALL_THICKNESS / 2, OUTER_HALF_WIDTH, OUTER_HALF_HEIGHT],          # stern wall (flat back)
    [side_wall_half_length, WALL_THICKNESS / 2, OUTER_HALF_HEIGHT],     # +Y side wall
    [side_wall_half_length, WALL_THICKNESS / 2, OUTER_HALF_HEIGHT],     # -Y side wall
    [panel_length / 2, WALL_THICKNESS / 2, OUTER_HALF_HEIGHT],          # +Y bow panel
    [panel_length / 2, WALL_THICKNESS / 2, OUTER_HALF_HEIGHT],          # -Y bow panel
]
frame_positions = [
    [0, 0, -OUTER_HALF_HEIGHT + WALL_THICKNESS / 2],                    # floor
    [-OUTER_HALF_LENGTH + WALL_THICKNESS / 2, 0, 0],                    # stern wall
    [side_wall_center_x, side_wall_y, 0],                               # +Y side wall
    [side_wall_center_x, -side_wall_y, 0],                              # -Y side wall
    [panel_center_x, panel_center_y, 0],                                # +Y bow panel
    [panel_center_x, -panel_center_y, 0],                               # -Y bow panel (mirrored)
]
frame_orientations = [
    IDENTITY_ORIENTATION,  # floor
    IDENTITY_ORIENTATION,  # stern wall
    IDENTITY_ORIENTATION,  # +Y side wall
    IDENTITY_ORIENTATION,  # -Y side wall
    p.getQuaternionFromEuler([0, 0, panel_angle]),   # +Y bow panel, angled inward
    p.getQuaternionFromEuler([0, 0, -panel_angle]),  # -Y bow panel, mirrored
]

hull_collision_shape = p.createCollisionShapeArray(
    shapeTypes=shape_types,
    halfExtents=half_extents_list,
    collisionFramePositions=frame_positions,
    collisionFrameOrientations=frame_orientations,
)
hull_visual_shape = p.createVisualShapeArray(
    shapeTypes=shape_types,
    halfExtents=half_extents_list,
    visualFramePositions=frame_positions,
    visualFrameOrientations=frame_orientations,
    rgbaColors=[[0.6, 0.3, 0.1, 1.0]] * 6,  # brown, so it reads as a solid hull, not water
)

RAFT_MASS = 120.0  # smaller than the plain-box version's 300 -- this hull's
                    # outer volume is less than half as large, so a similar
                    # relative floating depth needs proportionally less mass.
                    # Tune by feel once you see where it actually settles.

raft_id = p.createMultiBody(
    baseMass=RAFT_MASS,
    baseCollisionShapeIndex=hull_collision_shape,
    baseVisualShapeIndex=hull_visual_shape,
    basePosition=[0, 0, 4],  # starts high up, room to visibly drop and splash
)

# Damping lowered from the first version -- that one was damped enough to
# kill almost all oscillation. Lower damping lets it actually bounce/bob a
# few visible times before settling, which is the whole point of this test.
p.changeDynamics(raft_id, -1, linearDamping=0.15, angularDamping=0.2)


def apply_buoyancy_forces(body_id, half_extents, water_level, water_density, gravity):
    """Orientation-aware buoyancy. The earlier version computed one single
    force from the center, always treating half_extents[2] as "straight up"
    regardless of how the hull was actually rotated -- so flipping it upside
    down or sideways changed nothing, since the math never looked at the
    real orientation at all.

    This version instead samples all 8 corners of the hull's outer bounding
    box, converts each to its REAL current world position (accounting for
    rotation), and checks each one's own depth below the waterline
    independently. Whichever corners are actually lowest in world space --
    regardless of which corners those originally were -- are the ones that
    end up submerged and pushing back. A tilted hull then naturally gets
    more upward force on whichever side is pushed deeper, producing a
    self-righting torque, the same way a real boat resists tipping over.

    This is an approximation (it distributes force by depth-FRACTION per
    corner, not an exact computed submerged volume for a tilted box, which
    is a much harder geometry problem) -- but it's correct in the one case
    that matters most for a sanity check: when fully submerged, all 8
    corners contribute their full share and the total force exactly equals
    water_density * gravity * the hull's full volume, matching real
    Archimedes' principle for a fully-submerged object, regardless of
    orientation.

    Because the force is now distributed differently than the old center-
    only method, floating depth at the same mass WILL be different than
    before -- RAFT_MASS will likely need retuning by feel again."""
    hx, hy, hz = half_extents
    full_height = hz * 2
    hull_volume = (hx * 2) * (hy * 2) * (hz * 2)
    volume_share_per_corner = hull_volume / 8

    position, orientation = p.getBasePositionAndOrientation(body_id)

    local_corners = [
        [sx * hx, sy * hy, sz * hz]
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ]

    for local_corner in local_corners:
        world_corner, _ = p.multiplyTransforms(
            position, orientation, local_corner, [0, 0, 0, 1]
        )

        corner_depth = water_level - world_corner[2]
        corner_depth = max(0.0, min(corner_depth, full_height))  # clamp to [0, full_height]
        depth_fraction = corner_depth / full_height

        corner_force = water_density * gravity * depth_fraction * volume_share_per_corner
        p.applyExternalForce(
            body_id,
            -1,
            forceObj=[0, 0, corner_force],
            posObj=world_corner,
            flags=p.WORLD_FRAME,
        )


# -- Simulation loop --
for step in range(20000):
    apply_buoyancy_forces(
        raft_id, RAFT_HALF_EXTENTS, WATER_LEVEL, WATER_DENSITY, GRAVITY_MAG
    )

    p.stepSimulation()
    time.sleep(1 / 240)

p.disconnect()