"""CS 3630 Project 4: Autonomous Warehouse Challenge.

Meet RB-3630, a small three-omni-wheel robot on the night shift of a 100 m x
50 m fulfillment center, the warehouse of Chapter 4. Every mission starts the
same way: the robot wakes up somewhere on the floor with no idea where it is,
and an order comes in. Your job is to write its brain: eight short functions
from Chapter 4, "Warehouse Robots in 2D", of *Introduction to Robotics and
Perception* by Dellaert and Hutchinson
(https://www.roboticsbook.org/S40_logistics_intro.html). Part 1 is a particle
filter (motion model, sensor likelihoods, prediction, update, and the
estimate); Part 2 is value iteration. Put them together and the robot
localizes itself, drives to a storage rack, picks up a parcel, and delivers it
to a shipping dock, live in a 3D Webots warehouse. Everything else is
provided: the map and helpers, the proximity sensor model, the policy
extraction (``greedy_action``, ``vote_action``), and the staff engine
(physics, graphics, HUD) that calls your code.

Conventions (the same as the book's ``gtbook.logistics``)
---------------------------------------------------------
* A position is a point ``(x, y)`` in meters: x runs east along the 100 m
  side and y runs north along the 50 m side.
* Grid maps (``BASE_MAP``, the proximity maps, beliefs, value functions) are
  arrays of shape ``(50, 100)`` indexed ``[i, j] = [row, col] = [y, x]``.
  Cell ``(i, j)`` is centered at the point ``(x=j, y=i)``, and a point
  ``(x, y)`` lies in cell ``(round(y), round(x))``. Points are (x, y) but
  cells are (row, col) = (y, x); mixing them up is the number one bug in this
  project!
* A batch of points or particles is an array of shape ``(N, 2)`` with one
  ``(x, y)`` per row.
* Actions move one cell: 0 = WEST (j - 1), 1 = EAST (j + 1), 2 = SOUTH
  (i - 1), 3 = NORTH (i + 1), in the same order as the book's ``Q_value``.

How to work
-----------
* Follow ``project4.ipynb``: it explains every TODO, tests it with one
  ``check(...)`` cell, and draws pictures with your functions.
* Implement the TODOs in order. Each docstring says exactly what to compute,
  which book section and code it mirrors, and gives a hint. You may call the
  provided helpers and functions and your own earlier TODOs. Use NumPy and
  the standard library only.
* From a terminal in this folder, ``python -m pytest -q`` runs the public
  tests and ``python webots/support/run_mission.py`` plays a whole mission.
* Watch your robot in 3D: open ``webots/worlds/warehouse_challenge.wbt`` in
  Webots (the setup and controls are in ``project4.ipynb``).
* Submit only this file, ``project4.py``, to Gradescope. Keep every provided
  name and signature unchanged.

Points (50 for this code, all autograded; the reflection report is another 50)
-----------------------------------------------------------------------------
    Task                              Book section               Points
    0   student_info                  -                               1
    1   sample_motion_model           4.2.5                           5
    2   rfid_likelihood               4.3.2-4.3.3, 4.4.4.4            6
    3   zone_likelihood               4.3.1, 4.4.4.4                  5
    4   mcl_predict                   4.4.4.1                         6
    5   mcl_update                    4.4.4.4                         5
    6   estimate_position             4.1.3-4.1.4                     6
    7   q_value                       4.5.1                           6
    8   value_iteration               4.5.1                           7
    -   offline mission test (3 fixed missions, on Gradescope)        3
        Total                                                        50

Each function's points are split between its public tests (in
``webots/support/test_project4.py``) and hidden tests of edge cases. Every edge case is
described in the function's docstring: the hidden tests never check anything
the docstring does not say.
"""

from __future__ import annotations

import math

import numpy as np

###############################################################################
#                     PROVIDED CONSTANTS (do not modify)                      #
###############################################################################

MAP_HEIGHT = 50  # rows = y cells (1 m each)
MAP_WIDTH = 100  # columns = x cells (1 m each)

# MAP_COORDS[i, j] == (x, y) == (j, i): the center of every grid cell.
_X, _Y = np.meshgrid(np.arange(MAP_WIDTH, dtype=float), np.arange(MAP_HEIGHT, dtype=float))
MAP_COORDS = np.dstack((_X, _Y))

# Four blocks of storage shelves, exactly as in gtbook.logistics.base_map.
SHELF_WIDTH = 9
SHELF_GAP = 8
_SPACING = SHELF_WIDTH + SHELF_GAP
_SIDE = (MAP_WIDTH - SHELF_WIDTH * 4 - SHELF_GAP * 3) // 2  # 20

BASE_MAP = np.zeros((MAP_HEIGHT, MAP_WIDTH))
for _k in range(4):
    BASE_MAP[10:40, _SIDE + _k * _SPACING:_SIDE + SHELF_WIDTH + _k * _SPACING] = 1

# Eight RFID beacons at both ends of every shelf block (gtbook.logistics.beacons).
BEACONS = np.array(
    [[SHELF_WIDTH / 2 + _SIDE + _k * _SPACING - 0.5, 9.5] for _k in range(4)]
    + [[SHELF_WIDTH / 2 + _SIDE + _k * _SPACING - 0.5, 39.5] for _k in range(4)]
)

# Proximity sensor maps (gtbook.logistics.proximity_map_on / _off).
PROXIMITY_RANGE = 2  # cells
PROXIMITY_MAP_ON = np.zeros((MAP_HEIGHT, MAP_WIDTH))
for _k in range(4):
    PROXIMITY_MAP_ON[10 - PROXIMITY_RANGE:40 + PROXIMITY_RANGE,
                     _SIDE + _k * _SPACING - PROXIMITY_RANGE:
                     _SIDE + SHELF_WIDTH + _k * _SPACING + PROXIMITY_RANGE] = 1
PROXIMITY_MAP_ON[0:PROXIMITY_RANGE] = 1
PROXIMITY_MAP_ON[-PROXIMITY_RANGE:] = 1
PROXIMITY_MAP_ON[:, -PROXIMITY_RANGE:] = 1
PROXIMITY_MAP_ON[:, 0:PROXIMITY_RANGE] = 1
for _k in range(4):
    PROXIMITY_MAP_ON[10:40, _SIDE + _k * _SPACING:_SIDE + SHELF_WIDTH + _k * _SPACING] = 0

PROXIMITY_MAP_OFF = np.ones((MAP_HEIGHT, MAP_WIDTH))
PROXIMITY_MAP_OFF[PROXIMITY_MAP_ON == 1] = 0
PROXIMITY_MAP_OFF[BASE_MAP == 1] = 0

# Sensor and robot parameters (Sections 4.2 and 4.3).
RFID_MAX_RANGE = 8.0  # meters
RFID_SIGMA = 1.0  # meters, range noise used by the likelihood
# The zone reader: an infrared (IR) zone beacon in every named area of the
# warehouse broadcasts that area's id. IR does not pass through shelves or walls,
# so the robot hears the beacon of the zone it is in (near the edge of a zone it
# sometimes hears the next one). ZONE_MAP[i, j] is the zone of cell (i, j), -1 for
# shelf cells; ZONE_NAMES[k] is the name of zone k. RECEIVING (west of shelf A) and
# SHIPPING (east of shelf D) are cut into 10 m x 10 m bays: bands 1-5 from the
# south, W and E halves.
ZONE_NAMES = tuple(f"RECEIVING {b}{side}" for b in range(1, 6) for side in "WE") + (
    "AISLE 1", "AISLE 2", "AISLE 3", "CROSS AISLE S", "CROSS AISLE N") + tuple(
    f"SHIPPING {b}{side}" for b in range(1, 6) for side in "WE")
NUM_ZONES = len(ZONE_NAMES)  # 25
ZONE_MAP = np.full((MAP_HEIGHT, MAP_WIDTH), -1, dtype=int)
_EAST = _SIDE + 3 * _SPACING + SHELF_WIDTH  # 80: the first column east of shelf D
for _b in range(5):  # RECEIVING (west of shelf A) and SHIPPING (east of shelf D): 10 m x 10 m bays
    for _h in range(2):  # bands 1-5 from the south; W and E halves
        ZONE_MAP[10 * _b:10 * _b + 10, 10 * _h:10 * _h + 10] = 2 * _b + _h
        ZONE_MAP[10 * _b:10 * _b + 10, _EAST + 10 * _h:_EAST + 10 * _h + 10] = 15 + 2 * _b + _h
for _k in range(3):  # AISLE 1-3: between the shelf blocks
    ZONE_MAP[10:40, _SIDE + SHELF_WIDTH + _k * _SPACING:_SIDE + (_k + 1) * _SPACING] = 10 + _k
ZONE_MAP[:10, _SIDE:_EAST] = 13  # CROSS AISLE S: south of the shelves
ZONE_MAP[40:, _SIDE:_EAST] = 14  # CROSS AISLE N: north of the shelves
ZONE_P_CORRECT = 0.9  # probability that the reader names the zone the robot is in
ROBOT_RADIUS = 0.5  # meters (the book's circular robot)
WHEEL_RADIUS = 0.2  # meters
NUM_WHEELS = 3
MOTION_SIGMA = 0.2  # meters per step, the book's motion-model standard deviation

# Planning parameters (Section 4.5, Figure 1).
GAMMA = 0.975
GOAL_REWARD = 100.0
PROXIMITY_PENALTY = 50.0

# Actions move exactly one cell. The numbering matches the book's Q_value code
# (0: j-1, 1: j+1, 2: i-1, 3: i+1); we use compass names to avoid confusion.
WEST, EAST, SOUTH, NORTH = 0, 1, 2, 3
ACTION_NAMES = ("WEST", "EAST", "SOUTH", "NORTH")
ACTION_DELTAS = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])  # (di, dj) per action
ACTION_VECTORS = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])  # (dx, dy) meters


###############################################################################
#                       PROVIDED HELPERS (do not modify)                      #
###############################################################################

def point_to_cell(point) -> tuple[int, int]:
    """Return the grid cell ``(i, j)`` that contains the point ``(x, y)``.

    The cell may lie outside the map; use :func:`in_bounds` to check.
    """
    x, y = np.asarray(point, dtype=float)[:2]
    return int(np.round(y)), int(np.round(x))


def points_to_cells(points) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized :func:`point_to_cell` for an ``(N, 2)`` array of points.

    Returns ``(rows, cols, inside)``: integer row and column arrays of shape
    ``(N,)`` and a boolean mask that is True where the cell lies on the map.
    Rows and columns of outside points are clipped to the map so that they are
    always safe to use as indices; always combine lookups with ``inside``.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    cols = np.round(points[:, 0]).astype(int)
    rows = np.round(points[:, 1]).astype(int)
    inside = (rows >= 0) & (rows < MAP_HEIGHT) & (cols >= 0) & (cols < MAP_WIDTH)
    return np.clip(rows, 0, MAP_HEIGHT - 1), np.clip(cols, 0, MAP_WIDTH - 1), inside


def cell_to_point(cell) -> np.ndarray:
    """Return the ``(x, y)`` center of grid cell ``(i, j)``."""
    i, j = cell
    return np.array([float(j), float(i)])


def in_bounds(cell, shape=(MAP_HEIGHT, MAP_WIDTH)) -> bool:
    """True if cell ``(i, j)`` lies inside a grid of the given shape."""
    i, j = cell
    return 0 <= i < shape[0] and 0 <= j < shape[1]


def next_cell(cell, action: int, shape=(MAP_HEIGHT, MAP_WIDTH)) -> tuple[int, int]:
    """Cell reached by taking ``action`` from ``cell``.

    Exactly like the book's ``Q_value``: moving off the map leaves the robot
    where it is.
    """
    di, dj = ACTION_DELTAS[action]
    candidate = (int(cell[0] + di), int(cell[1] + dj))
    return candidate if in_bounds(candidate, shape) else (int(cell[0]), int(cell[1]))




###############################################################################
#                          SECTION 0: SETUP (1 point)                         #
###############################################################################

# Setup task: student_info -- 1 point
def student_info() -> dict[str, str]:
    """Return your name and GT username in the provided dictionary.

    Replace both TODO strings. Do not rename the ``name`` or ``gt_username``
    keys, because the autograder checks them.
    """
    return {"name": "TODO: your full name", "gt_username": "TODO: your GT username"}


# Not graded by the autograder: make the robot yours!
def robot_profile() -> dict[str, str]:
    """Personalize the robot that stars in the Webots demo.

    ``name`` is shown on the HUD and on the final score screen: at most 16
    plain ASCII characters. ``color`` is the robot's body color and must be one
    of: "gold", "navy", "orange", "cyan", "magenta", "lime", "white", "red".
    The Webots screenshot in your reflection should show a name you chose
    (not the default "RB-3630").
    """
    # Make it yours! For example {"name": "BUZZ-BOT", "color": "navy"}. Colors:
    # gold, navy, orange, cyan, magenta, lime, white, red.
    return {"name": "eva", "color": "white"}


###############################################################################
#               PART 1: STATE ESTIMATION (Book Sections 4.2-4.4)              #
###############################################################################

# TODO 1: sample_motion_model -- 5 points
def sample_motion_model(states, u, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Sample next states from the Gaussian motion model x' = x + u + w.

    Book: Section 4.2.5, equations (4.19)-(4.20). Each next state is the
    current state plus the commanded displacement ``u`` plus Gaussian noise
    ``w ~ N(0, sigma^2 I)``: every coordinate of every state gets its own,
    independent noise sample with standard deviation ``sigma``.

    Args:
        states: array-like of shape ``(2,)`` (one state) or ``(N, 2)`` (one
            state per row), in meters.
        u: array-like of shape ``(2,)``, the displacement ``(dx, dy)`` in
            meters, the same for every state.
        sigma: noise standard deviation in meters, ``sigma >= 0``. With
            ``sigma = 0`` the result is exactly ``states + u``.
        rng: a NumPy random ``Generator`` (as made by
            ``np.random.default_rng(seed)``). Draw all randomness from it,
            e.g. with ``rng.normal``; never use ``np.random.normal`` or
            ``np.random.seed``, so that every mission can be replayed from its
            seed.

    Returns:
        A new float array with the same shape as ``states``. Do not modify
        ``states`` itself.

    Hint: ``rng.normal(0.0, sigma, size=...)`` draws a whole array of noise at
    once.

    In the demo: moves every particle of the particle filter (TODO 4).
    """
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    states = np.asarray(states, dtype=float)
    u = np.asarray(u, dtype=float)
    noise = rng.normal(0.0, sigma, size=states.shape)
    return states + u + noise
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################


# PROVIDED: proximity_likelihood (not a TODO; do not modify)
def proximity_likelihood(points, z_on: bool) -> np.ndarray:
    """PROVIDED. Likelihood of each point given one proximity reading (ON or OFF).

    Book: Section 4.3.1, equation (4.28), and Section 4.4.4.2
    (``likelihood_off``). The proximity sensor is perfect: it reads ON exactly
    when the robot is near a shelf or a wall, as recorded in the provided maps.
    So the likelihood of a point is a lookup at its grid cell
    ``(i, j) = (round(y), round(x))``: ``PROXIMITY_MAP_ON[i, j]`` for ON and
    ``PROXIMITY_MAP_OFF[i, j]`` for OFF, and 0 for a cell off the map.

    Args:
        points: array-like of shape ``(N, 2)`` (or one point ``(2,)``).
        z_on: the reading, True (ON) or False (OFF).

    Returns:
        Float array of shape ``(N,)`` with the likelihoods (zeros and ones).

    Used by your ``mcl_update`` (TODO 5) and in the Webots demo.
    """
    rows, cols, inside = points_to_cells(points)
    lookup = PROXIMITY_MAP_ON if z_on else PROXIMITY_MAP_OFF
    return np.where(inside, lookup[rows, cols], 0.0).astype(float)


# TODO 2: rfid_likelihood -- 6 points
def rfid_likelihood(points, measurement, beacons=BEACONS, max_range: float = RFID_MAX_RANGE,
                    sigma: float = RFID_SIGMA) -> np.ndarray:
    """Likelihood of each point given one RFID range measurement.

    Book: Sections 4.3.2-4.3.3, equations (4.29)-(4.33), and Section 4.4.4.4
    (``likelihood_range``). The RFID reader reports the nearest beacon within
    ``max_range`` meters as ``measurement = (beacon_id, r)``, or
    ``(None, inf)`` when no beacon is within range.

    * ``(beacon_id, r)``: let ``d`` be the distance from the point to
      ``beacons[beacon_id]``. The likelihood is
      ``exp(-(r - d)^2 / (2 sigma^2))`` if ``d <= max_range``, and 0 if
      ``d > max_range`` (a beacon that far away could not have been heard).
      Over the map this is the book's annulus around the beacon.
    * ``(None, inf)``: the likelihood is 1 if the point is strictly farther
      than ``max_range`` from EVERY beacon (distance ``> max_range``), and 0
      otherwise. Over the map this is exactly the book's
      ``out_of_bound_map``.

    There is no map-bounds check here: the formulas above apply to any point.

    Args:
        points: array-like of shape ``(N, 2)``, one ``(x, y)`` per row, in
            meters. A single point of shape ``(2,)`` counts as ``N = 1``.
        measurement: tuple ``(beacon_id, r)``. ``beacon_id`` is either None
            or an integer index into ``beacons`` (possibly a NumPy integer such
            as ``np.int64``, which is what ``np.argmin`` returns); ``r`` is the
            measured range in meters (``inf`` when ``beacon_id`` is None).
        beacons: array of shape ``(B, 2)`` with the beacon positions. The
            default is the 8 warehouse beacons ``BEACONS``, but use whatever
            is passed in.
        max_range: detection range in meters.
        sigma: standard deviation of the range noise in meters.

    Returns:
        Float array of shape ``(N,)`` with values in [0, 1] (unnormalized).

    Hints:
        Test for the special case with ``beacon_id is None``: beacon 0 is a
        valid id, and ``if beacon_id:`` would treat it as missing!
        ``np.linalg.norm(..., axis=-1)`` computes many distances at once, and
        ``points[:, None, :] - beacons[None, :, :]`` has shape ``(N, B, 2)``.

    In the demo: part of every particle's weight (TODO 5); the Robot Vision
    hologram draws this annulus in orange after every RFID ping.
    """
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    beacons = np.asarray(beacons, dtype=float)
    beacon_id, r = measurement
    if beacon_id is None:
        distances = np.linalg.norm(points[:, None, :] - beacons[None, :, :], axis=-1)
        return np.where(np.all(distances > max_range, axis=1), 1.0, 0.0)
    distance = np.linalg.norm(points - beacons[int(beacon_id)], axis=-1)
    likelihood = np.exp(-((r - distance) ** 2) / (2.0 * sigma ** 2))
    return np.where(distance <= max_range, likelihood, 0.0)
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################


# TODO 3: zone_likelihood -- 5 points
def zone_likelihood(points, z_zone, zone_map=ZONE_MAP, p_correct: float = ZONE_P_CORRECT,
                    num_zones: int = NUM_ZONES) -> np.ndarray:
    """Likelihood of each point given one zone reading.

    Book: Section 4.3.1 (a discrete sensor whose likelihood is read from a map,
    like the proximity sensor) and Section 4.4.4.4. The zone reader reports a
    zone id ``z_zone`` (an index into ``ZONE_NAMES``). It names the robot's own
    zone with probability ``p_correct`` and makes a mistake otherwise; the model
    spreads the mistakes evenly over the other ``num_zones - 1`` zones. So for a
    point in the cell ``(i, j)`` with zone ``k = zone_map[i, j]``:

    * ``p_correct`` if ``k == z_zone``;
    * ``(1 - p_correct) / (num_zones - 1)`` if ``k`` is another zone;
    * 0 if the point is off the map or in a shelf cell (``k == -1``): the robot
      cannot be there.

    Args:
        points: array-like of shape ``(N, 2)``, one ``(x, y)`` per row, in
            meters. A single point of shape ``(2,)`` counts as ``N = 1``.
        z_zone: the reading, an integer zone id (possibly a NumPy integer).
        zone_map: ``(50, 100)`` integer array of zone ids, -1 for shelf cells.
            The default is ``ZONE_MAP``, but use whatever is passed in.
        p_correct: probability that the reading names the robot's zone.
        num_zones: the number of zones.

    Returns:
        Float array of shape ``(N,)`` (unnormalized).

    Hint: ``points_to_cells`` gives each point's cell and whether it is on the
    map, and ``np.where`` picks a value per point.

    In the demo: part of every particle's weight (TODO 5); the Robot Vision
    hologram outlines the zone the reader names (green).
    """
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    rows, cols, inside = points_to_cells(points)
    zone = np.asarray(zone_map)[rows, cols]
    mistake = (1.0 - p_correct) / (num_zones - 1)
    likelihood = np.where(zone == z_zone, p_correct, mistake).astype(float)
    return np.where(inside & (zone != -1), likelihood, 0.0)
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################


# TODO 4: mcl_predict -- 6 points
def mcl_predict(particles, weights, u, sigma: float, rng: np.random.Generator,
                num_samples: int | None = None) -> np.ndarray:
    """Prediction step of Monte Carlo localization (the particle filter).

    Book: Section 4.4.4.1, equation (4.44), and ``predict_samples``. Sample
    from the mixture of motion models in two steps:

    1. Resample: draw ``num_samples`` particle indices with replacement,
       choosing particle ``s`` with probability ``weights[s] / sum(weights)``,
       with ``rng.choice(len(particles), size=num_samples, p=...)``.
       Particles with weight 0 are never chosen.
    2. Move: apply your ``sample_motion_model`` (TODO 1) to the chosen
       particles, so that every copy of a particle that was chosen several
       times gets its own noise.

    Args:
        particles: array of shape ``(N, 2)``, one particle ``(x, y)`` per row.
        weights: array-like of shape ``(N,)``, nonnegative with a positive
            sum, NOT necessarily normalized (they are raw likelihoods from
            ``mcl_update`` and can be tiny, like 1e-30).
        u: array-like of shape ``(2,)``, the displacement in meters.
        sigma: motion noise standard deviation in meters.
        rng: NumPy random ``Generator``; draw all randomness (for both steps)
            from it.
        num_samples: number of particles to return. If None, return as many
            as were given (``N``). The demo uses this to shrink the particle
            set once the robot is localized.

    Returns:
        Float array of shape ``(num_samples, 2)``: the new, unweighted
        particles.

    Hint: normalize into a new array, ``p = weights / np.sum(weights)``,
    instead of in place like the book's ``weights /= np.sum(weights)``, which
    would change the caller's array.

    In the demo: the cyan particle cloud that follows the robot around.
    """
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    particles = np.asarray(particles, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if num_samples is None:
        num_samples = len(particles)
    probabilities = weights / np.sum(weights)
    chosen = rng.choice(len(particles), size=num_samples, p=probabilities)
    return sample_motion_model(particles[chosen], u, sigma, rng)
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################


# TODO 5: mcl_update -- 5 points
def mcl_update(particles, z_on: bool, rfid_measurement, zone_measurement=None) -> np.ndarray:
    """Measurement update of Monte Carlo localization: one weight per particle.

    Book: Section 4.4.4.4, equation (4.45) and the final MCL loop. The sensors
    are independent given the state, so a particle's weight is the product of
    the likelihoods of all available measurements:

        weight = proximity_likelihood(particles, z_on)
               * rfid_likelihood(particles, rfid_measurement)
               * zone_likelihood(particles, zone_measurement) (only if not None)

    Use the provided ``proximity_likelihood`` and your TODOs 2-3 with their
    default parameters.

    Args:
        particles: array of shape ``(N, 2)``, one particle per row.
        z_on: proximity reading (True means ON).
        rfid_measurement: ``(beacon_id, r)`` or ``(None, inf)``.
        zone_measurement: a zone id, or None when there is no zone reading
            (then there is no zone factor at all).

    Returns:
        Float array of shape ``(N,)`` with the unnormalized weights. Do not
        normalize them: the engine watches their raw size to notice when the
        robot has been kidnapped.

    Hint: the book's final MCL loop computes the weights with
    ``likelihood_range`` and then ``weights *= ...likelihood_off...``, one
    particle at a time; these functions already handle all particles at once.

    In the demo: the particle weights (particle size in the hologram), the
    LOCKED/LOST logic, and the confidence bar.
    """
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    weights = proximity_likelihood(particles, z_on) * rfid_likelihood(particles, rfid_measurement)
    if zone_measurement is not None:
        weights = weights * zone_likelihood(particles, zone_measurement)
    return weights
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################


# TODO 6: estimate_position -- 6 points
def estimate_position(particles, weights) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean and covariance of a weighted particle set.

    Book: Sections 4.1.3-4.1.4: a set of weighted samples represents a
    density, and its mean and covariance summarize where the robot probably is
    and how sure it is. With the normalized weights
    ``w_s = weights[s] / sum(weights)``:

        mean = sum over s of w_s x_s
        cov  = sum over s of w_s (x_s - mean)(x_s - mean)^T

    This is the "population" covariance: no N - 1 correction.

    Args:
        particles: array of shape ``(N, 2)``, one particle per row.
        weights: array-like of shape ``(N,)``, nonnegative with a positive
            sum, not necessarily normalized. Particles with weight 0 do not
            count.

    Returns:
        ``(mean, cov)``: ``mean`` has shape ``(2,)`` and ``cov`` has shape
        ``(2, 2)``. A single particle gives an all-zero ``cov``.

    Hint: ``np.average(particles, axis=0, weights=weights)`` gives the mean.
    For the covariance, weight each centered particle ``x_s - mean`` and use
    one ``(2, N) @ (N, 2)`` matrix product. (``np.cov`` agrees only with its
    ``aweights`` and ``bias`` options set just right.)

    In the demo: the magenta estimate ring with its 2-sigma ellipse, and the
    "estimate +/- sigma" telemetry on the HUD.
    """
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    particles = np.asarray(particles, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mean = np.average(particles, axis=0, weights=weights)
    centered = particles - mean
    normalized = weights / np.sum(weights)
    covariance = (centered * normalized[:, None]).T @ centered
    return mean, covariance
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################


###############################################################################
#           PART 2: PLANNING WITH VALUE ITERATION (Book Section 4.5)          #
###############################################################################

# TODO 7: q_value -- 6 points
def q_value(V, cell, action: int, goal_mask, obstacle_map=BASE_MAP,
            proximity_map=PROXIMITY_MAP_ON, gamma: float = GAMMA) -> float:
    """Q(x, a; V) for one grid cell and one action, exactly like the book's ``Q_value``.

    Book: Section 4.5.1, equation (4.55) and Figure 1. Actions are
    deterministic. For ``cell = (i, j)``:

    1. If ``(i, j)`` is a goal cell (``goal_mask[i, j]`` is True), return 0:
       the reward is only collected when FIRST arriving at the goal.
    2. If ``(i, j)`` is an obstacle (``obstacle_map[i, j]`` is nonzero),
       return 0 (an invalid state).
    3. Otherwise the next cell is ``(r, c) = next_cell((i, j), action,
       V.shape)`` (moving off the grid leaves the robot where it is), and the
       reward starts at 0 and gets
         +100 (``GOAL_REWARD``) if ``(r, c)`` is a goal cell,
         -50 (``PROXIMITY_PENALTY``) if ``(i, j)`` is in proximity
         (``proximity_map[i, j]`` nonzero), and
         another -50 if ``(r, c)`` is in proximity.
       Return ``reward + gamma * V[r, c]``.

    Args:
        V: array of shape ``(H, W)``, the current value function. Its shape
            defines the grid: the tests also use small grids such as 3 x 4.
        cell: ``(i, j)``, the (row, col) of the current cell.
        action: 0 = WEST, 1 = EAST, 2 = SOUTH, or 3 = NORTH.
        goal_mask: boolean array of shape ``(H, W)``, True at the goal
            cell(s); there may be more than one.
        obstacle_map: array of shape ``(H, W)``, nonzero at obstacles.
        proximity_map: array of shape ``(H, W)``, nonzero where the robot is
            too close to something.
        gamma: the discount factor.

    Returns:
        The Q value as a Python float.

    Hint: port the book's ``Q_value`` (Figure 1). The provided ``next_cell``
    replaces its four ``if a == ...`` lines, and the arguments replace its
    globals (``goal``, ``logistics.base_map``, ``logistics.proximity_map_on``,
    and ``0.975``).

    In the demo: every planning decision (TODO 8, and the provided
    ``greedy_action`` and ``vote_action``) is built from it.
    """
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    i, j = int(cell[0]), int(cell[1])
    if goal_mask[i, j] or obstacle_map[i, j] != 0:
        return 0.0
    r, c = next_cell((i, j), action, V.shape)
    reward = 0.0
    if goal_mask[r, c]:
        reward += GOAL_REWARD
    if proximity_map[i, j] != 0:
        reward -= PROXIMITY_PENALTY
    if proximity_map[r, c] != 0:
        reward -= PROXIMITY_PENALTY
    return float(reward + gamma * V[r, c])
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################


# TODO 8: value_iteration -- 7 points
def value_iteration(goal_mask, obstacle_map=BASE_MAP, proximity_map=PROXIMITY_MAP_ON,
                    gamma: float = GAMMA, tol: float = 1e-5, max_iterations: int = 1000) -> np.ndarray:
    """Run value iteration on the grid until the value function converges.

    Book: Section 4.5.1, equation (4.55) and Figure 2. Start from ``V = 0``
    everywhere and repeat "sweeps". One sweep computes, for every cell,

        V_new[i, j] = max over the 4 actions a of Q((i, j), a; V),

    where Q is exactly your ``q_value`` (TODO 7), so goal and obstacle cells
    always get 0. Sweeps are synchronous: every entry of ``V_new`` is computed
    from the previous ``V``, never from entries already updated in the current
    sweep (like the book's ``V_plus``). After each sweep compute
    ``error = max |V_new - V|`` and set ``V = V_new``; stop as soon as
    ``error <= tol``, or after ``max_iterations`` sweeps, whichever comes
    first.

    Args:
        goal_mask: boolean array of shape ``(H, W)``, True at the goal cell(s).
            Its shape defines the grid: any grid shape must work (the tests
            also use small grids), so do not hard-code the book's 50 x 100.
        obstacle_map, proximity_map, gamma: as in ``q_value``.
        tol: stop when the largest change in one sweep is at most ``tol``.
        max_iterations: the maximum number of sweeps.

    Returns:
        Float array of shape ``(H, W)``: V after the last sweep. For example,
        ``max_iterations=1`` returns the result of exactly one sweep from
        ``V = 0``. With the book's goal cell (35, 50) on the warehouse map, the
        loop stops after 86 sweeps, just like in the book.

    Hints:
        The straightforward version loops over all cells and calls your
        ``q_value``, like the book; it takes a second or two on the full map,
        which is fine. For a lightning-fast version (the demo re-plans live
        when you drop an obstacle), compute a whole grid of Q values for each
        action with NumPy (``np.clip`` of the row/column index grids keeps the
        moves on the map) and take the maximum over the 4 actions.

    In the demo: the value-function layer of the hologram (key M), the yellow
    planned path, and all re-planning around obstacles.
    """
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    goal_mask = np.asarray(goal_mask, dtype=bool)
    obstacle_map = np.asarray(obstacle_map)
    proximity_map = np.asarray(proximity_map)
    height, width = goal_mask.shape
    rows, cols = np.indices((height, width))
    invalid = goal_mask | (obstacle_map != 0)
    in_proximity = proximity_map != 0
    value = np.zeros((height, width), dtype=float)
    for _ in range(max_iterations):
        action_values = []
        for di, dj in ACTION_DELTAS:
            next_rows = np.clip(rows + int(di), 0, height - 1)
            next_cols = np.clip(cols + int(dj), 0, width - 1)
            reward = (
                GOAL_REWARD * goal_mask[next_rows, next_cols]
                - PROXIMITY_PENALTY * in_proximity
                - PROXIMITY_PENALTY * in_proximity[next_rows, next_cols]
            )
            q = reward + gamma * value[next_rows, next_cols]
            action_values.append(np.where(invalid, 0.0, q))
        updated = np.max(np.stack(action_values, axis=0), axis=0)
        error = float(np.max(np.abs(updated - value)))
        value = updated
        if error <= tol:
            break
    return value
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################


# PROVIDED: greedy_action (not a TODO; do not modify)
def greedy_action(V, cell, goal_mask, obstacle_map=BASE_MAP, proximity_map=PROXIMITY_MAP_ON,
                  gamma: float = GAMMA) -> int:
    """PROVIDED. The best action at ``cell``: the one with the largest Q value.

    Book: Section 4.5.1 ("at any cell, we can just look around in all 4
    directions"). Computes your ``q_value`` (TODO 7) for the 4 actions and
    returns the index of the largest; ties go to the lowest action index.

    Args:
        V, cell, goal_mask, obstacle_map, proximity_map, gamma: as in
            ``q_value``.

    Returns:
        The best action as a Python int in {0, 1, 2, 3}.

    Used for the yellow path preview in Webots, and by ``vote_action``.
    """
    # Rounded, so that tiny floating-point differences between implementations never change a tie.
    q = [round(float(q_value(V, cell, a, goal_mask, obstacle_map, proximity_map, gamma)), 6) for a in range(4)]
    return int(np.argmax(q))


# PROVIDED: vote_action (not a TODO; do not modify)
def vote_action(V, particles, weights, goal_mask, obstacle_map=BASE_MAP,
                proximity_map=PROXIMITY_MAP_ON, gamma: float = GAMMA) -> int:
    """PROVIDED. Choose one action for a whole particle belief by weighted voting.

    Book: Section 4.5.3 (closing the loop): while the robot is unsure where it
    is, every particle votes for ``greedy_action`` at its own cell, and each
    vote counts as much as the particle's weight. Particles off the map, in a
    goal or obstacle cell, or with weight 0 do not vote. Returns the action
    with the largest total weight (ties: the lowest index; no valid particle:
    0).

    Args:
        V: array of shape ``(50, 100)``, the value function.
        particles: array of shape ``(N, 2)``, one ``(x, y)`` per row.
        weights: array-like of shape ``(N,)``, nonnegative.
        goal_mask, obstacle_map, proximity_map, gamma: as in ``q_value``.

    Returns:
        The winning action as a Python int in {0, 1, 2, 3}.

    In the Webots demo the robot picks every move this way.
    """
    rows, cols, inside = points_to_cells(particles)
    weights = np.asarray(weights, dtype=float)
    goal = np.asarray(goal_mask, dtype=bool)
    valid = inside & ~goal[rows, cols] & (np.asarray(obstacle_map)[rows, cols] == 0) & (weights > 0)
    votes = np.zeros(4)
    tally = {}
    for i, j, w in zip(rows[valid], cols[valid], weights[valid]):
        tally[(int(i), int(j))] = tally.get((int(i), int(j)), 0.0) + float(w)
    for cell, w in tally.items():
        votes[greedy_action(V, cell, goal, obstacle_map, proximity_map, gamma)] += w
    return int(np.argmax(votes))


###############################################################################
#          PROVIDED FOR THE WEBOTS ROBOT (not part of the assignment)         #
###############################################################################

# PROVIDED: omni_jacobian (not a TODO; do not modify)
def omni_jacobian(num_wheels: int = NUM_WHEELS, wheel_radius: float = WHEEL_RADIUS) -> np.ndarray:
    """PROVIDED for the Webots robot. The Jacobian J that maps the velocity (vx, vy) to wheel speeds.

    Book: Section 4.2.3, equations (4.14)-(4.16), and ``jacobian_matrix``:
    wheel ``i`` sits at the angle ``theta_i = 2 pi i / num_wheels`` and row
    ``i`` of J is ``[-sin(theta_i), cos(theta_i)] / r``, so ``omega = J @ v``.

    Returns:
        Float array of shape ``(num_wheels, 2)``.
    """
    thetas = 2 * np.pi * np.arange(num_wheels) / num_wheels
    return np.column_stack((-np.sin(thetas), np.cos(thetas))) / wheel_radius


# PROVIDED: estimate_motion_model (not a TODO; do not modify)
def estimate_motion_model(states, controls, next_states) -> tuple[np.ndarray, np.ndarray, float]:
    """PROVIDED for the Webots robot. Estimate A, B, and sigma of x' = A x + B u + w from data.

    Book: Section 4.6.6, equations (4.68)-(4.71): least squares for ``[A B]``
    and the unbiased noise estimate from the residuals. At boot the robot gets
    a calibration log of 400 (state, control, next state) samples and uses
    sigma_hat as the motion noise of its particle filter (the boot screen
    shows the result).

    Args:
        states, controls, next_states: array-like of shape ``(N, 2)`` each.

    Returns:
        ``(A_hat, B_hat, sigma_hat)``: two ``(2, 2)`` arrays and a float.
    """
    X = np.asarray(states, dtype=float)
    U = np.asarray(controls, dtype=float)
    Z = np.asarray(next_states, dtype=float)
    XU = np.hstack((X, U))  # (N, 4): one row [x, y, ux, uy] per sample
    theta = np.linalg.solve(XU.T @ XU, XU.T @ Z)  # (4, 2) == [A B]^T
    A_hat, B_hat = theta[:2].T, theta[2:].T
    residuals = Z - XU @ theta
    n_samples = len(X)
    sigma_hat = float(np.sqrt(np.sum(residuals ** 2) / (2 * n_samples - 8)))
    return A_hat, B_hat, sigma_hat
