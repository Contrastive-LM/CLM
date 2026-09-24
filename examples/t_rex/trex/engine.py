"""A deterministic Python clone of the Chrome offline T-Rex runner.

Rules, constants, collision boxes and sprite coordinates follow the original game
(Chromium, BSD-3-Clause). The simulation advances in fixed 60 FPS steps, and the
obstacle course has its own seeded random stream, so two games started with the same
seed meet the same obstacles for as long as both survive.
"""

import math
import random
from dataclasses import dataclass

FPS = 60
FRAME_MS = 1000 / FPS
WIDTH = 600
HEIGHT = 150

# Runner (normal speed mode).
SPEED = 6
ACCELERATION = 0.001
MAX_SPEED = 13
GAP_COEFFICIENT = 0.6
MAX_GAP_COEFFICIENT = 1.5
CLEAR_TIME = 3000
BOTTOM_PAD = 10
MAX_OBSTACLE_LENGTH = 3
MAX_OBSTACLE_DUPLICATION = 2
INVERT_DISTANCE = 700
INVERT_FADE_DURATION = 12000
MAX_BLINK_COUNT = 3
GAMEOVER_CLEAR_TIME = 1200
INTRO_MS = 400  # The canvas reveal animation that ends the intro.

# T-rex.
TREX_WIDTH = 44
TREX_HEIGHT = 47
TREX_WIDTH_DUCK = 59
START_X = 50
INTRO_DURATION = 1500
GRAVITY = 0.6
INITIAL_JUMP_VELOCITY = -10
DROP_VELOCITY = -5
MIN_JUMP_HEIGHT = 30
MAX_JUMP_HEIGHT = 30
SPEED_DROP_COEFFICIENT = 3
GROUND_Y = HEIGHT - TREX_HEIGHT - BOTTOM_PAD
MIN_JUMP_Y = GROUND_Y - MIN_JUMP_HEIGHT
BLINK_TIMING = 7000

# Distance meter.
DISTANCE_COEFFICIENT = 0.025
ACHIEVEMENT_DISTANCE = 100
FLASH_DURATION = 1000 / 4
FLASH_ITERATIONS = 3
MAX_DISTANCE_UNITS = 5

# Scenery.
CLOUD_WIDTH = 46
CLOUD_FREQUENCY = 0.5
MAX_CLOUDS = 6
BG_CLOUD_SPEED = 0.2
MIN_CLOUD_GAP, MAX_CLOUD_GAP = 100, 400
MAX_SKY_LEVEL, MIN_SKY_LEVEL = 30, 71
HORIZON_Y = 127
MOON_PHASES = (140, 120, 100, 60, 40, 20, 0)
NIGHT_FADE_SPEED = 0.035
MOON_SPEED, STAR_SPEED, STAR_MAX_Y, NUM_STARS = 0.25, 0.3, 70, 2
RESTART_LOGO_PAUSE = 875
RESTART_FRAMES = 8

WAITING, RUNNING, JUMPING, DUCKING, CRASHED = "waiting", "running", "jumping", "ducking", "crashed"
# Sprite x offsets and animation rate per status.
ANIMATION = {
    WAITING: ((44, 0), 1000 / 3),
    RUNNING: ((88, 132), 1000 / 12),
    CRASHED: ((220,), 1000 / 60),
    JUMPING: ((0,), 1000 / 60),
    DUCKING: ((264, 323), 1000 / 8),
}


def js_round(value):
    """JavaScript Math.round: halves round toward positive infinity."""
    return math.floor(value + 0.5)


def random_int(rng, low, high):
    return math.floor(rng.random() * (high - low + 1)) + low


@dataclass
class Box:
    x: float
    y: float
    width: float
    height: float


def overlap(a, b):
    return (
        a.x < b.x + b.width
        and a.x + a.width > b.x
        and a.y < b.y + b.height
        and a.height + a.y > b.y
    )


TREX_BOXES = {
    False: (
        Box(22, 0, 17, 16),
        Box(1, 18, 30, 9),
        Box(10, 35, 14, 8),
        Box(1, 24, 29, 5),
        Box(5, 30, 21, 4),
        Box(9, 34, 15, 4),
    ),
    True: (Box(1, 18, 55, 25),),
}


@dataclass(frozen=True)
class ObstacleType:
    name: str
    width: int
    height: int
    y: tuple
    multiple_speed: float
    min_gap: int
    min_speed: float
    boxes: tuple
    sprite_x: int
    frames: int = 1
    frame_ms: float = 0
    speed_offset: float = 0


OBSTACLE_TYPES = (
    ObstacleType(
        "cactusSmall",
        17,
        35,
        (105,),
        4,
        120,
        0,
        (Box(0, 7, 5, 27), Box(4, 0, 6, 34), Box(10, 4, 7, 14)),
        228,
    ),
    ObstacleType(
        "cactusLarge",
        25,
        50,
        (90,),
        7,
        120,
        0,
        (Box(0, 12, 7, 38), Box(8, 0, 7, 49), Box(13, 10, 10, 38)),
        332,
    ),
    ObstacleType(
        "pterodactyl",
        46,
        40,
        (100, 75, 50),
        999,
        150,
        8.5,
        (
            Box(15, 15, 16, 5),
            Box(18, 21, 24, 6),
            Box(2, 14, 4, 3),
            Box(6, 10, 4, 7),
            Box(10, 8, 6, 9),
        ),
        134,
        frames=2,
        frame_ms=1000 / 6,
        speed_offset=0.8,
    ),
)


def obstacle_step(speed, offset):
    """Pixels an obstacle moves in one 60 FPS frame, exactly as the original computes it."""
    return math.floor(((speed + offset) * FPS / 1000) * FRAME_MS)


KINDS = {kind.name: kind for kind in OBSTACLE_TYPES}
BIRD_HEIGHTS = {100: "low bird", 75: "bird at head height", 50: "high bird"}


@dataclass(frozen=True)
class Spec:
    """One obstacle of a course: what it is and the gap that follows it.

    `gap` is a position in [0, 1] within the original game's random gap range, which is
    resolved against the actual speed at spawn time.
    """

    kind: str
    size: int = 1
    y: int = 0
    gap: float = 0.5
    faster: bool = False
    source: str = "random"

    @property
    def label(self):
        if self.kind == "pterodactyl":
            return BIRD_HEIGHTS[self.y]
        noun = "small cact" if self.kind == "cactusSmall" else "large cact"
        return f"{noun}us" if self.size == 1 else f"{self.size} {noun}i"


class RandomCourse:
    """The original game's obstacle rules, drawn from one random stream per obstacle.

    A stream per (seed, run, obstacle) keeps every obstacle reproducible no matter which
    process asks for it or when.
    """

    def __init__(self, seed):
        self.seed = seed

    def spec(self, game, index, source="random"):
        rng = random.Random(f"course-{self.seed}-{game.run_index}-{index}")
        while True:
            kind = OBSTACLE_TYPES[random_int(rng, 0, len(OBSTACLE_TYPES) - 1)]
            if not game.too_many(kind.name) and game.speed >= kind.min_speed:
                break
        size = random_int(rng, 1, MAX_OBSTACLE_LENGTH)
        y = kind.y[random_int(rng, 0, len(kind.y) - 1)] if len(kind.y) > 1 else kind.y[0]
        return Spec(kind.name, size, y, rng.random(), rng.random() > 0.5, source)


class Obstacle:
    serial = 0

    def __init__(self, spec, speed):
        Obstacle.serial += 1
        self.id = Obstacle.serial
        self.spec = spec
        kind = self.kind = KINDS[spec.kind]
        self.size = spec.size
        self.x = WIDTH + kind.width
        self.remove = False
        self.following_created = False
        self.frame = 0
        self.timer = 0.0
        self.boxes = [Box(b.x, b.y, b.width, b.height) for b in kind.boxes]
        # Groups only appear above the type's speed threshold, as in the original.
        if self.size > 1 and kind.multiple_speed > speed:
            self.size = 1
        self.width = kind.width * self.size
        self.y = spec.y if spec.y in kind.y else kind.y[0]
        # The central box stretches to cover grouped cacti.
        if self.size > 1:
            self.boxes[1].width = self.width - self.boxes[0].width - self.boxes[2].width
            self.boxes[2].x = self.width - self.boxes[2].width
        self.speed_offset = 0.0
        if kind.speed_offset:
            self.speed_offset = kind.speed_offset if spec.faster else -kind.speed_offset
        min_gap = js_round(self.width * speed + kind.min_gap * GAP_COEFFICIENT)
        span = js_round(min_gap * MAX_GAP_COEFFICIENT) - min_gap
        self.gap = min_gap + min(math.floor(spec.gap * (span + 1)), span)

    def update(self, dt, speed):
        if self.remove:
            return
        self.x -= math.floor(((speed + self.speed_offset) * FPS / 1000) * dt)
        if self.kind.frames > 1:
            self.timer += dt
            if self.timer >= self.kind.frame_ms:
                self.frame = 0 if self.frame == self.kind.frames - 1 else self.frame + 1
                self.timer = 0
        if self.x + self.width <= 0:
            self.remove = True

    @property
    def label(self):
        if self.kind.name == "pterodactyl":
            return BIRD_HEIGHTS[self.y]
        noun = "small cact" if self.kind.name == "cactusSmall" else "large cact"
        return f"{noun}us" if self.size == 1 else f"{self.size} {noun}i"


def jump_step(y, velocity, speed_drop, ms_per_frame, dt=FRAME_MS):
    """One frame of jump physics. Returns (y, velocity, reached_min_height, ended_jump)."""
    frames = dt / ms_per_frame
    if speed_drop:
        y += js_round(velocity * SPEED_DROP_COEFFICIENT * frames)
    else:
        y += js_round(velocity * frames)
    velocity += GRAVITY * frames
    return y, velocity, y < MIN_JUMP_Y or speed_drop, y < MAX_JUMP_HEIGHT or speed_drop


class Trex:
    def __init__(self, rng):
        self.rng = rng
        self.x = 0
        self.x_initial = 0
        self.y = GROUND_Y
        self.velocity = 0.0
        self.jumping = False
        self.ducking = False
        self.speed_drop = False
        self.reached_min_height = False
        self.jump_count = 0
        self.playing_intro = False
        self.blink_count = 0
        self.timer = 0.0
        self.clock = 0.0
        self.sprite = 0
        self.set_status(WAITING)

    def set_status(self, status):
        self.status = status
        self.frame = 0
        self.frames, self.ms_per_frame = ANIMATION[status]
        if status == WAITING:
            self.anim_start = self.clock
            self.blink_delay = math.ceil(self.rng.random() * BLINK_TIMING)

    def update(self, dt, status=None):
        self.clock += dt
        self.timer += dt
        if status is not None:
            self.set_status(status)
        if self.playing_intro and self.x < START_X:
            self.x += js_round(START_X / INTRO_DURATION * dt)
            self.x_initial = self.x
        if self.status == WAITING:
            if self.clock - self.anim_start >= self.blink_delay:
                self.sprite = self.frames[self.frame]
                if self.frame == 1:
                    self.anim_start = self.clock
                    self.blink_delay = math.ceil(self.rng.random() * BLINK_TIMING)
                    self.blink_count += 1
        else:
            self.sprite = self.frames[self.frame]
        if self.timer >= self.ms_per_frame:
            self.frame = 0 if self.frame == len(self.frames) - 1 else self.frame + 1
            self.timer = 0
        # Speed drop becomes a duck if the down key is still held at ground level.
        if self.speed_drop and self.y == GROUND_Y:
            self.speed_drop = False
            self.set_duck(True)

    def start_jump(self, speed):
        if not self.jumping:
            self.update(0, JUMPING)
            self.velocity = INITIAL_JUMP_VELOCITY - speed / 10
            self.jumping = True
            self.reached_min_height = False
            self.speed_drop = False

    def end_jump(self):
        if self.reached_min_height and self.velocity < DROP_VELOCITY:
            self.velocity = DROP_VELOCITY

    def update_jump(self, dt):
        self.y, self.velocity, reached, ended = jump_step(
            self.y, self.velocity, self.speed_drop, ANIMATION[self.status][1], dt
        )
        self.reached_min_height = self.reached_min_height or reached
        if ended:
            self.end_jump()
        if self.y > GROUND_Y:
            self.reset()
            self.jump_count += 1

    def set_speed_drop(self):
        self.speed_drop = True
        self.velocity = 1

    def set_duck(self, ducking):
        if ducking and self.status != DUCKING:
            self.update(0, DUCKING)
            self.ducking = True
        elif self.status == DUCKING:
            self.update(0, RUNNING)
            self.ducking = False

    def reset(self):
        self.x = self.x_initial
        self.y = GROUND_Y
        self.velocity = 0.0
        self.jumping = False
        self.ducking = False
        self.update(0, RUNNING)
        self.speed_drop = False
        self.jump_count = 0

    def collision_boxes(self):
        return TREX_BOXES[self.ducking]


def check_collision(obstacle, trex_x, trex_y, ducking):
    """The original two-stage test: outer bounds, then the detailed boxes."""
    outer = Box(trex_x + 1, trex_y + 1, TREX_WIDTH - 2, TREX_HEIGHT - 2)
    other = Box(obstacle.x + 1, obstacle.y + 1, obstacle.width - 2, obstacle.kind.height - 2)
    if not overlap(outer, other):
        return False
    for mine in TREX_BOXES[ducking]:
        a = Box(mine.x + outer.x, mine.y + outer.y, mine.width, mine.height)
        for theirs in obstacle.boxes:
            if overlap(a, Box(theirs.x + other.x, theirs.y + other.y, theirs.width, theirs.height)):
                return True
    return False


class Cloud:
    def __init__(self, rng):
        self.x = WIDTH
        self.gap = random_int(rng, MIN_CLOUD_GAP, MAX_CLOUD_GAP)
        self.y = random_int(rng, MAX_SKY_LEVEL, MIN_SKY_LEVEL)
        self.remove = False

    def update(self, speed):
        if not self.remove:
            self.x -= math.ceil(speed)
            if self.x + CLOUD_WIDTH <= 0:
                self.remove = True


class NightMode:
    def __init__(self, rng):
        self.rng = rng
        self.x = 0.0
        self.phase = 0
        self.opacity = 0.0
        self.draw_stars = False
        self.place_stars()

    def place_stars(self):
        segment = js_round(WIDTH / NUM_STARS)
        self.stars = [
            [
                random_int(self.rng, segment * i, segment * (i + 1)),
                random_int(self.rng, 0, STAR_MAX_Y),
            ]
            for i in range(NUM_STARS)
        ]

    @staticmethod
    def move(position, speed):
        return WIDTH if position < -20 else position - speed

    def update(self, activated):
        if activated and self.opacity == 0:
            self.phase = (self.phase + 1) % len(MOON_PHASES)
        if activated and (self.opacity < 1 or self.opacity == 0):
            self.opacity += NIGHT_FADE_SPEED
        elif self.opacity > 0:
            self.opacity -= NIGHT_FADE_SPEED
        if self.opacity > 0:
            self.x = self.move(self.x, MOON_SPEED)
            if self.draw_stars:
                for star in self.stars:
                    star[0] = self.move(star[0], STAR_SPEED)
        else:
            self.opacity = 0
            self.place_stars()
        self.draw_stars = True

    def reset(self):
        self.phase = 0
        self.opacity = 0
        self.update(False)


class Game:
    """One runner. Input methods mirror the original key handlers."""

    def __init__(self, seed=0, course=None):
        self.seed = seed
        self.scenery = random.Random(f"scenery-{seed}")
        self.run_index = 0
        self.course = course or RandomCourse(seed)
        self.fallback = RandomCourse(seed)
        self.obstacle_index = 0
        self.invincible = False  # For course look-ahead simulations only.
        self.trex = Trex(self.scenery)
        self.night = NightMode(self.scenery)
        self.clouds = [Cloud(self.scenery)]
        self.horizon_x = [0, WIDTH]
        self.horizon_source = [0, WIDTH]
        self.obstacles = []
        self.history = []
        self.speed = SPEED
        self.distance = 0.0
        self.running_time = 0.0
        self.high_score = 0
        self.max_score = int("9" * MAX_DISTANCE_UNITS)
        self.score_units = MAX_DISTANCE_UNITS
        self.achievement = False
        self.flash_timer = 0.0
        self.flash_iterations = 0
        self.score_visible = True
        self.shown_score = 0
        self.activated = False
        self.playing = False
        self.playing_intro = False
        self.intro_ms = 0.0
        self.crashed = False
        self.paused = False
        self.inverted = False
        self.invert_timer = 0.0
        self.invert_trigger = False
        self.invert_fade = 0.0  # 0 = day colours, 1 = inverted; eased like the CSS transition.
        self.crash_ms = 0.0
        self.restart_frame = 0
        self.restart_timer = 0.0
        self.clock = 0.0
        self.events = []
        self.deaths = 0

    # -- Input, as the original keydown/keyup handlers ---------------------------------
    def press_jump(self):
        if self.crashed:
            return
        if not self.playing:
            self.playing = True
            self.step()
        if not self.trex.jumping and not self.trex.ducking:
            self.events.append("press")
            self.trex.start_jump(self.speed)

    def press_duck(self):
        if not self.playing or self.crashed:
            return
        if self.trex.jumping:
            self.trex.set_speed_drop()
        elif not self.trex.ducking:
            self.trex.set_duck(True)

    def release_duck(self):
        self.trex.speed_drop = False
        self.trex.set_duck(False)

    def restart(self):
        if not self.crashed:
            return
        self.run_index += 1
        self.obstacle_index = 0
        self.history = []
        self.running_time = 0.0
        self.playing = True
        self.paused = False
        self.crashed = False
        self.distance = 0.0
        self.speed = SPEED
        self.achievement = False
        self.flash_iterations = 0
        self.flash_timer = 0.0
        self.shown_score = 0
        self.obstacles = []
        self.horizon_x = [0, WIDTH]
        self.night.reset()
        self.trex.reset()
        self.events.append("press")
        self.invert(reset=True)
        self.restart_frame = 0
        self.restart_timer = 0.0
        self.step()

    # -- Queries -----------------------------------------------------------------------
    @property
    def score(self):
        return self.actual_distance(math.ceil(self.distance))

    @staticmethod
    def actual_distance(distance):
        return js_round(distance * DISTANCE_COEFFICIENT) if distance else 0

    @property
    def has_obstacles(self):
        return self.running_time > CLEAR_TIME

    @property
    def reveal(self):
        """Visible canvas width: the original shows only the dino until the intro."""
        if not self.activated:
            return TREX_WIDTH
        if self.playing_intro:
            t = min(1.0, self.intro_ms / INTRO_MS)
            return TREX_WIDTH + (WIDTH - TREX_WIDTH) * (1 - (1 - t) ** 3)
        return WIDTH

    # -- Simulation --------------------------------------------------------------------
    def step(self):
        """Advance one 60 FPS frame, in the order of the original Runner.update."""
        dt = FRAME_MS
        self.clock += dt
        self.invert_fade += (1 if self.inverted else -1) * dt / 1500
        self.invert_fade = min(1.0, max(0.0, self.invert_fade))
        if self.crashed:
            self.crash_ms += dt
            self.restart_timer += dt
            if self.restart_frame == 0 and self.restart_timer > RESTART_LOGO_PAUSE:
                self.restart_timer = 0
                self.restart_frame = 1
            elif 0 < self.restart_frame < RESTART_FRAMES - 1:
                # The original never resets its timer here, so the morph plays one frame per tick.
                self.restart_frame += 1
            return
        if self.playing:
            trex = self.trex
            if trex.jumping:
                trex.update_jump(dt)
            self.running_time += dt
            if trex.jump_count == 1 and not self.playing_intro and not self.activated:
                self.playing_intro = trex.playing_intro = True
                self.activated = True
                self.intro_ms = 0.0
            if self.playing_intro:
                self.update_horizon(0, False)
                self.intro_ms += dt
                if self.intro_ms >= INTRO_MS:
                    self.running_time = 0.0
                    self.playing_intro = trex.playing_intro = False
            else:
                if not self.activated:
                    dt = 0
                self.update_horizon(dt, self.inverted)
            first = self.obstacles[0] if self.obstacles else None
            if (
                self.has_obstacles
                and first
                and not self.invincible
                and check_collision(first, trex.x, trex.y, trex.ducking)
            ):
                self.game_over()
            else:
                self.distance += self.speed * dt / FRAME_MS
                if self.speed < MAX_SPEED:
                    self.speed += ACCELERATION
            self.update_score(dt)
            self.update_night(dt)
        if self.playing or (not self.activated and self.trex.blink_count < MAX_BLINK_COUNT):
            self.trex.update(dt)

    def update_horizon(self, dt, night):
        increment = math.floor(self.speed * (FPS / 1000) * dt)
        line = 0 if self.horizon_x[0] <= 0 else 1
        other = 1 - line
        self.horizon_x[line] -= increment
        self.horizon_x[other] = self.horizon_x[line] + WIDTH
        if self.horizon_x[line] <= -WIDTH:
            self.horizon_x[line] += WIDTH * 2
            self.horizon_x[other] = self.horizon_x[line] - WIDTH
            self.horizon_source[line] = WIDTH if self.scenery.random() > 0.5 else 0
        self.night.update(night)
        cloud_speed = BG_CLOUD_SPEED / 1000 * dt * self.speed
        if not self.clouds:
            self.clouds.append(Cloud(self.scenery))
        else:
            for cloud in reversed(self.clouds):
                cloud.update(cloud_speed)
            last = self.clouds[-1]
            if (
                len(self.clouds) < MAX_CLOUDS
                and WIDTH - last.x > last.gap
                and CLOUD_FREQUENCY > self.scenery.random()
            ):
                self.clouds.append(Cloud(self.scenery))
            self.clouds = [c for c in self.clouds if not c.remove]
        if self.has_obstacles:
            self.update_obstacles(dt)

    def update_obstacles(self, dt):
        kept = list(self.obstacles)
        for obstacle in self.obstacles:
            obstacle.update(dt, self.speed)
            if obstacle.remove:
                kept.pop(0)
        self.obstacles = kept
        if self.obstacles:
            last = self.obstacles[-1]
            if (
                not last.following_created
                and last.x + last.width > 0
                and last.x + last.width + last.gap < WIDTH
            ):
                self.add_obstacle()
                last.following_created = True
        else:
            self.add_obstacle()

    def too_many(self, name):
        """The original's limit on consecutive obstacles of the same type."""
        duplicates = 0
        for seen in self.history:
            duplicates = duplicates + 1 if seen == name else 0
        return duplicates >= MAX_OBSTACLE_DUPLICATION

    def add_obstacle(self):
        spec = self.course.spec(self, self.obstacle_index)
        kind = KINDS[spec.kind]
        if self.too_many(kind.name) or self.speed < kind.min_speed:
            # A designed obstacle that breaks the game's rules at the actual speed is
            # replaced deterministically, so both sides still see the same course.
            spec = self.fallback.spec(self, self.obstacle_index, source="rule")
        self.obstacles.append(Obstacle(spec, self.speed))
        self.obstacle_index += 1
        self.history.insert(0, spec.kind)
        del self.history[MAX_OBSTACLE_DUPLICATION:]

    def update_score(self, dt):
        self.score_visible = True
        if not self.achievement:
            distance = self.score
            if distance > self.max_score and self.score_units == MAX_DISTANCE_UNITS:
                self.score_units += 1
                self.max_score = int(str(self.max_score) + "9")
            if distance > 0 and distance % ACHIEVEMENT_DISTANCE == 0:
                self.achievement = True
                self.flash_timer = 0
                self.events.append("score")
            # Digits freeze while the milestone flashes.
            self.shown_score = distance
        elif self.flash_iterations <= FLASH_ITERATIONS:
            self.flash_timer += dt
            if self.flash_timer < FLASH_DURATION:
                self.score_visible = False
            elif self.flash_timer > FLASH_DURATION * 2:
                self.flash_timer = 0
                self.flash_iterations += 1
        else:
            self.achievement = False
            self.flash_iterations = 0
            self.flash_timer = 0

    def update_night(self, dt):
        if self.invert_timer > INVERT_FADE_DURATION:
            self.invert_timer = 0
            self.invert_trigger = False
            self.invert(reset=False)
        elif self.invert_timer:
            self.invert_timer += dt
        else:
            actual = self.score
            if actual > 0:
                self.invert_trigger = not actual % INVERT_DISTANCE
                if self.invert_trigger and self.invert_timer == 0:
                    self.invert_timer += dt
                    self.invert(reset=False)

    def invert(self, reset):
        if reset:
            self.inverted = False
            self.invert_timer = 0
        else:
            self.inverted = self.invert_trigger

    def game_over(self):
        self.events.append("hit")
        self.playing = False
        self.paused = True
        self.crashed = True
        self.deaths += 1
        self.achievement = False
        self.crash_ms = 0.0
        self.restart_frame = 0
        self.restart_timer = 0.0
        if self.trex.ducking:
            self.trex.x += 1  # The original's standing-up adjustment when crashing low.
        self.trex.update(100, CRASHED)
        if self.distance > self.high_score:
            self.high_score = math.ceil(self.distance)

    def drain_events(self):
        events, self.events = self.events, []
        return events
