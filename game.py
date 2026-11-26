
from faust import engine

# Initialize game
game = engine.Game(
    title="test",
    window_size=(1280, 720)
)

# Load animated FBX character
player = game.load_model("assets/models/player.fbx")

player.position = (0, 0, 0)
player.rotation = (0, 0, 0)
player.scale = (1, 1, 1)

# Animation controller
player.animation.play("Idle")

# Third-person camera
camera = engine.ThirdPersonCamera(
    target=player,
    distance=5.0,
    height=2.0,
    sensitivity=0.15
)

game.set_camera(camera)

# Player movement
@game.update
def update(dt):
    movement = engine.Vector3(0, 0, 0)

    if game.input.key_down("W"):
        movement.z -= 1

    if game.input.key_down("S"):
        movement.z += 1

    if game.input.key_down("A"):
        movement.x -= 1

    if game.input.key_down("D"):
        movement.x += 1

    moving = movement.length() > 0

    if moving:
        movement = movement.normalized()

        # Camera-relative movement
        direction = camera.transform_direction(movement)

        player.move(
            direction * 5.0 * dt
        )

        # Rotate character towards movement direction
        player.look_at_direction(
            direction,
            smoothing=10.0,
            delta_time=dt
        )

        player.animation.blend(
            "Walk",
            weight=1.0,
            duration=0.15
        )

    else:
        player.animation.blend(
            "Idle",
            weight=1.0,
            duration=0.2
        )

    # Update camera
    camera.follow(dt)


game.run()
