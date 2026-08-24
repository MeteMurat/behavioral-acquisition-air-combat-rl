import numpy as np


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(3)
    magnitude = float(np.linalg.norm(value))
    if magnitude < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return value / magnitude


def _command_toward_direction(missile_entity, desired_direction: np.ndarray) -> None:
    desired = _unit(desired_direction)
    forward = _unit(missile_entity.orientation @ np.array([1.0, 0.0, 0.0]))
    up = _unit(missile_entity.orientation @ np.array([0.0, 1.0, 0.0]))
    right = _unit(missile_entity.orientation @ np.array([0.0, 0.0, 1.0]))
    forward_projection = max(1e-9, float(np.dot(desired, forward)))
    pitch_error_deg = float(np.degrees(np.arctan2(np.dot(desired, up), forward_projection)))
    yaw_error_deg = float(np.degrees(np.arctan2(np.dot(desired, right), forward_projection)))
    command_angle_deg = max(1.0, float(getattr(missile_entity, "guidance_command_angle_deg", 20.0)))
    missile_entity.control_inputs["pitch"] = float(np.clip(-pitch_error_deg / command_angle_deg, -1.0, 1.0))
    missile_entity.control_inputs["yaw"] = float(np.clip(-yaw_error_deg / command_angle_deg, -1.0, 1.0))


def missile_direct_attack_DEBUG(missile_entity):
    _command_toward_direction(
        missile_entity,
        missile_entity.target_entity.position - missile_entity.position,
    )


def missile_predictive_attack(missile_entity):
    """Predictive pursuit that produces bounded control commands, not attitude overwrites."""
    jet_pos = np.asarray(missile_entity.target_entity.position, dtype=float)
    jet_velocity = np.asarray(missile_entity.target_entity.velocity, dtype=float)
    missile_pos = np.asarray(missile_entity.position, dtype=float)
    missile_speed = max(float(np.linalg.norm(missile_entity.velocity)), 100.0)
    diff = jet_pos - missile_pos
    predicted_jet_pos = jet_pos.copy()
    for _ in range(20):
        distance = float(np.linalg.norm(diff))
        if distance < 0.1:
            break
        eta = distance / missile_speed
        predicted_jet_pos = jet_pos + eta * jet_velocity
        new_diff = predicted_jet_pos - missile_pos
        if abs(float(np.linalg.norm(new_diff)) - distance) < 0.1:
            diff = new_diff
            break
        diff = new_diff
    _command_toward_direction(missile_entity, diff)
