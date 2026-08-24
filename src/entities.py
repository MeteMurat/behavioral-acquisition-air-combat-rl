import numpy as np
from simulation import SIMULATION_TICKRATE
import physics
import reinforcementlearning as dl
ENTITYID_COUNTER = 0

# A bit of object oriented programming to manage entities in a smart way
# Base entity class that can enact common behaviors for all simulated entities
class entity():
    def __init__(self, starting_position: np.ndarray, starting_velocity: np.ndarray, starting_orientation: np.ndarray, mass: float, reference_area: float, min_drag_coefficient: float, max_drag_coefficient: float, max_lift_coefficient: float, moment_of_inertia_roll: float, moment_of_inertia_pitch: float, moment_of_inertia_yaw: float, optimal_lift_aoa: float, viz_shape: dict):
        global ENTITYID_COUNTER
        self.position = starting_position
        self.velocity = starting_velocity
        self.orientation = starting_orientation
        self.omega = np.array([0.0, 0.0, 0.0])
        self.mass = float(mass)
        self.reference_area = float(reference_area)
        self.min_drag_coefficient = float(min_drag_coefficient)
        self.max_drag_coefficient = float(max_drag_coefficient)
        self.max_lift_coefficient = float(max_lift_coefficient)
        self.moment_of_inertia = np.array([float(moment_of_inertia_roll), float(moment_of_inertia_pitch), float(moment_of_inertia_yaw)], dtype=float)
        self.optimal_lift_aoa = float(optimal_lift_aoa)
        self.shape = viz_shape.get("compound_shape")
        self.viz_shape = viz_shape  # Store the full viz_shape dict for visualization system
        self.viz_id = ENTITYID_COUNTER
        ENTITYID_COUNTER += 1
        self.alive = True



class jet(entity):
    def __init__(self, starting_position: np.ndarray, starting_velocity: np.ndarray, starting_orientation: np.ndarray, mass: float, wingspan: float, length: float, thrust_force: float, reference_area: float, min_drag_coefficient: float, max_drag_coefficient: float, max_lift_coefficient: float, moment_of_inertia_roll: float, moment_of_inertia_pitch: float, moment_of_inertia_yaw: float, optimal_lift_aoa: float, viz_shape: dict):
        self.control_inputs = {'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0}
        self.throttle = 0.0
        super().__init__(starting_position, starting_velocity, starting_orientation, mass, reference_area, min_drag_coefficient, max_drag_coefficient, max_lift_coefficient, moment_of_inertia_roll, moment_of_inertia_pitch, moment_of_inertia_yaw, optimal_lift_aoa, viz_shape)
        self.wingspan = float(wingspan)
        self.length = float(length)
        self.thrust_force = float(thrust_force)
        self.current_reward = 0.0
        self.max_g = 12.0
        self.q_soft_limit_pa = 80000.0
        self.q_hard_limit_pa = 120000.0

    def think(self, entities):
        pitch, roll, yaw, thr = dl.jet_ai_step(entities, self)
        self.control_inputs["pitch"] = float(np.clip(pitch, -1.0, 1.0))
        self.control_inputs["roll"] = float(np.clip(roll, -1.0, 1.0))
        self.control_inputs["yaw"] = float(np.clip(yaw, -1.0, 1.0))
        self.throttle = float(np.clip(thr, 0.0, 1.0))

    def tick(self):
        air_density = physics.get_air_density(self.position[1])
        mach = physics.get_mach_number(self.velocity, self.position[1])
        aoa = physics.get_angle_of_attack(self.velocity, self.orientation)
        sideslip = physics.get_sideslip(self.velocity, self.orientation)
        cd = physics.get_drag_coefficient(aoa, self.min_drag_coefficient, self.max_drag_coefficient, mach)
        cl = physics.get_lift_coefficient(aoa, self.max_lift_coefficient, self.optimal_lift_aoa, -2.0, mach)
        side_cl = physics.get_lift_coefficient(sideslip, self.max_lift_coefficient, self.optimal_lift_aoa, 0.0, mach)
        side_surface_area = self.reference_area * 0.1
        control_surface_area = self.reference_area * 0.8
        q = physics.get_dynamic_pressure(self.velocity, air_density)
        protection = physics.get_dynamic_pressure_control_scale(q, self.q_soft_limit_pa, self.q_hard_limit_pa, 0.10)
        pitch = self.control_inputs["pitch"] * protection
        roll = self.control_inputs["roll"] * protection
        yaw = self.control_inputs["yaw"] * protection
        throttle = self.throttle * protection
        gravity = physics.get_gravity_force(self.mass)
        forces = [
            [gravity, np.zeros(3)],
            [physics.get_thrust_force(self.orientation, throttle, self.thrust_force, self.length), np.array([-0.5 * self.length, 0.0, 0.0])],
            [physics.get_drag_force(self.velocity, air_density, self.reference_area, cd), np.array([-0.2 * self.length, 0.0, 0.0])],
            [physics.get_lift_force(self.velocity, self.reference_area, cl, self.orientation, air_density), np.array([-0.2 * self.length, 0.0, 0.0])],
            [physics.get_sideforce_force(self.velocity, side_surface_area, side_cl, self.orientation, air_density), np.array([-0.45 * self.length, 0.0, 0.0])],
            [physics.get_elevator_force(self.velocity, air_density, control_surface_area, self.max_lift_coefficient, self.orientation, pitch, self.optimal_lift_aoa, 0.30, 1.40), np.array([-0.45 * self.length, 0.0, 0.0])],
            [physics.get_aileron_force(self.velocity, air_density, control_surface_area, self.max_lift_coefficient, self.orientation, roll, self.optimal_lift_aoa, 0.04, 0.6), np.array([-0.05 * self.length, 0.0, 0.50 * self.wingspan])],
            [-physics.get_aileron_force(self.velocity, air_density, control_surface_area, self.max_lift_coefficient, self.orientation, roll, self.optimal_lift_aoa, 0.04, 0.6), np.array([-0.05 * self.length, 0.0, -0.50 * self.wingspan])],
            [physics.get_rudder_force(self.velocity, air_density, control_surface_area, self.max_lift_coefficient, self.orientation, yaw, self.optimal_lift_aoa, 0.03, 0.30), np.array([-0.45 * self.length, 0.0, 0.0])],
        ]
        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        for force, point in forces:
            total_force += force
            total_torque += physics.get_omega(force, self.orientation, point, self.moment_of_inertia)
        total_force = physics.limit_specific_force(total_force, gravity, self.mass, self.max_g)
        self.omega += total_torque / SIMULATION_TICKRATE
        self.omega *= 0.95
        self.omega = np.clip(self.omega, -180.0, 180.0)
        self.orientation = physics.integrate_orientation(self.orientation, self.omega, 1.0 / SIMULATION_TICKRATE)
        self.velocity = physics.integrate_velocity(self.velocity, total_force / self.mass, 1.0 / SIMULATION_TICKRATE)
        self.position = physics.integrate_position(self.position, self.velocity, 1.0 / SIMULATION_TICKRATE)
        if self.position[1] < 0.0:
            self.alive = False


class missile(entity):
    def __init__(self, starting_position, starting_velocity, starting_orientation, mass: float, thrust_force: float, max_g: float, reference_area: float, min_drag_coefficient: float, max_drag_coefficient: float, max_lift_coefficient: float, moment_of_inertia_roll: float, moment_of_inertia_pitch: float, moment_of_inertia_yaw: float, optimal_lift_aoa: float, length: float, target_entity, chase_strategy, viz_shape: dict, burn_time_s: float = 2.75, propellant_mass_fraction: float = 0.25, max_turn_rate_dps: float = 120.0):
        super().__init__(starting_position, starting_velocity, starting_orientation, mass, reference_area, min_drag_coefficient, max_drag_coefficient, max_lift_coefficient, moment_of_inertia_roll, moment_of_inertia_pitch, moment_of_inertia_yaw, optimal_lift_aoa, viz_shape)
        self.thrust_force = float(thrust_force)
        self.max_g = float(max_g)
        self.target_entity = target_entity
        self.chase_strategy = chase_strategy
        self.length = float(length)
        self.control_inputs = {'pitch': 0.0, 'yaw': 0.0}
        self.explosion_radius = 10.0
        self.lifetime = 0.0
        self.initial_mass = float(mass)
        self.propellant_mass_fraction = float(np.clip(propellant_mass_fraction, 0.0, 0.8))
        self.propellant_mass = self.initial_mass * self.propellant_mass_fraction
        self.dry_mass = self.initial_mass - self.propellant_mass
        self.burn_time_s = max(0.1, float(burn_time_s))
        self.mass_flow_rate = self.propellant_mass / self.burn_time_s
        self.max_turn_rate_dps = max(1.0, float(max_turn_rate_dps))
        self.guidance_command_angle_deg = 20.0
        self.max_lifetime_s = 20.0
        self.q_soft_limit_pa = 500000.0
        self.q_hard_limit_pa = 1000000.0
        # Generic bounded lateral-stability assumptions. These are not
        # platform measurements and must remain provenance-labelled.
        self.lateral_reference_area_ratio = 0.15
        self.lateral_coefficient_scale = 0.75
        self.sideslip_feedback_gain = 0.6
        self.sideslip_feedback_angle_deg = 30.0
        # C3O candidate-only stability-priority command allocation.
        # Values are generic development parameters, not platform data.
        self.beta_blend_start_deg = 25
        self.beta_blend_full_deg = 40
        self.minimum_guidance_weight = 0.2
        self.last_lateral_guidance_weight = 1.0
        self.last_lateral_stability_priority = 0.0
        self.last_lateral_stability_command = 0.0
        self.last_lateral_combined_yaw = 0.0

    @property
    def motor_burning(self):
        return self.lifetime <= self.burn_time_s and self.mass > self.dry_mass + 1e-9

    def think(self, entities):
        self.lifetime += 1.0 / SIMULATION_TICKRATE
        if self.lifetime >= self.max_lifetime_s:
            self.alive = False
            return
        if physics.get_distance(self.position, self.target_entity.position) < self.explosion_radius:
            self.alive = False
            self.target_entity.alive = False
            return
        if self.lifetime >= 2.0:
            direction_to_target = self.target_entity.position - self.position
            if np.dot(self.velocity, direction_to_target) < 0.0:
                self.alive = False
                return
        self.chase_strategy(self)

    def tick(self):
        air_density = physics.get_air_density(self.position[1])
        mach = physics.get_mach_number(self.velocity, self.position[1])
        aoa = physics.get_angle_of_attack(self.velocity, self.orientation)
        sideslip = physics.get_sideslip(self.velocity, self.orientation)
        cd = physics.get_drag_coefficient(aoa, self.min_drag_coefficient, self.max_drag_coefficient, mach)
        cl = physics.get_lift_coefficient(aoa, self.max_lift_coefficient, self.optimal_lift_aoa, 0.0, mach)
        side_cl = physics.get_lift_coefficient(
            sideslip,
            self.max_lift_coefficient * self.lateral_coefficient_scale,
            self.optimal_lift_aoa,
            0.0,
            mach,
        )
        lateral_area = self.reference_area * self.lateral_reference_area_ratio
        q = physics.get_dynamic_pressure(self.velocity, air_density)
        protection = physics.get_dynamic_pressure_control_scale(q, self.q_soft_limit_pa, self.q_hard_limit_pa, 0.05)
        pitch = self.control_inputs["pitch"] * protection
        raw_guidance_yaw = float(
            np.clip(self.control_inputs["yaw"], -1.0, 1.0)
        )
        beta_abs = abs(float(sideslip))
        blend_span_deg = max(
            1e-6,
            self.beta_blend_full_deg - self.beta_blend_start_deg,
        )
        stability_priority = float(
            np.clip(
                (
                    beta_abs - self.beta_blend_start_deg
                )
                / blend_span_deg,
                0.0,
                1.0,
            )
        )
        guidance_weight = 1.0 - stability_priority * (
            1.0 - self.minimum_guidance_weight
        )
        sideslip_feedback = self.sideslip_feedback_gain * float(
            np.clip(
                sideslip / self.sideslip_feedback_angle_deg,
                -1.0,
                1.0,
            )
        )
        combined_yaw = float(
            np.clip(
                guidance_weight * raw_guidance_yaw
                + sideslip_feedback,
                -1.0,
                1.0,
            )
        )
        yaw = combined_yaw * protection
        self.last_lateral_guidance_weight = guidance_weight
        self.last_lateral_stability_priority = stability_priority
        self.last_lateral_stability_command = sideslip_feedback
        self.last_lateral_combined_yaw = combined_yaw
        throttle = 1.0 if self.motor_burning else 0.0
        gravity = physics.get_gravity_force(self.mass)
        forces = [
            [gravity, np.zeros(3)],
            [physics.get_thrust_force(self.orientation, throttle, self.thrust_force, self.length), np.array([-0.5 * self.length, 0.0, 0.0])],
            [physics.get_drag_force(self.velocity, air_density, self.reference_area, cd), np.array([-0.2 * self.length, 0.0, 0.0])],
            [physics.get_lift_force(self.velocity, self.reference_area, cl, self.orientation, air_density), np.array([-0.2 * self.length, 0.0, 0.0])],
            [physics.get_sideforce_force(self.velocity, lateral_area, side_cl, self.orientation, air_density), np.array([-0.45 * self.length, 0.0, 0.0])],
            [physics.get_elevator_force(self.velocity, air_density, self.reference_area, self.max_lift_coefficient, self.orientation, pitch, self.optimal_lift_aoa, 0.5, 1.0), np.array([-0.45 * self.length, 0.0, 0.0])],
            [physics.get_rudder_force(self.velocity, air_density, self.reference_area, self.max_lift_coefficient, self.orientation, yaw, self.optimal_lift_aoa, 0.5, 1.0), np.array([-0.45 * self.length, 0.0, 0.0])],
        ]
        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        for force, point in forces:
            total_force += force
            total_torque += physics.get_omega(force, self.orientation, point, self.moment_of_inertia)
        total_force = physics.limit_specific_force(total_force, gravity, self.mass, self.max_g)
        self.omega += total_torque / SIMULATION_TICKRATE
        self.omega *= 0.92
        self.omega = np.clip(self.omega, -self.max_turn_rate_dps, self.max_turn_rate_dps)
        self.orientation = physics.integrate_orientation(self.orientation, self.omega, 1.0 / SIMULATION_TICKRATE)
        self.velocity = physics.integrate_velocity(self.velocity, total_force / self.mass, 1.0 / SIMULATION_TICKRATE)
        self.position = physics.integrate_position(self.position, self.velocity, 1.0 / SIMULATION_TICKRATE)
        if throttle > 0.0:
            self.mass = max(self.dry_mass, self.mass - self.mass_flow_rate / SIMULATION_TICKRATE)
        if self.position[1] < 0.0:
            self.alive = False
