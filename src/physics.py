import numpy as np
'''
ALL POSITIONS,VELOCITIES,ACCELERATIONS & FORCES ARE IN WORLD-SPACE

FORCES ACTING ON PARTICLES:
- Thrust
    In direction of the particle's forward axis.
    Magnitude linear to throttle
    Applied at the back of the particle (technically useless, but fun)
    *TORQUE ABSENT*

- Gravity
    Constant downward acceleration of 9.81 m/s²
    Must be transformed into a force
    Applied at the center of mass
    *TORQUE ABSENT*
    
- Drag
    In the opposite direction of velocity.
    Magnitude based on "0.5 * ad * v_mag**2 * reference_area * cd * drag_direction"
    Where ad is a dynamic air density based on altitude
    Where v_mag is the magnitude of the velocity vector
    Where cd is a simplified drag coefficient based on angle of attack
    IMPORTANT: reference_area & cd are based on the fuselage WITHOUT control surfaces (these are calculated separately)
    *TORQUE PRESENT*

- Lift
    Perpendicular to both velocity and right axis. ("Up" direction of velocity based on jet's roll)
    Magnitude based on "0.5 * ad * v_mag**2 * reference_area * cl * lift_direction"
    Where ad is a dynamic air density based on altitude
    Where v_mag is the magnitude of the velocity vector
    Where cl is a simplified lift coefficient based on angle of attack
    *TORQUE PRESENT*

- Sideforce
    Perpendicular to both velocity and up axis.
    *lift-based force* that creates a yaw restoring motion by the vertical stabilizer
    *TORQUE PRESENT*

- Control_Elevators
    *lift-based force* that creates a pitch motion
    Applied at the tail of the aircraft
    *TORQUE PRESENT*

- Control_Ailerons
    *lift-based force* that creates a roll motion
    Applied at the wings of the aircraft
    *TORQUE PRESENT*

- Control_Rudder
    *lift-based force* that creates a yaw motion
    Applied at the vertical stabilizer of the aircraft
    *TORQUE PRESENT*
'''

def get_air_density(altitude: float) -> float:
    """Return density from a bounded piecewise ISA approximation (0-20 km)."""
    h = max(0.0, float(altitude))
    if h <= 11000.0:
        temperature = 288.15 - 0.0065 * h
        return float(1.225 * (temperature / 288.15) ** 4.255879812716677)
    temperature = 216.65
    rho_11 = 0.3639176481016034
    scale_height = 287.05 * temperature / 9.80665
    return float(rho_11 * np.exp(-(h - 11000.0) / scale_height))

def get_angle_of_attack(velocity: np.ndarray, R: np.ndarray) -> float:
    '''Returns the aoa angle in degrees'''
    '''Between -180 and 180 degrees'''
    '''No simplification applied'''
    if np.linalg.norm(velocity) < 1e-12:
        return 0.0 #Value doesn't matter
    v_body = R.T @ velocity #Get velocity in body space
    aoa = -np.degrees(np.arctan2(v_body[1], v_body[0])) #deg
    return aoa

def get_sideslip(velocity: np.ndarray, R: np.ndarray) -> float:
    '''Returns the sideslip angle in degrees'''
    '''Between -180 and 180 degrees'''
    if np.linalg.norm(velocity) < 1e-12:
        return 0.0 #Value doesn't matter
    v_body = R.T @ velocity #Get velocity in body space
    sideslip = -np.degrees(np.arctan2(v_body[2], v_body[0]))

    return sideslip

def get_lift_coefficient(aoa: float, max_lift_coefficient: float, optimal_aoa: float, zero_lift_aoa: float, mach: float = 0.0) -> float:
    """Return a bounded low-order lift coefficient with explicit Mach attenuation."""
    sign = 1.0
    angle = (float(aoa) + 90.0) % 180.0 - 90.0 - float(zero_lift_aoa)
    if angle < 0.0:
        sign = -1.0
        angle = abs(angle)
    optimal = max(float(optimal_aoa), 1e-9)
    cl_max = float(max_lift_coefficient)
    non_stall_slope = cl_max / optimal
    if angle <= optimal:
        base = sign * angle * non_stall_slope
    else:
        base = sign * (-(4.0 * cl_max) / ((60.0 + optimal) ** 3) * (angle - (90.0 - optimal) / 2.0 - optimal) ** 3 + 0.5 * cl_max)
    m = max(0.0, float(mach))
    if m <= 0.8:
        scale = 1.0
    elif m <= 1.2:
        scale = 1.0 - 0.25 * ((m - 0.8) / 0.4)
    elif m <= 3.0:
        scale = 0.75 - 0.30 * ((m - 1.2) / 1.8)
    else:
        scale = max(0.20, 0.45 - 0.05 * (m - 3.0))
    return float(np.clip(base * scale, -cl_max, cl_max))

def get_drag_coefficient(aoa: float, min_drag_coefficient: float, max_drag_coefficient: float, mach: float = 0.0) -> float:
    """Return low-order angle drag with an explicit transonic/supersonic multiplier."""
    angle = abs((float(aoa) + 180.0) % 360.0 - 180.0)
    cd_min = float(min_drag_coefficient)
    cd_max = float(max_drag_coefficient)
    if angle <= 90.0:
        base = ((cd_max - cd_min) / 8100.0) * angle**2 + cd_min
    else:
        base = ((cd_max - cd_min) / 10125.0) * (angle - 180.0)**2 + 0.2 * (cd_max - cd_min) + cd_min
    m = max(0.0, float(mach))
    if m <= 0.3:
        multiplier = 1.0
    elif m <= 0.8:
        multiplier = 1.0 + 0.15 * ((m - 0.3) / 0.5)
    elif m <= 1.2:
        multiplier = 1.15 + 1.05 * ((m - 0.8) / 0.4)
    elif m <= 3.0:
        multiplier = 2.20 - 0.60 * ((m - 1.2) / 1.8)
    elif m <= 5.0:
        multiplier = 1.60 - 0.20 * ((m - 3.0) / 2.0)
    else:
        multiplier = min(3.0, 1.40 + 0.10 * (m - 5.0))
    return float(max(0.0, base * multiplier))

def get_speed_of_sound(altitude: float) -> float:
    """Return diagnostic local speed of sound for the same 0-20 km ISA contract."""
    h = max(0.0, float(altitude))
    temperature = 288.15 - 0.0065 * min(h, 11000.0)
    return float(np.sqrt(1.4 * 287.05 * temperature))


def get_mach_number(velocity: np.ndarray, altitude: float) -> float:
    return float(np.linalg.norm(velocity)) / max(get_speed_of_sound(altitude), 1e-9)


def get_dynamic_pressure_control_scale(dynamic_pressure: float, soft_limit_pa: float, hard_limit_pa: float, minimum_scale: float = 0.1) -> float:
    q = max(0.0, float(dynamic_pressure))
    soft = float(soft_limit_pa)
    hard = float(hard_limit_pa)
    minimum = float(np.clip(minimum_scale, 0.0, 1.0))
    if q <= soft:
        return 1.0
    if hard <= soft or q >= hard:
        return minimum
    fraction = (q - soft) / (hard - soft)
    return float(1.0 - fraction * (1.0 - minimum))


def limit_specific_force(total_force: np.ndarray, gravity_force: np.ndarray, mass: float, max_g: float) -> np.ndarray:
    """Limit non-gravity specific force to the declared load-factor contract."""
    non_gravity = np.asarray(total_force, dtype=float) - np.asarray(gravity_force, dtype=float)
    limit = max(0.0, float(max_g)) * float(mass) * 9.81
    magnitude = float(np.linalg.norm(non_gravity))
    if limit <= 0.0:
        return np.asarray(gravity_force, dtype=float).copy()
    if magnitude <= limit or magnitude < 1e-12:
        return np.asarray(total_force, dtype=float)
    return np.asarray(gravity_force, dtype=float) + non_gravity * (limit / magnitude)


def get_lift_force(velocity: np.ndarray, reference_area: float, cl: float, R: np.ndarray, air_density: float) -> np.ndarray:
    '''Returns the lift force vector'''
    velocity_magnitude = np.linalg.norm(velocity)
    lift_magnitude = 0.5 * air_density * velocity_magnitude**2 * reference_area * cl
    lift_force = lift_magnitude * get_lift_dir(velocity, R)
    return lift_force

def get_gravity_force(mass: float) -> np.ndarray:
    '''Returns the gravity force vector'''
    gravity_force = np.array([0.0, -9.81 * mass, 0.0])
    return gravity_force

def get_thrust_force(R:np.ndarray, throttle: float, thrust_force: float, length: float) -> np.ndarray:
    '''Returns the thrust force vector'''
    return thrust_force * throttle * get_forward_dir(R)

def get_drag_force(velocity: np.ndarray, air_density: float, reference_area: float, drag_coefficient: float) -> np.ndarray:
    '''Returns the drag force vector'''
    velocity_magnitude = np.linalg.norm(velocity)
    if velocity_magnitude < 1e-12:
        return np.zeros(3)
    drag_direction = -velocity / velocity_magnitude
    drag_magnitude = 0.5 * air_density * velocity_magnitude**2 * reference_area * drag_coefficient
    return drag_magnitude * drag_direction

def get_sideforce_force(velocity: np.ndarray, reference_area: float, side_cl: float, R: np.ndarray, air_density: float) -> np.ndarray:
    '''Returns the sideforce force vector'''
    '''Rotates R by 90 degrees on roll axis so "up" points towards right wing'''
    side_R = np.column_stack((get_forward_dir(R), get_right_dir(R), -get_up_dir(R)))
    return get_lift_force(velocity, reference_area, side_cl, side_R, air_density)

def get_dynamic_pressure(velocity: np.ndarray, air_density: float):
    '''Returns the dynamic pressure'''
    velocity_magnitude = float(np.linalg.norm(velocity))
    if velocity_magnitude < 1e-12: 
        return 0.0
    return float(0.5 * float(air_density) * velocity_magnitude * velocity_magnitude)


def get_control_effectiveness(velocity: np.ndarray, R: np.ndarray, optimal_lift_aoa: float, stall_fade_degrees: float = 25.0, minimum_effectiveness: float = 0.2) -> float:
    '''Returns the control effectiveness based on aoa'''
    angle_of_attack = abs(float(get_angle_of_attack(velocity, R)))
    if angle_of_attack <= float(optimal_lift_aoa): return 1.0
    if float(stall_fade_degrees) < 1e-12: return float(minimum_effectiveness)
    return float(max(float(minimum_effectiveness), 1.0 - (angle_of_attack - float(optimal_lift_aoa)) / float(stall_fade_degrees)))


def get_control_force_magnitude(dynamic_pressure: float, reference_area: float, surface_area_multiplier: float, max_lift_coefficient: float, lift_coefficient_multiplier: float, control_input: float, control_effectiveness: float) -> float:
    '''Returns the control force magnitude'''
    return float(dynamic_pressure) * float(reference_area) * float(surface_area_multiplier) * float(max_lift_coefficient) * float(lift_coefficient_multiplier) * float(control_input) * float(control_effectiveness)

def get_elevator_force(velocity: np.ndarray, air_density: float, reference_area: float, max_lift_coefficient: float, R: np.ndarray, pitch_input: float, optimal_lift_aoa: float, elevator_surface_area_multiplier: float = 0.18, elevator_lift_coefficient_multiplier: float = 1.0) -> np.ndarray:
    '''Returns the elevator force vector'''
    '''Placeholder function for now'''
    if abs(float(pitch_input)) < 1e-12: 
        return np.zeros(3)
    velocity_magnitude = np.linalg.norm(velocity)
    dynamic_pressure = get_dynamic_pressure(velocity, air_density)
    if velocity_magnitude < 1e-12: 
        return np.zeros(3)
    return get_control_force_magnitude(dynamic_pressure, reference_area, elevator_surface_area_multiplier, max_lift_coefficient, elevator_lift_coefficient_multiplier, pitch_input, get_control_effectiveness(velocity, R, optimal_lift_aoa)) * get_lift_dir(velocity, R)

def get_aileron_force(velocity: np.ndarray, air_density: float, reference_area: float, max_lift_coefficient: float, R: np.ndarray, roll_input: float, optimal_lift_aoa: float, aileron_surface_area_multiplier: float = 0.06, aileron_lift_coefficient_multiplier: float = 0.8)-> np.ndarray:
    '''Returns the aileron force vector'''
    '''Placeholder function for now'''
    if abs(float(roll_input)) < 1e-12: 
        return np.zeros(3)
    velocity_magnitude = np.linalg.norm(velocity)
    dynamic_pressure = get_dynamic_pressure(velocity, air_density)
    if velocity_magnitude < 1e-12: 
        return np.zeros(3)
    aileron_force_vector = get_control_force_magnitude(dynamic_pressure, reference_area, aileron_surface_area_multiplier, max_lift_coefficient, aileron_lift_coefficient_multiplier, roll_input, get_control_effectiveness(velocity, R, optimal_lift_aoa)) * get_lift_dir(velocity, R)
    return aileron_force_vector

def get_rudder_force(velocity: np.ndarray, air_density: float, reference_area: float, max_lift_coefficient: float, R: np.ndarray, yaw_input: float, optimal_lift_aoa: float, rudder_surface_area_multiplier: float = 0.10, rudder_lift_coefficient_multiplier: float = 0.7) -> np.ndarray:
    '''Returns the rudder force vector'''
    '''Placeholder function for now'''
    if abs(float(yaw_input)) < 1e-12: 
        return np.zeros(3)
    velocity_magnitude = np.linalg.norm(velocity)
    dynamic_pressure = get_dynamic_pressure(velocity, air_density)
    if velocity_magnitude < 1e-12: 
        return np.zeros(3)
    airflow_direction = -velocity / velocity_magnitude
    right_dir = get_right_dir(R)
    rudder_force_direction = right_dir - np.dot(right_dir, airflow_direction) * airflow_direction
    rudder_force_direction_magnitude = float(np.linalg.norm(rudder_force_direction))
    if rudder_force_direction_magnitude < 1e-12:
        rudder_force_direction = np.cross(airflow_direction, get_up_dir(R))
        rudder_force_direction_magnitude = float(np.linalg.norm(rudder_force_direction))
        if rudder_force_direction_magnitude < 1e-12: 
            return np.zeros(3)
    return get_control_force_magnitude(dynamic_pressure, reference_area, rudder_surface_area_multiplier, max_lift_coefficient, rudder_lift_coefficient_multiplier, yaw_input, get_control_effectiveness(velocity, R, optimal_lift_aoa)) * (rudder_force_direction / rudder_force_direction_magnitude)


def get_forward_dir(R: np.ndarray) -> np.ndarray: #Returns the "forward" unit vector of the plane
    return R @ np.array([1.0, 0.0, 0.0])

def get_up_dir(R: np.ndarray) -> np.ndarray: #Returns the "up" unit vector of the plane
    return R @ np.array([0.0, 1.0, 0.0])

def get_right_dir(R: np.ndarray) -> np.ndarray: #Returns the "right" unit vector of the plane (kinda useful on edge cases)
    return R @ np.array([0.0, 0.0, 1.0])

def get_lift_dir(velocity: np.ndarray, R: np.ndarray) -> np.ndarray:
    up_direction = get_up_dir(R)
    velocity_magnitude = np.linalg.norm(velocity)
    if velocity_magnitude < 1e-12:
        return up_direction
    airflow_direction = -velocity / velocity_magnitude  # unit airflow direction
    lift = up_direction - np.dot(up_direction, airflow_direction) * airflow_direction  # projection of up onto plane ⟂ flow
    lift_norm = np.linalg.norm(lift)
    if lift_norm < 1e-12:
        # up is (almost) parallel to flow -> pick any stable perpendicular direction
        right = get_right_dir(R)
        lift = np.cross(airflow_direction, right)
        lift_norm = np.linalg.norm(lift)
        if lift_norm < 1e-12:
            return up_direction  # absolute fallback
    return lift / lift_norm

def get_omega(force: np.ndarray, R: np.ndarray, application_point: np.ndarray, moment_of_inertia: np.ndarray) -> np.ndarray:
    '''MAIN FUNCTION TO GET THE ANGULAR ACCELERATION OFF A FORCE'''
    F_body = R.T @ force
    torque_body = np.cross(application_point, F_body)
    alpha_body = torque_body / moment_of_inertia
    return np.degrees(alpha_body)

def align_roll_to_velocity(R: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    velocity_magnitude = float(np.linalg.norm(velocity))
    if velocity_magnitude < 1e-12: return R

    forward_direction = get_forward_dir(R)
    forward_magnitude = float(np.linalg.norm(forward_direction))
    if forward_magnitude < 1e-12: return R
    forward_direction = forward_direction / forward_magnitude

    velocity_direction = velocity / velocity_magnitude
    lateral_direction = velocity_direction - np.dot(velocity_direction, forward_direction) * forward_direction
    lateral_magnitude = float(np.linalg.norm(lateral_direction))
    if lateral_magnitude < 1e-12: return R

    up_direction = (-lateral_direction) / lateral_magnitude
    right_direction = np.cross(forward_direction, up_direction)
    right_magnitude = float(np.linalg.norm(right_direction))
    if right_magnitude < 1e-12: return R
    right_direction = right_direction / right_magnitude

    return np.column_stack((forward_direction, np.cross(right_direction, forward_direction), right_direction))


def get_distance(pos0: np.ndarray, pos1: np.ndarray) -> float:
    '''Returns the distance between two positions'''
    return np.linalg.norm(pos0 - pos1).item()

def integrate_velocity(velocity: np.ndarray, acceleration: np.ndarray, dt: float) -> np.ndarray:
    return velocity + acceleration * dt

def integrate_position(position: np.ndarray, velocity: np.ndarray, dt: float) -> np.ndarray:
    return position + velocity * dt

def integrate_orientation(R: np.ndarray, omega_deg_s: np.ndarray, dt: float) -> np.ndarray:
    w = np.radians(omega_deg_s)
    th = float(np.linalg.norm(w) * dt)
    if th < 1e-12:
        return R
    axis = w / np.linalg.norm(w)
    K = np.array([[0.0, -axis[2],  axis[1]],
                  [axis[2],  0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]], dtype=float)
    dR = np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)

    Rn = R @ dR

    # re-orthonormalize (SVD)
    U, _, Vt = np.linalg.svd(Rn)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0.0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn



def get_R_from_angles(pitch: float, roll: float, yaw: float) -> np.ndarray:
    '''Returns rotation matrix from pitch/roll/yaw angles in degrees'''
    '''Order: yaw -> pitch -> roll'''
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)
    yaw_rad = np.radians(yaw)
    
    cp = np.cos(pitch_rad)
    sp = np.sin(pitch_rad)
    cr = np.cos(roll_rad)
    sr = np.sin(roll_rad)
    cy = np.cos(yaw_rad)
    sy = np.sin(yaw_rad)
    
    R = np.array([[cp*cy, sp*sr*cy - cr*sy, sp*cr*cy + sr*sy],
                  [cp*sy, sp*sr*sy + cr*cy, sp*cr*sy - sr*cy],
                  [-sp,   cp*sr,            cp*cr           ]], dtype=float)
    return R

def get_angles_from_R(R: np.ndarray) -> tuple[float, float, float]:
    '''Returns pitch/roll/yaw angles in degrees from rotation matrix'''
    '''Returns (pitch, roll, yaw)'''
    pitch = np.degrees(-np.arcsin(np.clip(R[2, 0], -1.0, 1.0)))
    
    if np.abs(R[2, 0]) < 0.99999:
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        # Gimbal lock case
        roll = 0.0
        if R[2, 0] < 0:
            yaw = np.degrees(np.arctan2(-R[0, 1], R[1, 1]))
        else:
            yaw = np.degrees(np.arctan2(R[0, 1], R[1, 1]))
    
    return float(pitch), float(roll), float(yaw)


if __name__=="__main__":
    '''Test to plot out Cl & Cd depending on aoa'''
    '''Mainly used to display the coefficients simplification'''
    import numpy as np
    import matplotlib.pyplot as plt
    aoa=np.arange(-180.0,180.0+1e-9,0.5)
    cl=np.array([get_lift_coefficient(a,1.8,15.0,-2.0) for a in aoa],float)
    cd=np.array([get_drag_coefficient(a,0.02,2) for a in aoa],float)
    fig=plt.figure(figsize=(10.5,5.8)); ax=fig.add_subplot(1,1,1)
    ax.plot(aoa,cl,color="blue",linewidth=2.6,label="Cl(α)")
    ax.plot(aoa,cd,color="red",linewidth=2.2,label="Cd(α)")
    ax.axhline(0.0,linewidth=1.0,alpha=0.35)
    ax.set_title("Lift/Drag coefficients vs AoA"); ax.set_xlabel("AoA α (deg)"); ax.set_ylabel("Coefficient")
    ax.set_xlim(-180,180); ax.grid(True,alpha=0.22); ax.legend(loc="upper right",framealpha=0.92)
    fig.tight_layout(); plt.show()
