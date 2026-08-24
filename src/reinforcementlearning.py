import os

# These controls are duplicated here so non-main entry points remain safe.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("JET_EXPERIMENT_SEED", "0")

import math
import pickle
import random
from pathlib import Path
import numpy as np
import tensorflow as tf
import physics

REWARD_PHASE=3
PHASE=3
TRAINING=os.environ.get('JET_TRAINING','1').strip().lower() not in ('0','false','no','n')
ACTION_SKIP=4

EXPERIMENT_SEED=int(os.environ.get('JET_EXPERIMENT_SEED','0'))
np.random.seed(EXPERIMENT_SEED)
_np_rng=np.random.default_rng(EXPERIMENT_SEED)
random.seed(EXPERIMENT_SEED)
tf.keras.utils.set_random_seed(EXPERIMENT_SEED)
try:
    tf.config.experimental.enable_op_determinism()
except AttributeError as exc:
    raise RuntimeError(
        "TensorFlow deterministic operations are unavailable in this runtime."
    ) from exc
_tf_rng=tf.random.Generator.from_seed(EXPERIMENT_SEED)
_legacy_checkpoint_payload=None

DT=1.0/60.0
SD=34
AD=4
CAP=250000
BATCH=256
START=5000
UPD_PER=4
GAMMA=0.99
TAU=0.005
LR=3e-4
LOG_EVERY=200
SAVE_EVERY=2000

_ROOT=Path(__file__).resolve().parents[2]
_MODEL_TAG=os.environ.get('JET_MODEL_TAG','sac_ai').strip() or 'sac_ai'
_DIR=_ROOT/'res'/'models'/_MODEL_TAG
_MODEL=_DIR/'sac_ai'  # Old format (for backward compatibility)
_LOG=_DIR/'train.log'
# New separate file paths
_MODEL_ACTOR=_DIR/'actor_weights.pkl'
_MODEL_Q1=_DIR/'q1_weights.pkl'
_MODEL_Q2=_DIR/'q2_weights.pkl'
_MODEL_Q1T=_DIR/'q1t_weights.pkl'
_MODEL_Q2T=_DIR/'q2t_weights.pkl'
_MODEL_LOG_ALPHA=_DIR/'log_alpha.pkl'
_MODEL_BUF=_DIR/'replay_buffer.pkl'
_MODEL_STEPS=_DIR/'steps.pkl'
_MODEL_UPD=_DIR/'updates.pkl'

_actor=_q1=_q2=_q1t=_q2t=None
_ao=_q1o=_q2o=_alo=None
_log_alpha=None

_s=_a=_r=_s2=_d=None
_ptr=0
_sz=0

_started=False
_last_s=None
_last_a=None
_hold=0
_racc=0.0
_steps=0
_upd=0
_rhist=[]


def configure_experiment_seed(seed):
    """Configure all stochastic generators before model initialization."""
    global EXPERIMENT_SEED,_np_rng,_tf_rng
    if _actor is not None:
        raise RuntimeError(
            "Experiment seed cannot be changed after model initialization."
        )
    EXPERIMENT_SEED=int(seed)
    os.environ['JET_EXPERIMENT_SEED']=str(EXPERIMENT_SEED)
    random.seed(EXPERIMENT_SEED)
    np.random.seed(EXPERIMENT_SEED)
    _np_rng=np.random.default_rng(EXPERIMENT_SEED)
    tf.keras.utils.set_random_seed(EXPERIMENT_SEED)
    _tf_rng=tf.random.Generator.from_seed(EXPERIMENT_SEED)
    return EXPERIMENT_SEED


def _missiles(entities, jet):
    jp=np.asarray(getattr(jet,'position',[0.0,0.0,0.0]),dtype=float).reshape(3)
    jv=np.asarray(getattr(jet,'velocity',[0.0,0.0,0.0]),dtype=float).reshape(3)
    out=[]
    for e in entities:
        if e is jet or not bool(getattr(e,'alive',True)):
            continue
        if e.__class__.__name__.lower()!='missile':
            continue
        mp=np.asarray(getattr(e,'position',[0.0,0.0,0.0]),dtype=float).reshape(3)
        mv=np.asarray(getattr(e,'velocity',[0.0,0.0,0.0]),dtype=float).reshape(3)
        rel=mp-jp
        d=float(np.linalg.norm(rel))
        if d>1e-6 and np.isfinite(d):
            out.append((d,rel,mv-jv,mv,mp))
    out.sort(key=lambda t:t[0])
    return out


def _state(entities, jet):
    p=np.asarray(getattr(jet,'position',[0.0,0.0,0.0]),dtype=float).reshape(3)
    v=np.asarray(getattr(jet,'velocity',[0.0,0.0,0.0]),dtype=float).reshape(3)
    R=np.asarray(getattr(jet,'orientation',np.eye(3)),dtype=float).reshape(3,3)
    w=np.asarray(getattr(jet,'omega',[0.0,0.0,0.0]),dtype=float).reshape(3)
    if not (np.isfinite(p).all() and np.isfinite(v).all() and np.isfinite(R).all() and np.isfinite(w).all()):
        return np.zeros(SD,dtype=np.float32)
    vb=R.T@v
    sp=float(np.linalg.norm(v))
    aoa=float(physics.get_angle_of_attack(v,R))
    slip=float(physics.get_sideslip(v,R))
    f=physics.get_forward_dir(R)
    u=physics.get_up_dir(R)
    ms=_missiles(entities,jet)
    m=[]
    for i in range(3):
        if i<len(ms):
            _,rel,dv,_,_=ms[i]
            rb=R.T@rel
            dvb=R.T@dv
            m.extend((rb/20000.0).tolist())
            m.extend((dvb/600.0).tolist())
        else:
            m.extend([0.0]*6)
    s=np.concatenate([
        np.array([p[1]/10000.0, sp/600.0],dtype=float),
        np.clip(vb/600.0,-5.0,5.0),
        np.clip(w/6.0,-5.0,5.0),
        np.array([aoa/45.0, slip/45.0],dtype=float),
        np.clip(np.concatenate([f,u]),-2.0,2.0),
        np.clip(np.array(m,dtype=float),-5.0,5.0)
    ],axis=0).astype(np.float32)
    if s.shape[0]!=SD:
        t=np.zeros(SD,dtype=np.float32)
        n=min(SD,int(s.shape[0]))
        t[:n]=s[:n]
        return t
    return s


def _reward(entities, jet):
    """
    Safety-aware reward shaping v1.

    Design intent:
    - Preserve nominal survival and speed/altitude tracking.
    - Penalize boundary approach before the hard 20 km box violation occurs.
    - Penalize low-altitude exposure before ground impact.
    - Penalize missile close-calls explicitly at 100 m, 30 m, and 10 m scales.
    - Keep the reward continuous where possible, but terminate on actual death,
      nonfinite states, and direct 10 m missile intercept exposure.
    """
    if not bool(getattr(jet, 'alive', True)):
        return -500.0, True

    p = np.asarray(getattr(jet, 'position', [0.0, 0.0, 0.0]), dtype=float).reshape(3)
    v = np.asarray(getattr(jet, 'velocity', [0.0, 0.0, 0.0]), dtype=float).reshape(3)
    R = np.asarray(getattr(jet, 'orientation', np.eye(3)), dtype=float).reshape(3, 3)

    if not (np.isfinite(p).all() and np.isfinite(v).all() and np.isfinite(R).all()):
        setattr(jet, 'alive', False)
        return -500.0, True

    alt = float(p[1])
    sp = float(np.linalg.norm(v))
    aoa = abs(float(physics.get_angle_of_attack(v, R)))
    slip = abs(float(physics.get_sideslip(v, R)))

    # Small living reward: encourages completing the episode, but is deliberately weak.
    r = 0.03 * DT

    # Speed corridor: avoid stall/low-energy states and excessive acceleration.
    r += 0.25 * np.clip((sp - 150.0) / 250.0, 0.0, 1.0) * DT
    r -= 1.10 * np.clip((150.0 - sp) / 150.0, 0.0, 2.0) ** 2 * DT
    r -= 0.35 * np.clip((sp - 560.0) / 200.0, 0.0, 3.0) ** 2 * DT

    # Altitude safety: soft floor before ground impact.
    r -= 2.80 * np.clip((3000.0 - alt) / 3000.0, 0.0, 3.0) ** 2 * DT
    r -= 1.20 * np.clip((alt - 8500.0) / 2500.0, 0.0, 3.0) ** 2 * DT

    if alt < 1500.0:
        r -= 8.0 * np.clip((1500.0 - alt) / 1500.0, 0.0, 2.0) ** 2 * DT
    if alt < 500.0:
        r -= 30.0 * np.clip((500.0 - alt) / 500.0, 0.0, 2.0) ** 2 * DT

    # Kinematic envelope penalties.
    if aoa > 16.0:
        r -= 3.00 * np.clip((aoa - 16.0) / 16.0, 0.0, 3.0) ** 2 * DT
    if slip > 8.0:
        r -= 1.50 * np.clip((slip - 8.0) / 12.0, 0.0, 3.0) ** 2 * DT

    # Box-boundary safety: penalize approach before violation.
    # Hard campaign box is |x|, |z| <= 20000 m. Start soft penalty at 16000 m.
    radial_xz = max(abs(float(p[0])), abs(float(p[2])))
    soft_box = 16000.0
    warn_box = 18000.0
    hard_box = 20000.0

    if radial_xz > soft_box:
        r -= 4.0 * np.clip((radial_xz - soft_box) / (hard_box - soft_box), 0.0, 2.0) ** 2 * DT
    if radial_xz > warn_box:
        r -= 12.0 * np.clip((radial_xz - warn_box) / (hard_box - warn_box), 0.0, 2.0) ** 2 * DT
    if radial_xz > hard_box:
        r -= 80.0 * (1.0 + np.clip((radial_xz - hard_box) / 1000.0, 0.0, 5.0)) ** 2 * DT

    # Phase-aware missile safety reward.
    phase = max(1, min(3, int(REWARD_PHASE)))
    ms = _missiles(entities, jet)

    if phase == 2:
        ms_filtered = ms[:1]
    elif phase >= 3:
        ms_filtered = ms[:3]
    else:
        ms_filtered = []

    jp = p
    for d, _, _, mv, mp in ms_filtered:
        rel = jp - mp
        dn = float(np.linalg.norm(rel))
        mvn = float(np.linalg.norm(mv))
        if dn < 1e-6 or mvn < 1e-6:
            continue

        c = float(np.clip(np.dot(mv / mvn, rel / dn), -1.0, 1.0))
        close = float(np.clip(1.0 - d / 8000.0, 0.0, 1.0))

        # Keep original directional logic, but make it less dominant than safety thresholds.
        r += 0.45 * close * (1.0 - c) * DT
        r -= 1.20 * close * (max(0.0, c) ** 2) * DT

        # Broad missile-distance pressure.
        r -= 2.20 * np.clip((2000.0 - d) / 2000.0, 0.0, 2.0) ** 2 * DT

        # Explicit close-call shaping aligned with campaign metrics.
        r -= 4.00 * np.clip((100.0 - d) / 100.0, 0.0, 1.0) ** 2 * DT
        r -= 14.0 * np.clip((30.0 - d) / 30.0, 0.0, 1.0) ** 2 * DT
        r -= 45.0 * np.clip((10.0 - d) / 10.0, 0.0, 1.0) ** 2 * DT

        # Direct intercept-scale exposure.
        if d <= 10.0:
            return float(r - 500.0), True

    return float(r), False


def _to_ctrl(a):
    a=np.clip(np.asarray(a,dtype=float).reshape(AD),-1.0,1.0)
    return float(a[0]),float(a[1]),float(a[2]),float((a[3]+1.0)*0.5)


def _build_actor():
    s=tf.keras.Input(shape=(SD,),dtype=tf.float32)
    x=tf.keras.layers.Dense(256,activation='relu')(s)
    x=tf.keras.layers.Dense(256,activation='relu')(x)
    mu=tf.keras.layers.Dense(AD)(x)
    ls=tf.keras.layers.Dense(AD)(x)
    return tf.keras.Model(s,[mu,ls])


def _build_q():
    s=tf.keras.Input(shape=(SD,),dtype=tf.float32)
    a=tf.keras.Input(shape=(AD,),dtype=tf.float32)
    x=tf.keras.layers.Concatenate()([s,a])
    x=tf.keras.layers.Dense(256,activation='relu')(x)
    x=tf.keras.layers.Dense(256,activation='relu')(x)
    q=tf.keras.layers.Dense(1)(x)
    return tf.keras.Model([s,a],q)



def _pi(mu,ls,det):
    ls=tf.clip_by_value(ls,-5.0,2.0)
    if det:
        a=tf.tanh(mu)
        lp=tf.zeros((tf.shape(mu)[0],),tf.float32)
        return a,lp
    std=tf.exp(ls)
    eps=_tf_rng.normal(shape=tf.shape(mu),dtype=mu.dtype)
    u=mu+std*eps
    a=tf.tanh(u)
    lp=-0.5*tf.reduce_sum(((u-mu)/(std+1e-6))**2+2.0*ls+math.log(2.0*math.pi),axis=1)
    lp-=tf.reduce_sum(tf.math.log(1.0-a*a+1e-6),axis=1)
    return a,lp



def _train_step(s,a,r,s2,d):
    alpha=tf.exp(_log_alpha)
    with tf.GradientTape(persistent=True) as t:
        mu2,ls2=_actor(s2,training=True)
        a2,lp2=_pi(mu2,ls2,False)
        q1t=_q1t([s2,a2],training=True)
        q2t=_q2t([s2,a2],training=True)
        y=r[:,None]+GAMMA*(1.0-d[:,None])*(tf.minimum(q1t,q2t)-alpha*lp2[:,None])
        q1=_q1([s,a],training=True)
        q2=_q2([s,a],training=True)
        lq1=tf.reduce_mean((q1-y)**2)
        lq2=tf.reduce_mean((q2-y)**2)
        mu,ls=_actor(s,training=True)
        ap,lp=_pi(mu,ls,False)
        qpi=tf.minimum(_q1([s,ap],training=True),_q2([s,ap],training=True))
        la=tf.reduce_mean(alpha*lp-tf.squeeze(qpi,1))
        lt=-tf.reduce_mean(_log_alpha*(lp-float(AD)))
    _q1o.apply_gradients(zip(t.gradient(lq1,_q1.trainable_variables),_q1.trainable_variables))
    _q2o.apply_gradients(zip(t.gradient(lq2,_q2.trainable_variables),_q2.trainable_variables))
    _ao.apply_gradients(zip(t.gradient(la,_actor.trainable_variables),_actor.trainable_variables))
    _alo.apply_gradients([(t.gradient(lt,[_log_alpha])[0],_log_alpha)])
    _log_alpha.assign(tf.clip_by_value(_log_alpha,-10.0,2.0))
    for v,tv in zip(_q1.variables,_q1t.variables):
        tv.assign((1.0-TAU)*tv+TAU*v)
    for v,tv in zip(_q2.variables,_q2t.variables):
        tv.assign((1.0-TAU)*tv+TAU*v)
    return lq1,lq2,la,alpha


_train_step=tf.function(_train_step,reduce_retracing=True)


def _add(s,a,r,s2,d):
    global _ptr,_sz
    _s[_ptr]=s
    _a[_ptr]=a
    _r[_ptr]=r
    _s2[_ptr]=s2
    _d[_ptr]=d
    _ptr=(_ptr+1)%CAP
    _sz=_sz+1 if _sz<CAP else CAP



def _batch():
    if _sz <= 0:
        raise RuntimeError("Replay sampling requested from an empty buffer.")
    idx=_np_rng.integers(0,_sz,size=BATCH,endpoint=False)
    return (
        tf.convert_to_tensor(_s[idx],dtype=tf.float32),
        tf.convert_to_tensor(_a[idx],dtype=tf.float32),
        tf.convert_to_tensor(_r[idx],dtype=tf.float32),
        tf.convert_to_tensor(_s2[idx],dtype=tf.float32),
        tf.convert_to_tensor(_d[idx],dtype=tf.float32),
    )




def _log_line(s):
    if not TRAINING:
        return
    _DIR.mkdir(parents=True,exist_ok=True)
    with open(_LOG,'a',encoding='utf-8') as f:
        f.write(s+'\n')




def _atomic_pickle(path,payload):
    tmp=path.with_name(path.name+'.tmp')
    with open(tmp,'wb') as f:
        pickle.dump(payload,f,protocol=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,path)


def _save():
    if not TRAINING:
        raise RuntimeError("Evaluation mode is read-only; checkpoint save denied.")
    _DIR.mkdir(parents=True,exist_ok=True)
    _atomic_pickle(_MODEL_ACTOR,_actor.get_weights())
    _atomic_pickle(_MODEL_Q1,_q1.get_weights())
    _atomic_pickle(_MODEL_Q2,_q2.get_weights())
    _atomic_pickle(_MODEL_Q1T,_q1t.get_weights())
    _atomic_pickle(_MODEL_Q2T,_q2t.get_weights())
    _atomic_pickle(_MODEL_LOG_ALPHA,float(_log_alpha.numpy()))
    _atomic_pickle(
        _MODEL_BUF,
        {'s':_s,'a':_a,'r':_r,'s2':_s2,'d':_d,'ptr':_ptr,'sz':_sz},
    )
    _atomic_pickle(_MODEL_STEPS,_steps)
    _atomic_pickle(_MODEL_UPD,_upd)




def _read_pickle(path):
    try:
        with open(path,'rb') as f:
            return pickle.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint artifact: {path}") from exc


def _load_weights():
    global _legacy_checkpoint_payload
    _legacy_checkpoint_payload=None

    if _MODEL_ACTOR.exists():
        _actor.set_weights(_read_pickle(_MODEL_ACTOR))
        weight_targets=(
            (_MODEL_Q1,_q1),
            (_MODEL_Q2,_q2),
            (_MODEL_Q1T,_q1t),
            (_MODEL_Q2T,_q2t),
        )
        if TRAINING:
            missing=[str(path) for path,model in weight_targets if not path.exists()]
            if not _MODEL_LOG_ALPHA.exists():
                missing.append(str(_MODEL_LOG_ALPHA))
            if missing:
                raise FileNotFoundError(
                    "Incomplete training checkpoint: "+", ".join(missing)
                )
        for path,model in weight_targets:
            if path.exists():
                model.set_weights(_read_pickle(path))
        if _MODEL_LOG_ALPHA.exists():
            _log_alpha.assign(float(_read_pickle(_MODEL_LOG_ALPHA)))
        return True

    if _MODEL.exists():
        payload=_read_pickle(_MODEL)
        if not isinstance(payload,dict) or 'w_actor' not in payload:
            raise RuntimeError("Legacy checkpoint does not contain actor weights.")
        _actor.set_weights(payload['w_actor'])
        for key,model in (
            ('w_q1',_q1),
            ('w_q2',_q2),
            ('w_q1t',_q1t),
            ('w_q2t',_q2t),
        ):
            if key in payload:
                model.set_weights(payload[key])
        if 'log_alpha' in payload:
            _log_alpha.assign(float(payload['log_alpha']))
        _legacy_checkpoint_payload=payload
        return True

    if TRAINING:
        return False
    raise FileNotFoundError(
        f"Evaluation actor checkpoint was not found under {_DIR}"
    )


def _load_training_state():
    global _ptr,_sz,_steps,_upd,_s,_a,_r,_s2,_d

    if _legacy_checkpoint_payload is not None:
        payload=_legacy_checkpoint_payload
        buffer_payload=payload.get('buf',{})
        _steps=int(payload.get('steps',0))
        _upd=int(payload.get('upd',0))
    else:
        state_paths=(_MODEL_BUF,_MODEL_STEPS,_MODEL_UPD)
        existing=[path.exists() for path in state_paths]
        if not any(existing):
            return False
        if not all(existing):
            missing=[
                str(path)
                for path,present in zip(state_paths,existing)
                if not present
            ]
            raise FileNotFoundError(
                "Incomplete mutable training state: "+", ".join(missing)
            )
        buffer_payload=_read_pickle(_MODEL_BUF)
        _steps=int(_read_pickle(_MODEL_STEPS))
        _upd=int(_read_pickle(_MODEL_UPD))

    required=('s','a','r','s2','d','ptr','sz')
    if not isinstance(buffer_payload,dict) or not all(
        key in buffer_payload for key in required
    ):
        raise RuntimeError("Replay-buffer checkpoint schema is invalid.")

    for key,target in (
        ('s',_s),
        ('a',_a),
        ('r',_r),
        ('s2',_s2),
        ('d',_d),
    ):
        source=np.asarray(buffer_payload[key])
        if source.shape != target.shape:
            raise RuntimeError(
                f"Replay-buffer shape mismatch for {key}: "
                f"expected={target.shape} observed={source.shape}"
            )
        target[:]=source

    _ptr=int(buffer_payload['ptr'])%CAP
    _sz=int(buffer_payload['sz'])
    if not 0 <= _sz <= CAP:
        raise RuntimeError(f"Replay-buffer size is invalid: {_sz}")
    return True




def initialize_deeplearning():
    global PHASE,_actor,_q1,_q2,_q1t,_q2t,_ao,_q1o,_q2o,_alo,_log_alpha,_s,_a,_r,_s2,_d
    PHASE=max(1, min(3, int(REWARD_PHASE)))
    if _actor is not None:
        return
    if TRAINING:
        _DIR.mkdir(parents=True,exist_ok=True)
    elif not _DIR.is_dir():
        raise FileNotFoundError(
            f"Evaluation model directory does not exist: {_DIR}"
        )
    _actor=_build_actor()
    _q1=_build_q(); _q2=_build_q()
    _q1t=_build_q(); _q2t=_build_q()
    _log_alpha=tf.Variable(0.0,dtype=tf.float32)
    _ao=tf.keras.optimizers.Adam(LR)
    _q1o=tf.keras.optimizers.Adam(LR)
    _q2o=tf.keras.optimizers.Adam(LR)
    _alo=tf.keras.optimizers.Adam(LR)
    ds=tf.zeros((1,SD),tf.float32)
    da=tf.zeros((1,AD),tf.float32)
    _actor(ds); _q1([ds,da]); _q2([ds,da]); _q1t([ds,da]); _q2t([ds,da])
    _q1t.set_weights(_q1.get_weights()); _q2t.set_weights(_q2.get_weights())
    _s=np.zeros((CAP,SD),dtype=np.float32)
    _a=np.zeros((CAP,AD),dtype=np.float32)
    _r=np.zeros((CAP,),dtype=np.float32)
    _s2=np.zeros((CAP,SD),dtype=np.float32)
    _d=np.zeros((CAP,),dtype=np.float32)
    _load_weights()
    if TRAINING:
        _load_training_state()




def cleanup_deeplearning(*,terminated=False,final_state=None):
    global _started,_last_s,_last_a,_hold,_racc
    if TRAINING and terminated:
        if (
            not _started
            or _last_s is None
            or _last_a is None
            or final_state is None
        ):
            raise RuntimeError(
                "A terminated cleanup requires the actual final state."
            )
        terminal_state=np.asarray(final_state,dtype=np.float32).reshape(SD)
        _add(_last_s,_last_a,_racc,terminal_state,1.0)
    if TRAINING:
        _save()
    _started=False
    _last_s=None
    _last_a=None
    _hold=0
    _racc=0.0



def jet_ai_step(entities, jet):
    global PHASE,_started,_last_s,_last_a,_hold,_racc,_steps,_upd,_rhist
    PHASE=max(1, min(3, int(REWARD_PHASE)))  # Clamp to 1-3
    initialize_deeplearning()
    s=_state(entities,jet)
    if not _started:
        _started=True
        _last_s=s
        st=tf.convert_to_tensor(s[None,:],tf.float32)
        mu,ls=_actor(st,training=False)
        a,_=_pi(mu,ls,not TRAINING)
        _last_a=a.numpy()[0].astype(np.float32)
        _hold=0
        _racc=0.0
        return _to_ctrl(_last_a)
    r,done=_reward(entities,jet)
    setattr(jet,'current_reward',float(r))
    _racc+=float(r)
    _hold+=1
    if done or _hold>=int(ACTION_SKIP):
        if TRAINING and _last_s is not None and _last_a is not None:
            _add(_last_s,_last_a,_racc,s,1.0 if done else 0.0)
            _rhist.append(float(_racc))
            if len(_rhist)>800:
                _rhist=_rhist[-800:]
            _steps+=1
            if _sz>=START:
                for _ in range(int(UPD_PER)):
                    bs,ba,br,bs2,bd=_batch()
                    _train_step(bs,ba,br,bs2,bd)
                    _upd+=1
                    if _upd%LOG_EVERY==0:
                        av=float(np.mean(_rhist[-200:])) if _rhist else 0.0
                        _log_line(f"[SAC update #{_upd//LOG_EVERY}] steps={_steps} avg_r/transition={av:.6f} phase={PHASE}")
                    if _upd%SAVE_EVERY==0:
                        _save()
        _last_s=s
        _racc=0.0
        _hold=0
        if done:
            _last_a=np.zeros((AD,),dtype=np.float32)
            return _to_ctrl(_last_a)
        st=tf.convert_to_tensor(s[None,:],tf.float32)
        mu,ls=_actor(st,training=False)
        a,_=_pi(mu,ls,not TRAINING)
        _last_a=a.numpy()[0].astype(np.float32)
    return _to_ctrl(_last_a)
# === Q1-C4A4 CANDIDATE FULL-STATE CHECKPOINT INTEGRATION ===
# Candidate-only integration. Canonical adoption is intentionally deferred.
import copy as _c4a4_copy
import hashlib as _c4a4_hashlib
import json as _c4a4_json
import os as _c4a4_os
import pickle as _c4a4_pickle
import random as _c4a4_random
import shutil as _c4a4_shutil
import sys as _c4a4_sys
import uuid as _c4a4_uuid

_C4A4_SCHEMA = "Q1-C4A4-FULL-STATE-V2"
_C4A4_CHECKPOINT_API = "PORTABLE_NUMERIC_TENSOR_STATE_MANIFEST_V2"
_C4A4_JSON_STRATEGY = "RECURSIVE_NUMPY_SCALAR_TO_NATIVE_V1"
# Capture optional pre-integration persistence hooks without assuming source shape.
# Some active lineages legitimately have _save but no _load; candidate import must still succeed.
_C4A4_LEGACY_SAVE = globals().get("_save")
_C4A4_LEGACY_LOAD = globals().get("_load")
_C4A4_LEGACY_INITIALIZE = globals().get("initialize_deeplearning")
_C4A4_LEGACY_CAPABILITIES = {
    "legacy_save_present": callable(_C4A4_LEGACY_SAVE),
    "legacy_load_present": callable(_C4A4_LEGACY_LOAD),
    "legacy_initialize_present": callable(_C4A4_LEGACY_INITIALIZE),
}
_C4A4_INITIALIZATION_SEED = None
_C4A4_ALLOW_LEGACY_LOAD = False
_C4A4_PYTHON_RNG = None
_C4A4_REPLAY_RNG = None
_C4A4_TF_RNG = None
_C4A4_RESTORE_IN_PROGRESS = False
_C4A4_CURRICULUM_STATE = {
    "curriculum_phase": 1,
    "next_episode": 0,
    "transitions_in_phase": 0,
    "phase_transition_count": 0,
    "development_evaluation_count": 0,
}


def _c4a4_native(value):
    if isinstance(value, dict):
        return {str(k): _c4a4_native(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_c4a4_native(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _c4a4_sha256_file(path):
    h = _c4a4_hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _c4a4_rebind_model_directory(path):
    global _DIR, _MODEL, _LOG
    global _MODEL_ACTOR, _MODEL_Q1, _MODEL_Q2, _MODEL_Q1T, _MODEL_Q2T
    global _MODEL_LOG_ALPHA, _MODEL_BUF, _MODEL_STEPS, _MODEL_UPD
    _DIR = Path(path).resolve()
    _MODEL = _DIR / "sac_ai"
    _LOG = _DIR / "train.log"
    _MODEL_ACTOR = _DIR / "actor_weights.pkl"
    _MODEL_Q1 = _DIR / "q1_weights.pkl"
    _MODEL_Q2 = _DIR / "q2_weights.pkl"
    _MODEL_Q1T = _DIR / "q1t_weights.pkl"
    _MODEL_Q2T = _DIR / "q2t_weights.pkl"
    _MODEL_LOG_ALPHA = _DIR / "log_alpha.pkl"
    _MODEL_BUF = _DIR / "replay_buffer.pkl"
    _MODEL_STEPS = _DIR / "steps.pkl"
    _MODEL_UPD = _DIR / "updates.pkl"


def configure_full_state_training(initialization_seed, model_dir=None, curriculum_state=None, allow_legacy_load=False):
    global _C4A4_INITIALIZATION_SEED, _C4A4_ALLOW_LEGACY_LOAD
    global _C4A4_PYTHON_RNG, _C4A4_REPLAY_RNG, _C4A4_TF_RNG, _C4A4_CURRICULUM_STATE
    global _actor, _q1, _q2, _q1t, _q2t, _ao, _q1o, _q2o, _alo, _log_alpha
    global _s, _a, _r, _s2, _d, _ptr, _sz, _steps, _upd
    global _started, _last_s, _last_a, _hold, _racc, _rhist
    if _actor is not None:
        raise RuntimeError("configure_full_state_training must run before initialization")
    seed = int(initialization_seed)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass
    _C4A4_INITIALIZATION_SEED = seed
    _C4A4_ALLOW_LEGACY_LOAD = bool(allow_legacy_load)
    _C4A4_PYTHON_RNG = _c4a4_random.Random(seed + 404)
    _C4A4_REPLAY_RNG = np.random.default_rng(seed + 202)
    _C4A4_TF_RNG = tf.random.Generator.from_seed(seed + 505)
    if curriculum_state is not None:
        _C4A4_CURRICULUM_STATE = _c4a4_copy.deepcopy(dict(curriculum_state))
    else:
        _C4A4_CURRICULUM_STATE = {
            "curriculum_phase": 1,
            "next_episode": 0,
            "transitions_in_phase": 0,
            "phase_transition_count": 0,
            "development_evaluation_count": 0,
        }
    if model_dir is not None:
        _c4a4_rebind_model_directory(model_dir)
    _actor = _q1 = _q2 = _q1t = _q2t = None
    _ao = _q1o = _q2o = _alo = None
    _log_alpha = None
    _s = _a = _r = _s2 = _d = None
    _ptr = _sz = _steps = _upd = 0
    _started = False
    _last_s = None
    _last_a = None
    _hold = 0
    _racc = 0.0
    _rhist = []


def set_curriculum_state(state):
    global _C4A4_CURRICULUM_STATE
    _C4A4_CURRICULUM_STATE = _c4a4_copy.deepcopy(dict(state))


def get_curriculum_state():
    return _c4a4_copy.deepcopy(_C4A4_CURRICULUM_STATE)


def _c4a4_optimizer_variables(opt):
    values = opt.variables
    if callable(values):
        values = values()
    return list(values)


def _c4a4_build_optimizer_slots(opt, variables):
    build = getattr(opt, "build", None)
    if callable(build):
        build(variables)
        return
    snapshots = [v.numpy().copy() for v in variables]
    opt.apply_gradients([(tf.zeros_like(v), v) for v in variables])
    for variable, snapshot in zip(variables, snapshots):
        variable.assign(snapshot)
    opt.iterations.assign(0)


def _pi(mu, ls, det):
    ls = tf.clip_by_value(ls, -5.0, 2.0)
    if det:
        action = tf.tanh(mu)
        log_prob = tf.zeros((tf.shape(mu)[0],), tf.float32)
        return action, log_prob
    std = tf.exp(ls)
    if _C4A4_TF_RNG is None:
        eps = tf.random.normal(tf.shape(mu))
    else:
        eps = _C4A4_TF_RNG.normal(tf.shape(mu), dtype=tf.float32)
    u = mu + std * eps
    action = tf.tanh(u)
    log_prob = -0.5 * tf.reduce_sum(
        ((u - mu) / (std + 1e-6)) ** 2 + 2.0 * ls + math.log(2.0 * math.pi), axis=1
    )
    log_prob -= tf.reduce_sum(tf.math.log(1.0 - action * action + 1e-6), axis=1)
    return action, log_prob


def _batch():
    if _C4A4_REPLAY_RNG is None:
        idx = np.random.randint(0, _sz, size=BATCH)
    else:
        idx = _C4A4_REPLAY_RNG.integers(0, _sz, size=BATCH)
    return (
        tf.convert_to_tensor(_s[idx], dtype=tf.float32),
        tf.convert_to_tensor(_a[idx], dtype=tf.float32),
        tf.convert_to_tensor(_r[idx], dtype=tf.float32),
        tf.convert_to_tensor(_s2[idx], dtype=tf.float32),
        tf.convert_to_tensor(_d[idx], dtype=tf.float32),
    )


def initialize_deeplearning():
    global PHASE, _actor, _q1, _q2, _q1t, _q2t, _ao, _q1o, _q2o, _alo, _log_alpha
    global _s, _a, _r, _s2, _d
    PHASE = max(1, min(3, int(REWARD_PHASE)))
    if _actor is not None:
        return
    full_state_dir = _c4a4_os.environ.get("JET_FULL_STATE_DIR")
    if full_state_dir and Path(full_state_dir).is_dir() and not _C4A4_RESTORE_IN_PROGRESS:
        return load_full_training_state(full_state_dir, model_dir=_DIR)
    if _C4A4_INITIALIZATION_SEED is None:
        configure_full_state_training(int(_c4a4_os.environ.get("JET_INITIALIZATION_SEED", "0")), model_dir=_DIR)
    _DIR.mkdir(parents=True, exist_ok=True)
    _actor = _build_actor()
    _q1 = _build_q()
    _q2 = _build_q()
    _q1t = _build_q()
    _q2t = _build_q()
    _log_alpha = tf.Variable(0.0, dtype=tf.float32, name="log_alpha")
    _ao = tf.keras.optimizers.Adam(LR, name="actor_adam")
    _q1o = tf.keras.optimizers.Adam(LR, name="q1_adam")
    _q2o = tf.keras.optimizers.Adam(LR, name="q2_adam")
    _alo = tf.keras.optimizers.Adam(LR, name="alpha_adam")
    ds = tf.zeros((1, SD), tf.float32)
    da = tf.zeros((1, AD), tf.float32)
    _actor(ds)
    _q1([ds, da])
    _q2([ds, da])
    _q1t([ds, da])
    _q2t([ds, da])
    _q1t.set_weights(_q1.get_weights())
    _q2t.set_weights(_q2.get_weights())
    _c4a4_build_optimizer_slots(_ao, _actor.trainable_variables)
    _c4a4_build_optimizer_slots(_q1o, _q1.trainable_variables)
    _c4a4_build_optimizer_slots(_q2o, _q2.trainable_variables)
    _c4a4_build_optimizer_slots(_alo, [_log_alpha])
    _s = np.zeros((CAP, SD), dtype=np.float32)
    _a = np.zeros((CAP, AD), dtype=np.float32)
    _r = np.zeros((CAP,), dtype=np.float32)
    _s2 = np.zeros((CAP, SD), dtype=np.float32)
    _d = np.zeros((CAP,), dtype=np.float32)
    if _C4A4_ALLOW_LEGACY_LOAD:
        if not callable(_C4A4_LEGACY_LOAD):
            raise RuntimeError(
                "Legacy checkpoint load was explicitly requested, but the active source "
                "does not define a callable _load hook"
            )
        _C4A4_LEGACY_LOAD()


def _c4a4_tensor_state_groups():
    """Return the exact ordered TensorFlow state groups persisted by C4A4.

    This intentionally avoids TensorFlow SaveV2/checkpoint graph serialization.
    Every variable is persisted as a numeric .npy array and restored only after
    exact group, count, index, shape, dtype and SHA256 validation.
    """
    return {
        "actor": list(_actor.variables),
        "q1": list(_q1.variables),
        "q2": list(_q2.variables),
        "q1_target": list(_q1t.variables),
        "q2_target": list(_q2t.variables),
        "actor_optimizer": _c4a4_optimizer_variables(_ao),
        "q1_optimizer": _c4a4_optimizer_variables(_q1o),
        "q2_optimizer": _c4a4_optimizer_variables(_q2o),
        "alpha_optimizer": _c4a4_optimizer_variables(_alo),
        "log_alpha": [_log_alpha],
        "tensorflow_rng": [_C4A4_TF_RNG.state],
    }


def _c4a4_safe_component_name(name):
    text = str(name)
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in text)


def _c4a4_write_tensor_state(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    variable_entries = []
    group_entries = []
    groups = _c4a4_tensor_state_groups()
    for group_name in sorted(groups):
        variables = groups[group_name]
        group_entries.append({"group": str(group_name), "variable_count": int(len(variables))})
        group_dir = root / _c4a4_safe_component_name(group_name)
        group_dir.mkdir(parents=True, exist_ok=False)
        for index, variable in enumerate(variables):
            array = np.asarray(variable.numpy())
            file_name = f"{index:04d}.npy"
            path = group_dir / file_name
            # allow_pickle=False is mandatory: only numeric tensor payloads are permitted.
            with path.open("wb") as handle:
                np.save(handle, array, allow_pickle=False)
            variable_entries.append({
                "group": str(group_name),
                "index": int(index),
                "file": str(path.relative_to(root)).replace("\\", "/"),
                "shape": [int(v) for v in array.shape],
                "dtype": str(array.dtype),
                "bytes": int(path.stat().st_size),
                "sha256": _c4a4_sha256_file(path),
            })
    return {
        "schema": "PORTABLE_NUMERIC_TENSOR_MANIFEST_V2",
        "groups": group_entries,
        "variables": variable_entries,
    }


def _c4a4_variable_numpy_dtype(variable):
    """Resolve a NumPy dtype across tf.Variable and Keras 3 Variable APIs.

    TensorFlow variables commonly expose ``variable.dtype`` as ``tf.DType``,
    while Keras 3 backend variables may expose the same property as a string
    such as ``"float32"``.  Checkpoint restore must accept both without
    weakening exact dtype validation.
    """
    dtype_value = getattr(variable, "dtype", None)
    if dtype_value is not None:
        numpy_dtype = getattr(dtype_value, "as_numpy_dtype", None)
        if numpy_dtype is not None:
            try:
                return np.dtype(numpy_dtype)
            except (TypeError, ValueError):
                pass
        try:
            return np.dtype(dtype_value)
        except (TypeError, ValueError):
            pass
        dtype_name = getattr(dtype_value, "name", None)
        if dtype_name is not None:
            try:
                return np.dtype(str(dtype_name))
            except (TypeError, ValueError):
                pass
    # Final compatibility fallback uses the actual numeric value, never an
    # object/pickled representation.  This preserves strict exact comparison.
    return np.asarray(variable.numpy()).dtype


def _c4a4_restore_tensor_state(root, manifest):
    root = Path(root)
    if manifest.get("schema") != "PORTABLE_NUMERIC_TENSOR_MANIFEST_V2":
        raise RuntimeError("Unsupported tensor-state manifest schema")
    expected_groups = _c4a4_tensor_state_groups()
    saved_group_counts = {
        str(entry["group"]): int(entry["variable_count"]) for entry in manifest.get("groups", [])
    }
    by_group = {name: [] for name in saved_group_counts}
    for entry in manifest.get("variables", []):
        group = str(entry["group"])
        if group not in by_group:
            raise RuntimeError("Tensor-state variable references undeclared group: " + group)
        by_group[group].append(entry)
    if set(saved_group_counts) != set(expected_groups):
        raise RuntimeError(
            "Tensor-state group mismatch: saved=" + repr(sorted(saved_group_counts))
            + " current=" + repr(sorted(expected_groups))
        )
    restored = 0
    for group_name in sorted(expected_groups):
        variables = expected_groups[group_name]
        entries = sorted(by_group[group_name], key=lambda item: int(item["index"]))
        if int(saved_group_counts[group_name]) != len(entries):
            raise RuntimeError("Tensor-state declared/observed count mismatch for " + group_name)
        if len(entries) != len(variables):
            raise RuntimeError(
                f"Tensor-state variable-count mismatch for {group_name}: "
                f"saved={len(entries)} current={len(variables)}"
            )
        if [int(e["index"]) for e in entries] != list(range(len(variables))):
            raise RuntimeError("Tensor-state indices are not contiguous for " + group_name)
        for variable, entry in zip(variables, entries):
            path = root / str(entry["file"])
            if not path.is_file():
                raise RuntimeError("Tensor-state file missing: " + str(path))
            if _c4a4_sha256_file(path) != str(entry["sha256"]):
                raise RuntimeError("Tensor-state SHA256 mismatch: " + str(path))
            with path.open("rb") as handle:
                array = np.load(handle, allow_pickle=False)
            expected_shape = tuple(int(v) for v in variable.shape)
            expected_dtype = _c4a4_variable_numpy_dtype(variable)
            if tuple(array.shape) != expected_shape:
                raise RuntimeError(
                    f"Tensor-state shape mismatch for {group_name}[{entry['index']}]: "
                    f"saved={tuple(array.shape)} current={expected_shape}"
                )
            if np.dtype(array.dtype) != expected_dtype:
                raise RuntimeError(
                    f"Tensor-state dtype mismatch for {group_name}[{entry['index']}]: "
                    f"saved={array.dtype} current={expected_dtype}"
                )
            if [int(v) for v in array.shape] != [int(v) for v in entry["shape"]]:
                raise RuntimeError("Tensor-state manifest shape mismatch: " + str(path))
            if str(array.dtype) != str(entry["dtype"]):
                raise RuntimeError("Tensor-state manifest dtype mismatch: " + str(path))
            variable.assign(array)
            assigned = np.asarray(variable.numpy())
            if assigned.shape != array.shape or assigned.dtype != array.dtype:
                raise RuntimeError("Tensor-state post-assign metadata mismatch: " + str(path))
            if not np.array_equal(assigned, array):
                raise RuntimeError("Tensor-state post-assign exact-value mismatch: " + str(path))
            restored += 1
    return {
        "tensor_group_count": int(len(expected_groups)),
        "tensor_variable_count": int(restored),
        "strict_group_count_shape_dtype_sha256_assign_pass": True,
    }


def _c4a4_find_scenario_module():
    for name, module in list(_c4a4_sys.modules.items()):
        if name.endswith("sim_presets") and hasattr(module, "get_scenario_rng_state"):
            return module
    return None


def _c4a4_scenario_state_or_none():
    module = _c4a4_find_scenario_module()
    if module is None:
        return None
    return module.get_scenario_rng_state()


def _c4a4_apply_scenario_state(state):
    if state is None:
        return False
    module = _c4a4_find_scenario_module()
    if module is None:
        return False
    module.set_scenario_rng_state(state)
    return True


def save_full_training_state(checkpoint_dir, scenario_rng_state=None, curriculum_state=None):
    global _C4A4_CURRICULUM_STATE
    if _actor is None:
        raise RuntimeError("Deep-learning state is not initialized")
    target = Path(checkpoint_dir).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.parent / ("." + target.name + ".tmp-" + _c4a4_uuid.uuid4().hex)
    backup = target.parent / ("." + target.name + ".old-" + _c4a4_uuid.uuid4().hex)
    if temp.exists():
        _c4a4_shutil.rmtree(temp)
    temp.mkdir(parents=True)
    tensor_manifest = _c4a4_write_tensor_state(temp / "tensor_state")
    (temp / "tensor_manifest.json").write_text(
        _c4a4_json.dumps(_c4a4_native(tensor_manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    if curriculum_state is not None:
        _C4A4_CURRICULUM_STATE = _c4a4_copy.deepcopy(dict(curriculum_state))
    if scenario_rng_state is None:
        scenario_rng_state = _c4a4_scenario_state_or_none()
    python_state = {
        "schema": _C4A4_SCHEMA,
        "initialization_seed": int(_C4A4_INITIALIZATION_SEED),
        "replay": {"s": _s, "a": _a, "r": _r, "s2": _s2, "d": _d, "ptr": int(_ptr), "sz": int(_sz)},
        "steps": int(_steps),
        "updates": int(_upd),
        "reward_history": list(_rhist),
        "lifecycle": {
            "started": bool(_started),
            "last_s": None if _last_s is None else np.asarray(_last_s).copy(),
            "last_a": None if _last_a is None else np.asarray(_last_a).copy(),
            "hold": int(_hold),
            "reward_accumulator": float(_racc),
        },
        "python_rng_state": _C4A4_PYTHON_RNG.getstate(),
        "replay_rng_state": _c4a4_copy.deepcopy(_C4A4_REPLAY_RNG.bit_generator.state),
        "scenario_rng_state": _c4a4_copy.deepcopy(scenario_rng_state),
        "curriculum_state": _c4a4_copy.deepcopy(_C4A4_CURRICULUM_STATE),
    }
    with (temp / "python_state.pkl").open("wb") as handle:
        _c4a4_pickle.dump(python_state, handle, protocol=4)
    contract = {
        "schema": _C4A4_SCHEMA,
        "checkpoint_api_strategy": _C4A4_CHECKPOINT_API,
        "json_serialization_strategy": _C4A4_JSON_STRATEGY,
        "tensor_restore_contract": "EXACT_GROUP_COUNT_INDEX_SHAPE_DTYPE_SHA256_ASSIGN_KERAS3_COMPAT_V3",
        "dtype_resolution_strategy": "TF_DTYPE_OR_KERAS3_STRING_TO_NUMPY_DTYPE_WITH_VALUE_FALLBACK_V1",
        "source_sha256": _c4a4_sha256_file(Path(__file__).resolve()),
        "state_dimension": int(SD),
        "action_dimension": int(AD),
        "replay_capacity": int(CAP),
        "batch_size": int(BATCH),
        "save_counter_not_applicable": True,
        "tensor_group_count": int(len(tensor_manifest["groups"])),
        "tensor_variable_count": int(len(tensor_manifest["variables"])),
    }
    (temp / "contract.json").write_text(
        _c4a4_json.dumps(_c4a4_native(contract), indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = []
    for path in sorted(p for p in temp.rglob("*") if p.is_file() and p.name != "manifest.json"):
        manifest.append({
            "file": str(path.relative_to(temp)).replace("\\", "/"),
            "bytes": int(path.stat().st_size),
            "sha256": _c4a4_sha256_file(path),
        })
    (temp / "manifest.json").write_text(
        _c4a4_json.dumps(_c4a4_native(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    moved_old = False
    try:
        if target.exists():
            _c4a4_os.replace(target, backup)
            moved_old = True
        _c4a4_os.replace(temp, target)
        if moved_old and backup.exists():
            _c4a4_shutil.rmtree(backup)
    except Exception:
        if target.exists() and not temp.exists():
            _c4a4_shutil.rmtree(target, ignore_errors=True)
        if moved_old and backup.exists() and not target.exists():
            _c4a4_os.replace(backup, target)
        raise
    return {
        "checkpoint_dir": str(target),
        "checkpoint_api_strategy": _C4A4_CHECKPOINT_API,
        "portable_numeric_tensor_state_written": True,
        "save_counter_not_applicable": True,
        "atomic_directory_replace_pass": True,
        "manifest_file_count": int(len(manifest)),
        "tensor_group_count": int(contract["tensor_group_count"]),
        "tensor_variable_count": int(contract["tensor_variable_count"]),
    }


def load_full_training_state(checkpoint_dir, model_dir=None):
    global _ptr, _sz, _steps, _upd, _rhist
    global _started, _last_s, _last_a, _hold, _racc, _C4A4_CURRICULUM_STATE
    global _C4A4_RESTORE_IN_PROGRESS
    target = Path(checkpoint_dir).resolve()
    manifest = _c4a4_json.loads((target / "manifest.json").read_text(encoding="utf-8-sig"))
    for entry in manifest:
        path = target / entry["file"]
        if not path.is_file():
            raise RuntimeError("Checkpoint manifest file missing: " + str(path))
        if _c4a4_sha256_file(path) != entry["sha256"]:
            raise RuntimeError("Checkpoint manifest hash mismatch: " + str(path))
    contract = _c4a4_json.loads((target / "contract.json").read_text(encoding="utf-8-sig"))
    if contract.get("schema") != _C4A4_SCHEMA:
        raise RuntimeError("Unsupported full-state schema")
    if contract.get("checkpoint_api_strategy") != _C4A4_CHECKPOINT_API:
        raise RuntimeError("Unsupported checkpoint API strategy")
    if contract.get("source_sha256") != _c4a4_sha256_file(Path(__file__).resolve()):
        raise RuntimeError("Candidate source hash mismatch during restore")
    with (target / "python_state.pkl").open("rb") as handle:
        state = _c4a4_pickle.load(handle)
    if int(contract["replay_capacity"]) != int(CAP):
        raise RuntimeError("Replay capacity mismatch during restore")
    _C4A4_RESTORE_IN_PROGRESS = True
    try:
        configure_full_state_training(
            int(state["initialization_seed"]), model_dir=model_dir if model_dir is not None else _DIR,
            curriculum_state=state["curriculum_state"], allow_legacy_load=False,
        )
        initialize_deeplearning()
    finally:
        _C4A4_RESTORE_IN_PROGRESS = False
    tensor_manifest = _c4a4_json.loads((target / "tensor_manifest.json").read_text(encoding="utf-8-sig"))
    restore_contract = _c4a4_restore_tensor_state(target / "tensor_state", tensor_manifest)
    if int(contract["tensor_group_count"]) != int(restore_contract["tensor_group_count"]):
        raise RuntimeError("Tensor group count does not match checkpoint contract")
    if int(contract["tensor_variable_count"]) != int(restore_contract["tensor_variable_count"]):
        raise RuntimeError("Tensor variable count does not match checkpoint contract")
    replay = state["replay"]
    for key, destination in (("s", _s), ("a", _a), ("r", _r), ("s2", _s2), ("d", _d)):
        source = np.asarray(replay[key])
        if source.shape != destination.shape or source.dtype != destination.dtype:
            raise RuntimeError("Replay shape/dtype mismatch for " + key)
        destination[:] = source
        if not np.array_equal(destination, source):
            raise RuntimeError("Replay exact restore mismatch for " + key)
    _ptr = int(replay["ptr"]) % CAP
    _sz = int(replay["sz"])
    _steps = int(state["steps"])
    _upd = int(state["updates"])
    _rhist = [float(v) for v in state["reward_history"]]
    lifecycle = state["lifecycle"]
    _started = bool(lifecycle["started"])
    _last_s = None if lifecycle["last_s"] is None else np.asarray(lifecycle["last_s"], dtype=np.float32)
    _last_a = None if lifecycle["last_a"] is None else np.asarray(lifecycle["last_a"], dtype=np.float32)
    _hold = int(lifecycle["hold"])
    _racc = float(lifecycle["reward_accumulator"])
    _C4A4_PYTHON_RNG.setstate(state["python_rng_state"])
    _C4A4_REPLAY_RNG.bit_generator.state = _c4a4_copy.deepcopy(state["replay_rng_state"])
    _C4A4_CURRICULUM_STATE = _c4a4_copy.deepcopy(state["curriculum_state"])
    scenario_applied = _c4a4_apply_scenario_state(state.get("scenario_rng_state"))
    return {
        "checkpoint_manifest_verified": True,
        "checkpoint_restore_tensor_manifest_strict": True,
        "tensor_state_exact_reloaded": True,
        "save_counter_not_applicable": True,
        "scenario_rng_state": _c4a4_copy.deepcopy(state.get("scenario_rng_state")),
        "scenario_rng_state_applied": bool(scenario_applied),
        "checkpoint_api_strategy": _C4A4_CHECKPOINT_API,
        "tensor_group_count": int(restore_contract["tensor_group_count"]),
        "tensor_variable_count": int(restore_contract["tensor_variable_count"]),
    }


def _save():
    full_state_dir = _c4a4_os.environ.get("JET_FULL_STATE_DIR")
    if full_state_dir:
        return save_full_training_state(full_state_dir)
    if callable(_C4A4_LEGACY_SAVE):
        return _C4A4_LEGACY_SAVE()
    raise RuntimeError(
        "No full-state checkpoint directory is configured and the active source "
        "does not define a callable legacy _save hook"
    )


def _load():
    full_state_dir = _c4a4_os.environ.get("JET_FULL_STATE_DIR")
    if full_state_dir and Path(full_state_dir).is_dir():
        return load_full_training_state(full_state_dir, model_dir=_DIR)
    if _C4A4_ALLOW_LEGACY_LOAD:
        if not callable(_C4A4_LEGACY_LOAD):
            raise RuntimeError(
                "Legacy checkpoint load was explicitly requested, but the active source "
                "does not define a callable _load hook"
            )
        return _C4A4_LEGACY_LOAD()
    return None

# === END Q1-C4A4 CANDIDATE FULL-STATE CHECKPOINT INTEGRATION ===
