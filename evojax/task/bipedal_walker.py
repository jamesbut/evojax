from typing import Tuple

from evojax.task.base import VectorizedTask, TaskState

import jax
import jax.numpy as jnp
from flax.struct import dataclass


@dataclass
class State(TaskState):
    obs: jnp.ndarray


def update_state(action: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
    return State([0.0] * 24)


class BipedalWalker(VectorizedTask):

    def __init__(self):

        print('Init BPW')

        self.max_steps = 100
        self.obs_shape = tuple([24,])
        self.act_shape = tuple([4,])

        def reset_fn(key):
            return State(jnp.array([0.] * 24))
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(action, state):
            cur_state = update_state(action, state.obs)
            return State(cur_state)
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.array) -> State:
        return self._reset_fn(key)

    def step(self,
             state: State,
             action: jnp.ndarray) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(action, state)
