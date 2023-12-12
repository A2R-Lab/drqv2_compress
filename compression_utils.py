import random
import numpy as np
import torch


class SparseArray:
    def __init__(self):
        self.reset()

    def reset(self):
        self._shape = None
        self._dtype = None
        self._store_raw = False
        self._arr = None
        self._inds = None
        self._vals = None

    def set(self, arr):
        self.reset()

        u, v = np.nonzero(arr)
        u = u.astype(np.uint8)
        v = v.astype(np.uint8)

        self._shape = arr.shape
        self._dtype = arr.dtype

        # don't store as sparse if it will be larger
        if len(u) > np.prod(self._shape) / 4:
            self._store_raw = True
            self._arr = arr
        else:
            self._inds = (u, v)
            self._vals = arr[self._inds]

    def nonzero_count(self):
        if self._arr is None and self._vals is None:
            return 0
        return np.prod(self._shape) / 4 if self._store_raw else len(self._vals)

    def to_numpy(self):
        if self._store_raw:
            return self._arr
        arr = np.zeros(self._shape, dtype=self._dtype)
        arr[self._inds] = self._vals
        return arr


class ObservationCompressor:
    def __init__(self, buffer_size, n_stack=3, image_shape=(84, 84)):
        self._buffer_size = buffer_size
        # self.n_envs = n_envs
        self._n_stack = n_stack
        self._image_shape = image_shape
        self._dtype = np.uint8

        self._obs_size = self._buffer_size // self._n_stack
        if self._buffer_size % self._n_stack != 0:
            self._obs_size += 1
        self._obs = np.zeros((self._obs_size,) + self._image_shape, dtype=self._dtype)
        self._sparse_obs = [
            [SparseArray() for _ in range(self._n_stack - 1)]
            for _ in range(self._obs_size)
        ]
        self._obs_inds = -1 * np.ones(
            (self._buffer_size, self._n_stack), dtype=np.int32
        )

    def get_nonzero_count(self):
        total = 0
        for stack in self._sparse_obs:
            for sa in stack:
                total += sa.nonzero_count()
        return total

    def add(self, obs, pos, episode_pos):
        assert obs.shape == (self._n_stack,) + self._image_shape
        assert 0 <= pos < self._buffer_size

        # Get positions in self.obs array and self.sparse_obs
        obs_pos = pos // self._n_stack
        sparse_obs_pos = pos % self._n_stack

        # Zero out the next self.n_stack - 1 positions to ensure
        #   future references to the current position are erased
        self._obs_inds[pos : pos + self._n_stack] = -1
        # NOTE: Not strictly necessary, as the sampling logic ensures
        #       that no overlap/overwrite will occur

        # TODO: dm_control repeats the first obs (obs_0, obs_0, obs_0).
        #       We will need to keep track of first n_stack adds
        #       (starting w/ StepType.FIRST), so we can create positions
        #       [(0, 0, 0), (0, 0, 1), (0, 1, 2), ...]

        if episode_pos < self._n_stack:
            self._obs_inds[pos] = pos - episode_pos
            for i in range(episode_pos):
                self._obs_inds[pos, -1 - i] += episode_pos - i
        else:
            for i in range(self._n_stack):
                self._obs_inds[pos, i] = pos - (self._n_stack - 1 - i)

        # Store the observation
        observation = np.array(obs[-1]).copy()

        # If sparse_obs_pos == 0, then add full ob to self.obs
        if sparse_obs_pos == 0:
            self._obs[obs_pos] = observation
        else:
            # TYPE I: Diff obs with self.obs[obs_pos]
            obs_diff = observation.astype(np.int16) - self._obs[obs_pos].astype(
                np.int16
            )
            self._sparse_obs[obs_pos][sparse_obs_pos - 1].set(obs_diff)

            # TYPE II: Diff obs with previous obs self.sparse_obs[obs_pos][sparse_obs_pos - 1]

    #             obs_prev = self.obs[obs_pos]
    #             for i in range(sparse_obs_pos - 1):
    #                 obs_prev += self.sparse_obs[obs_pos][i].to_numpy()
    #             obs_diff = observation - obs_prev
    #             obs_diff_sparse = SparseArray(obs_diff)
    #             self.sparse_obs[obs_pos][sparse_obs_pos - 1] = obs_diff_sparse

    def _get_obs(self, idx):
        # Get observation at index idx from self.obs and/or self.sparse_obs
        if idx == -1:
            return np.zeros(self._image_shape)

        assert 0 <= idx < self._buffer_size

        # Get positions in self.obs array and self.sparse_obs
        obs_idx = idx // self._n_stack
        sparse_obs_idx = idx % self._n_stack

        obs_base = self._obs[obs_idx]

        if sparse_obs_idx == 0:
            return obs_base

        # TYPE I: Diff obs with self.obs[obs_pos]
        return obs_base + self._sparse_obs[obs_idx][sparse_obs_idx - 1].to_numpy()

        # TYPE II: Diff obs with previous obs self.sparse_obs[obs_pos][sparse_obs_pos - 1]

    #         for i in range(sparse_obs_idx - 1):
    #             obs_base += self.sparse_obs[obs_idx][i].to_numpy()
    #         return obs_base

    def get(self, pos):
        assert 0 <= pos <= self._buffer_size

        obs = np.zeros(
            (self._n_stack,) + self._image_shape,
            dtype=self._dtype,
        )

        inds = self._obs_inds[pos]
        for i in range(self._n_stack):
            obs[i] = self._get_obs(inds[i]).astype(self._dtype)
        return obs


class CompressedReplayBuffer:
    def __init__(
        self,
        max_size,
        batch_size,
        frame_stack,
        nstep,
        discount,
        data_specs,
        use_sparse=True,
    ):
        self._max_size = max_size
        self._batch_size = batch_size
        self._frame_stack = frame_stack
        self._nstep = nstep
        self._discount = discount
        self._use_sparse = use_sparse

        assert len(data_specs) == 4
        (
            self._obs_spec,
            self._action_spec,
            self._reward_spec,
            self._discount_spec,
        ) = data_specs

        # self._observations = np.zeros(
        #     (self._max_size,) + self._obs_spec.shape, dtype=np.uint8
        # )
        if self._use_sparse:
            self._obs_comp = ObservationCompressor(self._max_size, self._frame_stack)
        else:
            self._observations = np.zeros((self._max_size, 84, 84), dtype=np.uint8)
            self._obs_inds = np.zeros(
                (self._max_size, self._frame_stack), dtype=np.uint32
            )

        self._actions = np.zeros(
            (self._max_size, self._action_spec.shape[0]), dtype=self._action_spec.dtype
        )
        self._rewards = np.zeros(self._max_size, dtype=self._reward_spec.dtype)
        self._discounts = np.zeros(self._max_size, dtype=self._discount_spec.dtype)

        self._pos = 0
        self._full = False

        # Sampling-specific variables
        self._episodes = []  # (episode_start, episode_length)
        self._cur_episode_start = None

    def add(self, time_step):
        # NOTE: original paper does not store time_step.last()
        # self._observations[self._pos] = np.array(time_step.observation).copy()
        self._actions[self._pos] = np.array(time_step.action).copy()
        self._rewards[self._pos] = time_step.reward
        self._discounts[self._pos] = time_step.discount

        # Check if we are overwriting a previously stored episode
        if (
            len(self._episodes)
            and (self._pos - self._episodes[0][0]) % self._max_size
            < self._episodes[0][1]
        ):
            del self._episodes[0]

        # Check if we are at the first step of an episode
        if time_step.first():
            self._cur_episode_start = self._pos

        # TODO: Add obs_comp add here!
        episode_pos = self._pos - self._cur_episode_start
        if self._use_sparse:
            self._obs_comp.add(time_step.observation, self._pos, episode_pos)
        else:
            self._observations[self._pos : self._pos + self._frame_stack] = 0
            self._obs_inds[self._pos : self._pos + self._frame_stack] = 0
            if episode_pos < self._frame_stack:
                self._obs_inds[self._pos] = self._pos - episode_pos
                for i in range(episode_pos):
                    self._obs_inds[self._pos, -1 - i] += episode_pos - i
            else:
                for i in range(self._frame_stack):
                    self._obs_inds[self._pos, i] = self._pos - (
                        self._frame_stack - 1 - i
                    )
            self._observations[self._pos] = np.array(time_step.observation[-1]).copy()

        # Check if we are at the end of an episode and have a previously stored start
        if time_step.last() and self._cur_episode_start is not None:
            episode_length = (self._pos - self._cur_episode_start + 1) % self._max_size
            self._episodes.append((self._cur_episode_start, episode_length))
            self._cur_episode_start = None

        # Update position counter
        self._pos += 1
        if self._pos == self._max_size:
            self._full = True
            self._pos = 0

    def to_torch(self, array, copy=False):
        if copy:
            return torch.tensor(array)
        return torch.as_tensor(array)

    def __len__(self):
        return self._max_size if self._full else self._pos

    def __iter__(self):
        return self

    def __next__(self):
        if self._use_sparse:
            observations = np.zeros(
                (self._batch_size,) + self._obs_spec.shape, dtype=np.uint8
            )
            actions = np.zeros(
                (self._batch_size, self._action_spec.shape[0]),
                dtype=self._action_spec.dtype,
            )
            next_observations = np.zeros(
                (self._batch_size,) + self._obs_spec.shape, dtype=np.uint8
            )
        else:
            batch_inds = []

        rewards = np.zeros((self._batch_size, 1), dtype=self._reward_spec.dtype)
        discounts = np.zeros((self._batch_size, 1), dtype=self._discount_spec.dtype)

        assert len(self._episodes)
        for i in range(self._batch_size):
            episode_start, episode_length = random.choice(self._episodes)

            # add +1 for the first dummy transition
            idx = (
                np.random.randint(
                    episode_start, episode_start + episode_length - self._nstep - 1
                )
                + 1
            )

            if self._use_sparse:
                # obs = self._observations[(idx - 1) % self._max_size]
                obs = self._obs_comp.get((idx - 1) % self._max_size)
                action = self._actions[idx % self._max_size]
                # next_obs = self._observations[(idx + self._nstep - 1) % self._max_size]
                next_obs = self._obs_comp.get((idx + self._nstep - 1) % self._max_size)

                observations[i] = obs
                actions[i] = action
                next_observations[i] = next_obs
            else:
                batch_inds.append(idx)

            reward = 0.0
            discount = 1.0
            for j in range(self._nstep):
                step_reward = self._rewards[(idx + j) % self._max_size]
                reward += discount * step_reward
                discount *= self._discounts[(idx + j) % self._max_size] * self._discount
            rewards[i, 0] = reward
            discounts[i, 0] = discount

        if self._use_sparse:
            data = (
                observations,
                actions,
                rewards,
                discounts,
                next_observations,
            )
        else:
            batch_inds = np.array(batch_inds)
            data = (
                self._observations[self._obs_inds[(batch_inds - 1) % self._max_size]],
                self._actions[batch_inds % self._max_size],
                rewards,
                discounts,
                self._observations[
                    self._obs_inds[(batch_inds + self._nstep - 1) % self._max_size]
                ],
            )

        return tuple(map(self.to_torch, data))
