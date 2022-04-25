# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Forward Flux Sampling (FFS).

Implementation of the direct version of the FFS algorithm.
FFS uses a series of nonintersecting interfaces between the initial and 
the final states. The initial and final states are defined in terms of 
an order parameter. The method allows to calculate rate constants and 
generate transition paths.
"""

from typing import Callable, Mapping, NamedTuple, Optional
from warnings import warn

from jax import numpy as np

from pysages.backends import ContextWrapper
from pysages.methods.core import SamplingMethod, generalize
from pysages.utils import JaxArray, copy

import sys


class FFSState(NamedTuple):
    bias: JaxArray
    xi: JaxArray

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class FFS(SamplingMethod):
    """
    Constructor of the Forward Flux Sampling method.

    Parameters
    ----------
    self : FFS
        See parent class
    cvs:
        See parent class
    args:
        See parent class
    kwargs:
        See parent class

    Attributes
    ----------
    snapshot_flags:
        Indicate the particle properties required from a snapshot.
    """

    snapshot_flags = {"positions", "indices"}

    def build(self, snapshot, helpers):
        self.helpers = helpers
        return _ffs(self, snapshot, helpers)

    # We override the default run method as FFS is algorithmically fairly different
    def run(
        self,
        context_generator: Callable,
        sampling_steps_win: int,
        dt: float,
        win_i: float,
        win_l: float,
        Nw: int,
        Nmax_replicas: int,
        basin_sampling: bool = True,
        sampling_steps_basin: int = None,
        ini_snaps: list = None,
        write_snaps: bool = True,
        write_traj: bool = True,
        verbose: bool = False,
        callback: Optional[Callable] = None,
        context_args: Mapping = dict(),
        **kwargs,
    ):
        """
        Direct version of the Forward Flux Sampling algorithm
        [Phys. Rev. Lett. 94, 018104 (2005), https://doi.org/10.1103/PhysRevLett.94.018104;
        J. Chem. Phys. 124, 024102 (2006), https://doi.org/10.1063/1.2140273].

        Arguments
        ---------
        context_generator: Callable
            User defined function that sets up a simulation context with the backend.
            Must return an instance of `hoomd.context.SimulationContext` for HOOMD-blue
            and `simtk.openmm.Simulation` for OpenMM. The function gets `context_args`
            unpacked for additional user arguments.

        timesteps: int
            Number of timesteps the simulation is running.

        dt: float
            Timestep of the simulation.

        win_i: float
            Initial window for the system.

        win_l: float
            Last window to be calculated in FFS.

        Nw: int
            Number of equally spaced windows.

        sampling_steps_basin: int
            Period for sampling configurations in the basin.

        Nmax_replicas: int
            Number of stored configurations for each window.

        verbose: bool
            If True more information will be logged (useful for debbuging).

        callback: Optional[Callable]
            Allows for user defined actions into the simulation workflow of the method.
            `kwargs` gets passed to the backend `run` function.

        NOTE:
            The current implementation runs a single simulation/replica,
            but multiple concurrent simulations can be scripted on top of this.
        """

        context = context_generator(**context_args)
        self.context = ContextWrapper(context, self, callback)

        with self.context:
            sampler = self.context.sampler
            xi = sampler.state.xi.block_until_ready()
            windows = np.linspace(win_i, win_l, num=Nw)

            if win_i < win_l:
                increase = True
            elif win_i > win_l:
                increase = False
            else:
                raise ValueError("State A equal to state B")

            is_configuration_good = check_input(windows, xi, increase, verbose=verbose)
            if not is_configuration_good:
                raise ValueError("Bad initial configuration")

            run = self.context.run
            helpers = self.helpers
            cv = self.cv

            reference_snapshot = copy(sampler.snapshot)

            # We initially sample from basin A
            # TODO: bundle the arguments into data structures
            if basin_sampling:
                ini_snapshots, basin_steps = basin_sampling(
                    Nmax_replicas,
                    sampling_steps_basin,
                    windows,
                    run,
                    sampler,
                    reference_snapshot,
                    helpers,
                    cv,
                    increase,
                    write_traj,
                )
                if write_snaps:
                    write_snapshots('Basin', ini_snapshots)   
            else:
                if ini_snaps is None: 
                    raise ValueError("Provide initial snapshots or set Asampling to True")
                ini_snapshots = ini_snaps
                if len(ini_snapshots) != Nmax_replicas:
                    raise ValueError("Wrong number of initial configurations")
                basin_steps = 0
            
            # Calculate initial flow
            phi_a, snaps_0, flow_steps = initial_flow(
                Nmax_replicas,
                dt,
                sampling_steps_win,
                windows,
                ini_snapshots,
                run,
                sampler,
                helpers,
                cv,
                increase,
                basin_steps,
                write_traj,
            )

            write_to_file(phi_a)
            hist = np.zeros(len(windows))
            hist = hist.at[0].set(phi_a)

            if write_snaps:
                write_snapshots('initial flow', snaps_0)

            # Calculate conditional probability for each window
            for k in range(1, len(windows)):
                if k == 1:
                    old_snaps = snaps_0
                prob, w1_snapshots = running_window(
                    windows,
                    k,
                    sampling_steps_win,
                    old_snaps,
                    run,
                    sampler,
                    helpers,
                    cv,
                    increase,
                    flow_steps,
                    write_traj,
                )
                write_to_file(prob)
                hist = hist.at[k].set(prob)
                old_snaps = increase_snaps(w1_snapshots, snaps_0)
                print(f"size of snapshots: {len(old_snaps)}\n")
                if write_snaps:
                    write_snapshots('Window_' + str(k), w1_snapshots)

            K_t = np.prod(hist)
            write_to_file("# Flux Constant")
            write_to_file(K_t)


def _ffs(method, snapshot, helpers):
    """
    Internal function that generates an `initialize` and an `update` functions.
    `initialize` is ran once just before the time integration starts and `update`
    is called after each simulation timestep.

    Arguments
    ---------
    method: FFS

    snapshot:
        PySAGES snapshot of the simulation (backend dependent).

    helpers
        Helper function bundle as generated by
        `SamplingMethod.context.get_backend().build_helpers`.

    Returns
    -------
    Tuple `(snapshot, initialize, update)` as described above.
    """
    cv = method.cv
    dt = snapshot.dt
    natoms = np.size(snapshot.positions, 0)

    # initialize method
    def initialize():
        bias = np.zeros((natoms, 3))
        xi, _ = cv(helpers.query(snapshot))
        return FFSState(bias, xi)

    def update(state, data):
        xi, _ = cv(data)
        bias = state.bias
        return FFSState(bias, xi)

    return snapshot, initialize, generalize(update, helpers)


def write_to_file(value):
    with open("ffs_results.dat", "a+") as f:
        f.write(str(value) + "\n")

#to do - write window, initial step and final step, success or fail
def write_trajectory_info(id, stage, step_ini, step_end, status):
    with open("trajectories.dat", "a+") as f:
        f.write(str(id) + "\t" + stage + "\t" + str(step_ini) + "\t" + str(step_end) + "\t" + status + "\n")

#to do - write window, initial step and final step, success or fail
def write_snapshots(stage, id, snapshots):
    file = stage + '.npy'
    with open(file, "wb") as f:
        np.save(f, snapshots)
        
# Since snapshots are depleted each window, this function restores the list to
# its initial values. This only works with stochastic integrators like BD or
# Langevin, for other, velocity resampling is needed
def increase_snaps(windows, initial_w):
    if len(windows) > 0:
        ratio = len(initial_w) // len(windows)
        windows = windows * ratio

    return windows


def check_input(grid, xi, increase, verbose=False):
    """
    Verify whether the initial configuration is a good one.
    """
    if increase:
        is_good = xi < grid[0]
    else:
        is_good = xi > grid[0]

    if is_good:
        print("Good initial configuration\n")
        print(xi)
    elif verbose:
        print(xi)

    return is_good


def basin_sampling(
    max_num_snapshots, sampling_time, grid, run, sampler, reference_snapshot, helpers, cv, increase, write_traj
):
    """
    Sampling of basing configurations for initial flux calculations.
    """
    basin_snapshots = []
    win_A = grid[0]
    xi = sampler.state.xi.block_until_ready()
    total_steps = 0

    print("Starting basin sampling\n")
    while len(basin_snapshots) < int(max_num_snapshots):
        run(sampling_time)
        total_steps += sampling_time
        xi = sampler.state.xi.block_until_ready()

        if increase:
            if np.all(xi < win_A):
                snap = copy(sampler.snapshot)
                basin_snapshots.append(snap)
                print("Storing basing configuration with cv value:\n")
                print(xi)
            else:
                helpers.restore(sampler.snapshot, reference_snapshot)
                xi, _ = cv(helpers.query(sampler.snapshot))
                print("Restoring basing configuration since system left basin with cv value:\n")
                print(xi)
        else:
            if np.all(xi > win_A):
                snap = copy(sampler.snapshot)
                basin_snapshots.append(snap)
                print("Storing basing configuration with cv value:\n")
                print(xi)
            else:
                helpers.restore(sampler.snapshot, reference_snapshot)
                xi, _ = cv(helpers.query(sampler.snapshot))
                print("Restoring basing configuration since system left basin with cv value:\n")
                print(xi)

    print(f"Finish sampling basin with {max_num_snapshots} snapshots\n")
    if write_traj:
        write_trajectory_info(0, 'Basin', 1, total_steps, 'initial sampling')
    return basin_snapshots, total_steps


def initial_flow(Num_window0, timestep, freq, grid, initial_snapshots, run, sampler, helpers, cv, increase, basin_steps, write_traj):
    """
    Selects snapshots from list generated with `basin_sampling`.
    """

    success = 0
    time_count = 0.0
    window0_snaps = []
    win_A = grid[0]
    initial_step = basin_steps
    steps_count = basin_steps

    for i in range(0, Num_window0):
        print(f"Initial stored configuration: {i}\n")
        helpers.restore(sampler.snapshot, initial_snapshots[i])
        xi, _ = cv(helpers.query(sampler.snapshot))
        print(xi)

        has_reached_A = False
        while not has_reached_A:
            # TODO: make the number of timesteps below a parameter of the method.
            run(freq)
            time_count += freq * timestep
            steps_count += freq
            xi = sampler.state.xi.block_until_ready()
            
            if increase:
                if np.all(xi >= win_A) and np.all(xi < grid[1]):
                    success += 1
                    has_reached_A = True
                    
                    if len(window0_snaps) <= Num_window0:
                        snap = copy(sampler.snapshot)
                        window0_snaps.append(snap)
                    if write_traj:
                        write_trajectory_info(i + 1, 'initial_flow', initial_step, steps_count, 'success')
                    initial_step = steps_count

                    break
            else:
                if np.all(xi <= win_A) and np.all(xi > grid[1]):
                    success += 1
                    has_reached_A = True

                    if len(window0_snaps) <= Num_window0:
                        snap = copy(sampler.snapshot)
                        window0_snaps.append(snap)
                    if write_traj:
                        write_trajectory_info(i + 1, 'initial_flow', initial_step, steps_count, 'success')
                    initial_step = steps_count

                    break

    print(f"Finish Initial Flow with {success} succeses over {time_count} time\n")
    phi_a = float(success) / (time_count)

    return phi_a, window0_snaps, steps_count


def running_window(grid, step, freq, old_snapshots, run, sampler, helpers, cv, increase, flow_steps, write_traj):
    success = 0
    new_snapshots = []
    win_A = grid[0]
    win_value = grid[int(step)]
    stage = 'Window_' + str(win_value)
    has_conf_stored = False
    initial_step = flow_steps
    steps_count = flow_steps

    for i in range(0, len(old_snapshots)):
        helpers.restore(sampler.snapshot, old_snapshots[i])
        xi, _ = cv(helpers.query(sampler.snapshot))
        print(f"Stored configuration: {i} of window: {step}\n")
        print(xi)

        # limit running time to avoid zombie trajectories
        # this can be probably be improved
        running = True
        while running:
            run(freq)
            steps_count += freq
            xi = sampler.state.xi.block_until_ready()

            if increase:
                if np.all(xi < win_A):
                    running = False
                    if write_traj:
                        write_trajectory_info(i + 1, stage, initial_step, steps_count, 'fail')
                    initial_step = steps_count
                elif np.all(xi >= win_value):
                    snap = copy(sampler.snapshot)
                    new_snapshots.append(snap)
                    success += 1
                    running = False
                    if not has_conf_stored:
                        has_conf_stored = True
                    if write_traj:
                        write_trajectory_info(i + 1, stage, initial_step, steps_count, 'success')
                    initial_step = steps_count
            else:
                if np.all(xi > win_A):
                    running = False
                    if write_traj:
                        write_trajectory_info(i + 1, stage, initial_step, steps_count, 'fail')
                    initial_step = steps_count
                elif np.all(xi <= win_value):
                    snap = copy(sampler.snapshot)
                    new_snapshots.append(snap)
                    success += 1
                    running = False
                    if not has_conf_stored:
                        has_conf_stored = True
                    if write_traj:
                        write_trajectory_info(i + 1, stage, initial_step, steps_count, 'success')
                    initial_step = steps_count

    if success == 0:
        warn(f"Unable to estimate probability, exiting early at window {step}\n")
        sys.exit(0)

    if len(new_snapshots) > 0:
        prob_local = float(success) / len(old_snapshots)
        print(f"Finish window {step} with {len(new_snapshots)} snapshots\n")
        return prob_local, new_snapshots
