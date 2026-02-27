# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyright: reportMissingImports=false
# pyright: reportInvalidTypeForm=false

import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase
from .kernels_body import (
    eval_body_joint_forces,
)
from .kernels_contact import (
    eval_body_contact_forces,
    eval_particle_body_contact_forces,
    eval_particle_contact_forces,
    eval_triangle_contact_forces,
)
from .kernels_muscle import (
    eval_muscle_forces,
)
from .kernels_particle import (
    eval_bending_forces,
    eval_spring_forces,
    eval_tetrahedra_forces,
    eval_triangle_forces,
)


@wp.kernel
def _accum_particle_forces(dst: wp.array(dtype=wp.vec3), src: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    wp.atomic_add(dst, i, src[i])


@wp.kernel
def _accum_body_wrenches(
    dst: wp.array(dtype=wp.spatial_vector),
    src: wp.array(dtype=wp.spatial_vector),
):
    i = wp.tid()
    wp.atomic_add(dst, i, src[i])


class SolverSemiImplicit(SolverBase):
    """A semi-implicit integrator using symplectic Euler.

    After constructing `Model` and `State` objects this time-integrator
    may be used to advance the simulation state forward in time.

    Semi-implicit time integration is a variational integrator that
    preserves energy, however it not unconditionally stable, and requires a time-step
    small enough to support the required stiffness and damping forces.

    See: https://en.wikipedia.org/wiki/Semi-implicit_Euler_method

    Example
    -------

    .. code-block:: python

        solver = newton.solvers.SolverSemiImplicit(model)

        # simulation loop
        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

    """

    def __init__(
        self,
        model: Model,
        angular_damping: float = 0.05,
        friction_smoothing: float = 1.0,
        joint_attach_ke: float = 1.0e4,
        joint_attach_kd: float = 1.0e2,
        enable_tri_contact: bool = True,
    ):
        """
        Args:
            model (Model): the model to be simulated.
            angular_damping (float, optional): Angular damping factor to be used in rigid body integration. Defaults to 0.05.
            friction_smoothing (float, optional): Huber norm delta used for friction velocity normalization (see :func:`warp.math.norm_huber`). Defaults to 1.0.
            joint_attach_ke (float, optional): Joint attachment spring stiffness. Defaults to 1.0e4.
            joint_attach_kd (float, optional): Joint attachment spring damping. Defaults to 1.0e2.
            enable_tri_contact (bool, optional): Enable triangle contact. Defaults to True.
        """
        super().__init__(model=model)
        self.angular_damping = angular_damping
        self.friction_smoothing = friction_smoothing
        self.joint_attach_ke = joint_attach_ke
        self.joint_attach_kd = joint_attach_kd
        self.enable_tri_contact = enable_tri_contact

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ):
        """
        Simulate the model for a given time step using the given control input.

        Args:
            state_in: The input state.
            state_out: The output state.
            control: The control input.
                Defaults to `None` which means the control values from the
                :class:`Model` are used.
            contacts: The contact information.
                Defaults to `None` which means no contacts are used.
            dt: The time step (typically in seconds).

        .. warning::
            The ``eval_particle_contact`` kernel for particle-particle contact handling may corrupt the gradient computation
            for simulations involving particle collisions.
            To disable it, set :attr:`newton.Model.particle_grid` to `None` prior to calling :meth:`step`.
        """
        with wp.ScopedTimer("simulate", False):
            particle_f = None
            body_f = None

            if state_in.particle_count:
                particle_f = state_in.particle_f

            if state_in.body_count:
                body_f = state_in.body_f

            model = self.model

            if control is None:
                control = model.control(clone_variables=False)

            body_f_work = body_f
            if body_f is not None and model.joint_count and control.joint_f is not None:
                # Avoid accumulating joint_f into the persistent state body_f buffer.
                body_f_work = wp.clone(body_f)

            particle_f_ext = getattr(model, "particle_f_ext", None)
            if particle_f is not None and particle_f_ext is not None:
                if getattr(particle_f_ext, "shape", None) != particle_f.shape:
                    raise ValueError(
                        "model.particle_f_ext must match state_in.particle_f shape "
                        f"(got {getattr(particle_f_ext, 'shape', None)} vs {particle_f.shape})"
                    )
                wp.launch(
                    kernel=_accum_particle_forces,
                    dim=state_in.particle_count,
                    inputs=[particle_f, particle_f_ext],
                    device=particle_f.device,
                )

            body_f_ext = getattr(model, "body_f_ext", None)
            if body_f_work is not None and body_f_ext is not None:
                if getattr(body_f_ext, "shape", None) != body_f_work.shape:
                    raise ValueError(
                        "model.body_f_ext must match state_in.body_f shape "
                        f"(got {getattr(body_f_ext, 'shape', None)} vs {body_f_work.shape})"
                    )
                wp.launch(
                    kernel=_accum_body_wrenches,
                    dim=state_in.body_count,
                    inputs=[body_f_work, body_f_ext],
                    device=body_f_work.device,
                )

            # damped springs
            eval_spring_forces(model, state_in, particle_f)

            # triangle elastic and lift/drag forces
            eval_triangle_forces(model, state_in, control, particle_f)

            # triangle bending
            eval_bending_forces(model, state_in, particle_f)

            # tetrahedral FEM
            eval_tetrahedra_forces(model, state_in, control, particle_f)

            # body joints
            eval_body_joint_forces(model, state_in, control, body_f_work, self.joint_attach_ke, self.joint_attach_kd)

            # muscles
            if False:
                eval_muscle_forces(model, state_in, control, body_f)

            # particle-particle interactions
            eval_particle_contact_forces(model, state_in, particle_f)

            # triangle/triangle contacts
            if self.enable_tri_contact:
                eval_triangle_contact_forces(model, state_in, particle_f)

            # body contacts
            eval_body_contact_forces(
                model, state_in, contacts, friction_smoothing=self.friction_smoothing, body_f_out=body_f_work
            )

            # particle shape contact
            eval_particle_body_contact_forces(
                model, state_in, contacts, particle_f, body_f_work, body_f_in_world_frame=False
            )

            self.integrate_particles(model, state_in, state_out, dt)

            if body_f_work is body_f:
                self.integrate_bodies(model, state_in, state_out, dt, self.angular_damping)
            else:
                body_f_prev = state_in.body_f
                state_in.body_f = body_f_work
                self.integrate_bodies(model, state_in, state_out, dt, self.angular_damping)
                state_in.body_f = body_f_prev
