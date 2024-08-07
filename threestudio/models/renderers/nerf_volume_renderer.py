from dataclasses import dataclass, field
from functools import partial

import nerfacc
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.estimators import ImportanceEstimator
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.networks import create_network_with_input_encoding
from threestudio.models.renderers.base import VolumeRenderer
from threestudio.systems.utils import parse_optimizer, parse_scheduler_to_instance
from threestudio.utils.ops import chunk_batch, get_activation, validate_empty_rays
from threestudio.utils.typing import *



@threestudio.register("nerf-gaussian-volume-bounded-renderer")
class NeRFGaussianVolumeBoundedRenderer(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        num_samples_per_ray: int = 512
        eval_chunk_size: int = 160000
        randomized: bool = True

        near_plane: float = 0.0
        far_plane: float = 1e10

        return_comp_normal: bool = False
        return_normal_perturb: bool = False

        # in ["occgrid", "proposal", "importance"]
        estimator: str = "occgrid"

        # for occgrid
        grid_prune: bool = True
        prune_alpha_threshold: bool = True

        # for proposal
        proposal_network_config: Optional[dict] = None
        prop_optimizer_config: Optional[dict] = None
        prop_scheduler_config: Optional[dict] = None
        num_samples_per_ray_proposal: int = 64

        # for importance
        num_samples_per_ray_importance: int = 64

    cfg: Config


    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)

        self.bound = None
        if self.cfg.estimator == "occgrid":
            self.estimator = nerfacc.OccGridEstimator(
                roi_aabb=self.bbox.view(-1), resolution=32, levels=1
            )
            if not self.cfg.grid_prune:
                self.estimator.occs.fill_(True)
                self.estimator.binaries.fill_(True)
            self.render_step_size = (
                1.732 * 2 * self.cfg.radius / self.cfg.num_samples_per_ray
            )
            self.randomized = self.cfg.randomized
        elif self.cfg.estimator == "importance":
            self.estimator = ImportanceEstimator()
        elif self.cfg.estimator == "proposal":
            self.prop_net = create_network_with_input_encoding(
                **self.cfg.proposal_network_config
            )
            self.prop_optim = parse_optimizer(
                self.cfg.prop_optimizer_config, self.prop_net
            )
            self.prop_scheduler = (
                parse_scheduler_to_instance(
                    self.cfg.prop_scheduler_config, self.prop_optim
                )
                if self.cfg.prop_scheduler_config is not None
                else None
            )
            self.estimator = nerfacc.PropNetEstimator(
                self.prop_optim, self.prop_scheduler
            )

            def get_proposal_requires_grad_fn(
                target: float = 5.0, num_steps: int = 1000
            ):
                schedule = lambda s: min(s / num_steps, 1.0) * target

                steps_since_last_grad = 0

                def proposal_requires_grad_fn(step: int) -> bool:
                    nonlocal steps_since_last_grad
                    target_steps_since_last_grad = schedule(step)
                    requires_grad = steps_since_last_grad > target_steps_since_last_grad
                    if requires_grad:
                        steps_since_last_grad = 0
                    steps_since_last_grad += 1
                    return requires_grad

                return proposal_requires_grad_fn

            self.proposal_requires_grad_fn = get_proposal_requires_grad_fn()
            self.randomized = self.cfg.randomized
        else:
            raise NotImplementedError(
                "Unknown estimator, should be one of ['occgrid', 'proposal', 'importance']."
            )

        # for proposal
        self.vars_in_forward = {}

    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        gaussian_mean: np.ndarray = None,
        gaussian_var: np.ndarray = None,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:

        if len(gaussian_mean) == 3:
            rays_o = rays_o + gaussian_mean


        batch_size, height, width = rays_o.shape[:3]
        rays_o_flatten: Float[Tensor, "Nr 3"] = rays_o.reshape(-1, 3)
        rays_d_flatten: Float[Tensor, "Nr 3"] = rays_d.reshape(-1, 3)
        light_positions_flatten: Float[Tensor, "Nr 3"] = (
            light_positions.reshape(-1, 1, 1, 3)
            .expand(-1, height, width, -1)
            .reshape(-1, 3)
        )
        n_rays = rays_o_flatten.shape[0]
        if self.cfg.estimator == "occgrid":
            if not self.cfg.grid_prune:
                with torch.no_grad():
                    ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                        rays_o_flatten,
                        rays_d_flatten,
                        sigma_fn=None,
                        near_plane=self.cfg.near_plane,
                        far_plane=self.cfg.far_plane,
                        render_step_size=self.render_step_size,
                        alpha_thre=0.0,
                        stratified=self.randomized,
                        cone_angle=0.0,
                        early_stop_eps=0,
                    )
            else:
                def sigma_fn(t_starts, t_ends, ray_indices):
                    t_starts, t_ends = t_starts[..., None], t_ends[..., None]
                    t_origins = rays_o_flatten[ray_indices]
                    t_positions = (t_starts + t_ends) / 2.0
                    t_dirs = rays_d_flatten[ray_indices]
                    positions = t_origins + t_dirs * t_positions # shape（num_points, 3）
 
                    r = positions - gaussian_mean
                    gaussian_field = torch.exp(-0.5 * torch.sum(r * r / gaussian_var, dim=-1))
                    a = torch.sum(gaussian_field>0.5)
                    if self.training:
                        sigma = self.geometry.forward_density(positions)[..., 0]
                        sigma = sigma * gaussian_field
                    else:
                        sigma = chunk_batch(
                            self.geometry.forward_density,
                            self.cfg.eval_chunk_size,
                            positions,
                        )[..., 0]
                        sigma = sigma * gaussian_field
                    #delete constraint and selector
                    

                    return sigma

                with torch.no_grad():
                    ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                        rays_o_flatten,
                        rays_d_flatten,
                        sigma_fn=sigma_fn if self.cfg.prune_alpha_threshold else None,
                        near_plane=self.cfg.near_plane,
                        far_plane=self.cfg.far_plane,
                        render_step_size=self.render_step_size,
                        alpha_thre=0.01 if self.cfg.prune_alpha_threshold else 0.0,
                        stratified=self.randomized,
                        cone_angle=0.0,
                    )

        elif self.cfg.estimator == "proposal":

            def prop_sigma_fn(
                t_starts: Float[Tensor, "Nr Ns"],
                t_ends: Float[Tensor, "Nr Ns"],
                proposal_network,
            ):
                t_origins: Float[Tensor, "Nr 1 3"] = rays_o_flatten.unsqueeze(-2)
                t_dirs: Float[Tensor, "Nr 1 3"] = rays_d_flatten.unsqueeze(-2)
                positions: Float[Tensor, "Nr Ns 3"] = (
                    t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
                )
                aabb_min, aabb_max = self.bbox[0], self.bbox[1]
                positions = (positions - aabb_min) / (aabb_max - aabb_min)
                selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
                density_before_activation = (
                    proposal_network(positions.view(-1, 3))
                    .view(*positions.shape[:-1], 1)
                    .to(positions)
                )
                density: Float[Tensor, "Nr Ns 1"] = (
                    get_activation("shifted_trunc_exp")(density_before_activation)
                    * selector[..., None]
                )
                return density.squeeze(-1)

            t_starts_, t_ends_ = self.estimator.sampling(
                prop_sigma_fns=[partial(prop_sigma_fn, proposal_network=self.prop_net)],
                prop_samples=[self.cfg.num_samples_per_ray_proposal],
                num_samples=self.cfg.num_samples_per_ray,
                n_rays=n_rays,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                sampling_type="uniform",
                stratified=self.randomized,
                requires_grad=self.vars_in_forward["requires_grad"],
            )
            ray_indices = (
                torch.arange(n_rays, device=rays_o_flatten.device)
                .unsqueeze(-1)
                .expand(-1, t_starts_.shape[1])
            )
            ray_indices = ray_indices.flatten()
            t_starts_ = t_starts_.flatten()
            t_ends_ = t_ends_.flatten()
        elif self.cfg.estimator == "importance":

            def prop_sigma_fn(
                t_starts: Float[Tensor, "Nr Ns"],
                t_ends: Float[Tensor, "Nr Ns"],
                proposal_network,
            ):
                t_origins: Float[Tensor, "Nr 1 3"] = rays_o_flatten.unsqueeze(-2)
                t_dirs: Float[Tensor, "Nr 1 3"] = rays_d_flatten.unsqueeze(-2)
                positions: Float[Tensor, "Nr Ns 3"] = (
                    t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
                )
                with torch.no_grad():
                    geo_out = chunk_batch(
                        proposal_network,
                        self.cfg.eval_chunk_size,
                        positions.reshape(-1, 3),
                        output_normal=False,
                    )
                    density = geo_out["density"]
                return density.reshape(positions.shape[:2])

            t_starts_, t_ends_ = self.estimator.sampling(
                prop_sigma_fns=[partial(prop_sigma_fn, proposal_network=self.geometry)],
                prop_samples=[self.cfg.num_samples_per_ray_importance],
                num_samples=self.cfg.num_samples_per_ray,
                n_rays=n_rays,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                sampling_type="uniform",
                stratified=self.randomized,
            )
            ray_indices = (
                torch.arange(n_rays, device=rays_o_flatten.device)
                .unsqueeze(-1)

                .expand(-1, t_starts_.shape[1])
            )
            ray_indices = ray_indices.flatten()
            t_starts_ = t_starts_.flatten()
            t_ends_ = t_ends_.flatten()
        else:
            raise NotImplementedError

        ray_indices, t_starts_, t_ends_ = validate_empty_rays(
            ray_indices, t_starts_, t_ends_
        )
        ray_indices = ray_indices.long()
        t_starts, t_ends = t_starts_[..., None], t_ends_[..., None]
        t_origins = rays_o_flatten[ray_indices]
        t_dirs = rays_d_flatten[ray_indices]
        t_light_positions = light_positions_flatten[ray_indices]
        t_positions = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * t_positions
        t_intervals = t_ends - t_starts

        if self.training:
            geo_out = self.geometry(
                positions, output_normal=self.material.requires_normal
            )
            rgb_fg_all = self.material(
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                **geo_out,
                **kwargs
            )
            comp_rgb_bg = self.background(dirs=rays_d)
        else:
            geo_out = chunk_batch(
                self.geometry,
                self.cfg.eval_chunk_size,
                positions,
                output_normal=self.material.requires_normal,
            )
            rgb_fg_all = chunk_batch(
                self.material,
                self.cfg.eval_chunk_size,
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                **geo_out
            )
            comp_rgb_bg = chunk_batch(
                self.background, self.cfg.eval_chunk_size, dirs=rays_d
            )

        weights: Float[Tensor, "Nr 1"]
        weights_, trans_, _ = nerfacc.render_weight_from_density(
            t_starts[..., 0],
            t_ends[..., 0],
            geo_out["density"][..., 0],
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        if self.training and self.cfg.estimator == "proposal":
            self.vars_in_forward["trans"] = trans_.reshape(n_rays, -1)

        weights = weights_[..., None]
        opacity: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=None, ray_indices=ray_indices, n_rays=n_rays
        )
        depth: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=t_positions, ray_indices=ray_indices, n_rays=n_rays
        )
        comp_rgb_fg: Float[Tensor, "Nr Nc"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=rgb_fg_all, ray_indices=ray_indices, n_rays=n_rays
        )

        # populate depth and opacity to each point
        weights_normalized = weights / opacity.clamp(min=1e-5)[ray_indices]  # num_pts
        # z-variance loss from HiFA: https://hifa-team.github.io/HiFA-site/
        z_mean: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights_normalized[..., 0],
            values=t_positions,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        z_variance_unmasked = nerfacc.accumulate_along_rays(
            weights_normalized[..., 0],
            values=(t_positions - z_mean[ray_indices]) ** 2,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        z_variance = z_variance_unmasked * (opacity > 0.5).float()

        if bg_color is None:
            bg_color = comp_rgb_bg
        else:
            if bg_color.shape[:-1] == (batch_size,):
                # e.g. constant random color used for Zero123
                # [bs,3] -> [bs, 1, 1, 3]):
                bg_color = bg_color.unsqueeze(1).unsqueeze(1)
                #        -> [bs, height, width, 3]):
                bg_color = bg_color.expand(-1, height, width, -1)

        if bg_color.shape[:-1] == (batch_size, height, width):
            bg_color = bg_color.reshape(batch_size * height * width, -1)

        comp_rgb = comp_rgb_fg + bg_color * (1.0 - opacity)

        out = {
            "comp_rgb": comp_rgb.view(batch_size, height, width, -1),
            "comp_rgb_fg": comp_rgb_fg.view(batch_size, height, width, -1),
            "comp_rgb_bg": comp_rgb_bg.view(batch_size, height, width, -1),
            "opacity": opacity.view(batch_size, height, width, 1),
            "depth": depth.view(batch_size, height, width, 1),
            "z_variance": z_variance.view(batch_size, height, width, 1),
            "geo_out": geo_out,
            "positions": positions
        }

        if self.training:
            out.update(
                {
                    "weights": weights,
                    "t_points": t_positions,
                    "t_intervals": t_intervals,
                    "t_dirs": t_dirs,
                    "ray_indices": ray_indices,
                    "points": positions,
                    **geo_out,
                }
            )
            if "normal" in geo_out:
                if self.cfg.return_comp_normal:
                    comp_normal: Float[Tensor, "Nr 3"] = nerfacc.accumulate_along_rays(
                        weights[..., 0],
                        values=geo_out["normal"],
                        ray_indices=ray_indices,
                        n_rays=n_rays,
                    )
                    comp_normal = F.normalize(comp_normal, dim=-1)
                    comp_normal = (
                        (comp_normal + 1.0) / 2.0 * opacity
                    )  # for visualization
                    out.update(
                        {
                            "comp_normal": comp_normal.view(
                                batch_size, height, width, 3
                            ),
                        }
                    )
                if self.cfg.return_normal_perturb:
                    normal_perturb = self.geometry(
                        positions + torch.randn_like(positions) * 1e-2,
                        output_normal=self.material.requires_normal,
                    )["normal"]
                    out.update({"normal_perturb": normal_perturb})
        else:
            if "normal" in geo_out:
                comp_normal = nerfacc.accumulate_along_rays(
                    weights[..., 0],
                    values=geo_out["normal"],
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
                comp_normal = F.normalize(comp_normal, dim=-1)
                comp_normal = (comp_normal + 1.0) / 2.0 * opacity  # for visualization
                out.update(
                    {
                        "comp_normal": comp_normal.view(batch_size, height, width, 3),
                    }
                )

        return out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        if self.cfg.estimator == "occgrid":
            if self.cfg.grid_prune:

                def occ_eval_fn(x):
                    density = self.geometry.forward_density(x)
                    # approximate for 1 - torch.exp(-density * self.render_step_size) based on taylor series
                    return density * self.render_step_size

                if self.training and not on_load_weights:
                    self.estimator.update_every_n_steps(
                        step=global_step, occ_eval_fn=occ_eval_fn
                    )
        elif self.cfg.estimator == "proposal":
            if self.training:
                requires_grad = self.proposal_requires_grad_fn(global_step)
                self.vars_in_forward["requires_grad"] = requires_grad
            else:
                self.vars_in_forward["requires_grad"] = False

    def update_step_end(self, epoch: int, global_step: int) -> None:
        if self.cfg.estimator == "proposal" and self.training:
            self.estimator.update_every_n_steps(
                self.vars_in_forward["trans"],
                self.vars_in_forward["requires_grad"],
                loss_scaler=1.0,
            )

    def train(self, mode=True):
        self.randomized = mode and self.cfg.randomized
        if self.cfg.estimator == "proposal":
            self.prop_net.train()
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        if self.cfg.estimator == "proposal":
            self.prop_net.eval()
        return super().eval()



@threestudio.register("nerf-multi-volume-bounded-renderer")
class NeRFMultiVolumeBoundedRenderer(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        num_samples_per_ray: int = 512
        eval_chunk_size: int = 160000
        randomized: bool = True

        near_plane: float = 0.0
        far_plane: float = 1e10

        return_comp_normal: bool = False
        return_normal_perturb: bool = False

        # in ["occgrid", "proposal", "importance"]
        estimator: str = "occgrid"

        # for occgrid
        grid_prune: bool = True
        prune_alpha_threshold: bool = True

        # for proposal
        proposal_network_config: Optional[dict] = None
        prop_optimizer_config: Optional[dict] = None
        prop_scheduler_config: Optional[dict] = None
        num_samples_per_ray_proposal: int = 64

        # for importance
        num_samples_per_ray_importance: int = 64

    cfg: Config


    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)

        self.bound = None
        if self.cfg.estimator == "occgrid":
            self.estimator = nerfacc.OccGridEstimator(
                roi_aabb=self.bbox.view(-1), resolution=32, levels=1
            )
            if not self.cfg.grid_prune:
                self.estimator.occs.fill_(True)
                self.estimator.binaries.fill_(True)
            self.render_step_size = (
                1.732 * 2 * self.cfg.radius / self.cfg.num_samples_per_ray
            )
            self.randomized = self.cfg.randomized
        elif self.cfg.estimator == "importance":
            self.estimator = ImportanceEstimator()
        elif self.cfg.estimator == "proposal":
            self.prop_net = create_network_with_input_encoding(
                **self.cfg.proposal_network_config
            )
            self.prop_optim = parse_optimizer(
                self.cfg.prop_optimizer_config, self.prop_net
            )
            self.prop_scheduler = (
                parse_scheduler_to_instance(
                    self.cfg.prop_scheduler_config, self.prop_optim
                )
                if self.cfg.prop_scheduler_config is not None
                else None
            )
            self.estimator = nerfacc.PropNetEstimator(
                self.prop_optim, self.prop_scheduler
            )

            def get_proposal_requires_grad_fn(
                target: float = 5.0, num_steps: int = 1000
            ):
                schedule = lambda s: min(s / num_steps, 1.0) * target

                steps_since_last_grad = 0

                def proposal_requires_grad_fn(step: int) -> bool:
                    nonlocal steps_since_last_grad
                    target_steps_since_last_grad = schedule(step)
                    requires_grad = steps_since_last_grad > target_steps_since_last_grad
                    if requires_grad:
                        steps_since_last_grad = 0
                    steps_since_last_grad += 1
                    return requires_grad

                return proposal_requires_grad_fn

            self.proposal_requires_grad_fn = get_proposal_requires_grad_fn()
            self.randomized = self.cfg.randomized
        else:
            raise NotImplementedError(
                "Unknown estimator, should be one of ['occgrid', 'proposal', 'importance']."
            )

        # for proposal
        self.vars_in_forward = {}

    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        bound: np.ndarray = None,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:

        resolution = len(bound)
        bound_i, bound_j, bound_k = torch.nonzero(bound, as_tuple=True)
        min_i, max_i = torch.min(bound_i), torch.max(bound_i)
        min_j, max_j = torch.min(bound_j), torch.max(bound_j)
        min_k, max_k = torch.min(bound_k), torch.max(bound_k)

        scale_i, scale_j, scale_k = max_i - min_i, max_j - min_j, max_k - min_k
        ratio_i, ratio_j, ratio_k = scale_i / resolution, scale_j / resolution, scale_k / resolution

        center_i, center_j, center_k = torch.mean(bound_i.float()), torch.mean(bound_j.float()), torch.mean(bound_k.float())
        center_i, center_j, center_k = center_i- resolution//2, center_j- resolution//2, center_k- resolution//2
        center_i, center_j, center_k = center_i / resolution, center_j / resolution, center_k / resolution

        # rays_o = rays_o * torch.tensor([ratio_i*3, ratio_j*3, ratio_k*3]).to(rays_o.device)
        rays_o = rays_o + torch.tensor([center_i, center_j, center_k]).to(rays_o.device)


        batch_size, height, width = rays_o.shape[:3]
        rays_o_flatten: Float[Tensor, "Nr 3"] = rays_o.reshape(-1, 3)
        rays_d_flatten: Float[Tensor, "Nr 3"] = rays_d.reshape(-1, 3)
        light_positions_flatten: Float[Tensor, "Nr 3"] = (
            light_positions.reshape(-1, 1, 1, 3)
            .expand(-1, height, width, -1)
            .reshape(-1, 3)
        )
        n_rays = rays_o_flatten.shape[0]
        self.bound = bound.clone().detach().to(self.device)
        if self.cfg.estimator == "occgrid":
            if not self.cfg.grid_prune:
                with torch.no_grad():
                    ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                        rays_o_flatten,
                        rays_d_flatten,
                        sigma_fn=None,
                        near_plane=self.cfg.near_plane,
                        far_plane=self.cfg.far_plane,
                        render_step_size=self.render_step_size,
                        alpha_thre=0.0,
                        stratified=self.randomized,
                        cone_angle=0.0,
                        early_stop_eps=0,
                    )
            else:
                def sigma_fn(t_starts, t_ends, ray_indices):
                    t_starts, t_ends = t_starts[..., None], t_ends[..., None]
                    t_origins = rays_o_flatten[ray_indices]
                    t_positions = (t_starts + t_ends) / 2.0
                    t_dirs = rays_d_flatten[ray_indices]
                    positions = t_origins + t_dirs * t_positions # shape（num_points, 3）
                    resolution = len(self.bound) # x, y, z
    
                    #expand the height and width to the resolution
                    enlarged_positions = deepcopy(positions)
                    enlarged_positions = enlarged_positions * resolution//2 + resolution//2
                    enlarged_positions = enlarged_positions.to(torch.int64)

                    enlarged_positions = torch.clamp(enlarged_positions, 0, resolution-1)
                    selector = self.bound[enlarged_positions[:, 0], enlarged_positions[:, 1], enlarged_positions[:, 2]] > 0
                    # print(torch.max(positions), torch.min(positions)) # here is the place to change the density

                    if self.training:
                        sigma = self.geometry.forward_density(positions)[..., 0]
                        sigma = sigma * selector
                    else:
                        sigma = chunk_batch(
                            self.geometry.forward_density,
                            self.cfg.eval_chunk_size,
                            positions,
                        )[..., 0]
                        sigma = sigma * selector
                    #delete constraint and selector
                    

                    return sigma

                with torch.no_grad():
                    ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                        rays_o_flatten,
                        rays_d_flatten,
                        sigma_fn=sigma_fn if self.cfg.prune_alpha_threshold else None,
                        near_plane=self.cfg.near_plane,
                        far_plane=self.cfg.far_plane,
                        render_step_size=self.render_step_size,
                        alpha_thre=0.01 if self.cfg.prune_alpha_threshold else 0.0,
                        stratified=self.randomized,
                        cone_angle=0.0,
                    )

        elif self.cfg.estimator == "proposal":

            def prop_sigma_fn(
                t_starts: Float[Tensor, "Nr Ns"],
                t_ends: Float[Tensor, "Nr Ns"],
                proposal_network,
            ):
                t_origins: Float[Tensor, "Nr 1 3"] = rays_o_flatten.unsqueeze(-2)
                t_dirs: Float[Tensor, "Nr 1 3"] = rays_d_flatten.unsqueeze(-2)
                positions: Float[Tensor, "Nr Ns 3"] = (
                    t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
                )
                aabb_min, aabb_max = self.bbox[0], self.bbox[1]
                positions = (positions - aabb_min) / (aabb_max - aabb_min)
                selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
                density_before_activation = (
                    proposal_network(positions.view(-1, 3))
                    .view(*positions.shape[:-1], 1)
                    .to(positions)
                )
                density: Float[Tensor, "Nr Ns 1"] = (
                    get_activation("shifted_trunc_exp")(density_before_activation)
                    * selector[..., None]
                )
                return density.squeeze(-1)

            t_starts_, t_ends_ = self.estimator.sampling(
                prop_sigma_fns=[partial(prop_sigma_fn, proposal_network=self.prop_net)],
                prop_samples=[self.cfg.num_samples_per_ray_proposal],
                num_samples=self.cfg.num_samples_per_ray,
                n_rays=n_rays,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                sampling_type="uniform",
                stratified=self.randomized,
                requires_grad=self.vars_in_forward["requires_grad"],
            )
            ray_indices = (
                torch.arange(n_rays, device=rays_o_flatten.device)
                .unsqueeze(-1)
                .expand(-1, t_starts_.shape[1])
            )
            ray_indices = ray_indices.flatten()
            t_starts_ = t_starts_.flatten()
            t_ends_ = t_ends_.flatten()
        elif self.cfg.estimator == "importance":

            def prop_sigma_fn(
                t_starts: Float[Tensor, "Nr Ns"],
                t_ends: Float[Tensor, "Nr Ns"],
                proposal_network,
            ):
                t_origins: Float[Tensor, "Nr 1 3"] = rays_o_flatten.unsqueeze(-2)
                t_dirs: Float[Tensor, "Nr 1 3"] = rays_d_flatten.unsqueeze(-2)
                positions: Float[Tensor, "Nr Ns 3"] = (
                    t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
                )
                with torch.no_grad():
                    geo_out = chunk_batch(
                        proposal_network,
                        self.cfg.eval_chunk_size,
                        positions.reshape(-1, 3),
                        output_normal=False,
                    )
                    density = geo_out["density"]
                return density.reshape(positions.shape[:2])

            t_starts_, t_ends_ = self.estimator.sampling(
                prop_sigma_fns=[partial(prop_sigma_fn, proposal_network=self.geometry)],
                prop_samples=[self.cfg.num_samples_per_ray_importance],
                num_samples=self.cfg.num_samples_per_ray,
                n_rays=n_rays,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                sampling_type="uniform",
                stratified=self.randomized,
            )
            ray_indices = (
                torch.arange(n_rays, device=rays_o_flatten.device)
                .unsqueeze(-1)

                .expand(-1, t_starts_.shape[1])
            )
            ray_indices = ray_indices.flatten()
            t_starts_ = t_starts_.flatten()
            t_ends_ = t_ends_.flatten()
        else:
            raise NotImplementedError

        ray_indices, t_starts_, t_ends_ = validate_empty_rays(
            ray_indices, t_starts_, t_ends_
        )
        ray_indices = ray_indices.long()
        t_starts, t_ends = t_starts_[..., None], t_ends_[..., None]
        t_origins = rays_o_flatten[ray_indices]
        t_dirs = rays_d_flatten[ray_indices]
        t_light_positions = light_positions_flatten[ray_indices]
        t_positions = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * t_positions
        t_intervals = t_ends - t_starts

        if self.training:
            geo_out = self.geometry(
                positions, output_normal=self.material.requires_normal
            )
            rgb_fg_all = self.material(
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                **geo_out,
                **kwargs
            )
            comp_rgb_bg = self.background(dirs=rays_d)
        else:
            geo_out = chunk_batch(
                self.geometry,
                self.cfg.eval_chunk_size,
                positions,
                output_normal=self.material.requires_normal,
            )
            rgb_fg_all = chunk_batch(
                self.material,
                self.cfg.eval_chunk_size,
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                **geo_out
            )
            comp_rgb_bg = chunk_batch(
                self.background, self.cfg.eval_chunk_size, dirs=rays_d
            )

        weights: Float[Tensor, "Nr 1"]
        weights_, trans_, _ = nerfacc.render_weight_from_density(
            t_starts[..., 0],
            t_ends[..., 0],
            geo_out["density"][..., 0],
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        if self.training and self.cfg.estimator == "proposal":
            self.vars_in_forward["trans"] = trans_.reshape(n_rays, -1)

        weights = weights_[..., None]
        opacity: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=None, ray_indices=ray_indices, n_rays=n_rays
        )
        depth: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=t_positions, ray_indices=ray_indices, n_rays=n_rays
        )
        comp_rgb_fg: Float[Tensor, "Nr Nc"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=rgb_fg_all, ray_indices=ray_indices, n_rays=n_rays
        )

        # populate depth and opacity to each point
        weights_normalized = weights / opacity.clamp(min=1e-5)[ray_indices]  # num_pts
        # z-variance loss from HiFA: https://hifa-team.github.io/HiFA-site/
        z_mean: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights_normalized[..., 0],
            values=t_positions,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        z_variance_unmasked = nerfacc.accumulate_along_rays(
            weights_normalized[..., 0],
            values=(t_positions - z_mean[ray_indices]) ** 2,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        z_variance = z_variance_unmasked * (opacity > 0.5).float()

        if bg_color is None:
            bg_color = comp_rgb_bg
        else:
            if bg_color.shape[:-1] == (batch_size,):
                # e.g. constant random color used for Zero123
                # [bs,3] -> [bs, 1, 1, 3]):
                bg_color = bg_color.unsqueeze(1).unsqueeze(1)
                #        -> [bs, height, width, 3]):
                bg_color = bg_color.expand(-1, height, width, -1)

        if bg_color.shape[:-1] == (batch_size, height, width):
            bg_color = bg_color.reshape(batch_size * height * width, -1)

        comp_rgb = comp_rgb_fg + bg_color * (1.0 - opacity)

        out = {
            "comp_rgb": comp_rgb.view(batch_size, height, width, -1),
            "comp_rgb_fg": comp_rgb_fg.view(batch_size, height, width, -1),
            "comp_rgb_bg": comp_rgb_bg.view(batch_size, height, width, -1),
            "opacity": opacity.view(batch_size, height, width, 1),
            "depth": depth.view(batch_size, height, width, 1),
            "z_variance": z_variance.view(batch_size, height, width, 1),
            "geo_out": geo_out,
            "positions": positions
        }

        if self.training:
            out.update(
                {
                    "weights": weights,
                    "t_points": t_positions,
                    "t_intervals": t_intervals,
                    "t_dirs": t_dirs,
                    "ray_indices": ray_indices,
                    "points": positions,
                    **geo_out,
                }
            )
            if "normal" in geo_out:
                if self.cfg.return_comp_normal:
                    comp_normal: Float[Tensor, "Nr 3"] = nerfacc.accumulate_along_rays(
                        weights[..., 0],
                        values=geo_out["normal"],
                        ray_indices=ray_indices,
                        n_rays=n_rays,
                    )
                    comp_normal = F.normalize(comp_normal, dim=-1)
                    comp_normal = (
                        (comp_normal + 1.0) / 2.0 * opacity
                    )  # for visualization
                    out.update(
                        {
                            "comp_normal": comp_normal.view(
                                batch_size, height, width, 3
                            ),
                        }
                    )
                if self.cfg.return_normal_perturb:
                    normal_perturb = self.geometry(
                        positions + torch.randn_like(positions) * 1e-2,
                        output_normal=self.material.requires_normal,
                    )["normal"]
                    out.update({"normal_perturb": normal_perturb})
        else:
            if "normal" in geo_out:
                comp_normal = nerfacc.accumulate_along_rays(
                    weights[..., 0],
                    values=geo_out["normal"],
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
                comp_normal = F.normalize(comp_normal, dim=-1)
                comp_normal = (comp_normal + 1.0) / 2.0 * opacity  # for visualization
                out.update(
                    {
                        "comp_normal": comp_normal.view(batch_size, height, width, 3),
                    }
                )

        return out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        if self.cfg.estimator == "occgrid":
            if self.cfg.grid_prune:

                def occ_eval_fn(x):
                    density = self.geometry.forward_density(x)
                    # approximate for 1 - torch.exp(-density * self.render_step_size) based on taylor series
                    return density * self.render_step_size

                if self.training and not on_load_weights:
                    self.estimator.update_every_n_steps(
                        step=global_step, occ_eval_fn=occ_eval_fn
                    )
        elif self.cfg.estimator == "proposal":
            if self.training:
                requires_grad = self.proposal_requires_grad_fn(global_step)
                self.vars_in_forward["requires_grad"] = requires_grad
            else:
                self.vars_in_forward["requires_grad"] = False

    def update_step_end(self, epoch: int, global_step: int) -> None:
        if self.cfg.estimator == "proposal" and self.training:
            self.estimator.update_every_n_steps(
                self.vars_in_forward["trans"],
                self.vars_in_forward["requires_grad"],
                loss_scaler=1.0,
            )

    def train(self, mode=True):
        self.randomized = mode and self.cfg.randomized
        if self.cfg.estimator == "proposal":
            self.prop_net.train()
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        if self.cfg.estimator == "proposal":
            self.prop_net.eval()
        return super().eval()



@threestudio.register("nerf-volume-bounded-renderer")
class NeRFVolumeBoundedRenderer(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        num_samples_per_ray: int = 512
        eval_chunk_size: int = 160000
        randomized: bool = True

        near_plane: float = 0.0
        far_plane: float = 1e10

        return_comp_normal: bool = False
        return_normal_perturb: bool = False

        # in ["occgrid", "proposal", "importance"]
        estimator: str = "occgrid"

        # for occgrid
        grid_prune: bool = True
        prune_alpha_threshold: bool = True

        # for proposal
        proposal_network_config: Optional[dict] = None
        prop_optimizer_config: Optional[dict] = None
        prop_scheduler_config: Optional[dict] = None
        num_samples_per_ray_proposal: int = 64

        # for importance
        num_samples_per_ray_importance: int = 64

        # bound: Any = field(default_factory=lambda: [0.1, 0.2, 0.3])
        bound_path: str = None

    cfg: Config


    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        assert self.cfg.bound_path is not None, "bound_path should be provided."
        self.bound = np.load(self.cfg.bound_path)
        self.bound = torch.tensor(self.bound).to(self.device)

        if self.cfg.estimator == "occgrid":
            self.estimator = nerfacc.OccGridEstimator(
                roi_aabb=self.bbox.view(-1), resolution=32, levels=1
            )
            if not self.cfg.grid_prune:
                self.estimator.occs.fill_(True)
                self.estimator.binaries.fill_(True)
            self.render_step_size = (
                1.732 * 2 * self.cfg.radius / self.cfg.num_samples_per_ray
            )
            self.randomized = self.cfg.randomized
        elif self.cfg.estimator == "importance":
            self.estimator = ImportanceEstimator()
        elif self.cfg.estimator == "proposal":
            self.prop_net = create_network_with_input_encoding(
                **self.cfg.proposal_network_config
            )
            self.prop_optim = parse_optimizer(
                self.cfg.prop_optimizer_config, self.prop_net
            )
            self.prop_scheduler = (
                parse_scheduler_to_instance(
                    self.cfg.prop_scheduler_config, self.prop_optim
                )
                if self.cfg.prop_scheduler_config is not None
                else None
            )
            self.estimator = nerfacc.PropNetEstimator(
                self.prop_optim, self.prop_scheduler
            )

            def get_proposal_requires_grad_fn(
                target: float = 5.0, num_steps: int = 1000
            ):
                schedule = lambda s: min(s / num_steps, 1.0) * target

                steps_since_last_grad = 0

                def proposal_requires_grad_fn(step: int) -> bool:
                    nonlocal steps_since_last_grad
                    target_steps_since_last_grad = schedule(step)
                    requires_grad = steps_since_last_grad > target_steps_since_last_grad
                    if requires_grad:
                        steps_since_last_grad = 0
                    steps_since_last_grad += 1
                    return requires_grad

                return proposal_requires_grad_fn

            self.proposal_requires_grad_fn = get_proposal_requires_grad_fn()
            self.randomized = self.cfg.randomized
        else:
            raise NotImplementedError(
                "Unknown estimator, should be one of ['occgrid', 'proposal', 'importance']."
            )

        # for proposal
        self.vars_in_forward = {}

    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        # import pdb; pdb.set_trace()
        batch_size, height, width = rays_o.shape[:3]
        rays_o_flatten: Float[Tensor, "Nr 3"] = rays_o.reshape(-1, 3)
        rays_d_flatten: Float[Tensor, "Nr 3"] = rays_d.reshape(-1, 3)
        light_positions_flatten: Float[Tensor, "Nr 3"] = (
            light_positions.reshape(-1, 1, 1, 3)
            .expand(-1, height, width, -1)
            .reshape(-1, 3)
        )
        n_rays = rays_o_flatten.shape[0]

        if self.cfg.estimator == "occgrid":
            if not self.cfg.grid_prune:
                with torch.no_grad():
                    ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                        rays_o_flatten,
                        rays_d_flatten,
                        sigma_fn=None,
                        near_plane=self.cfg.near_plane,
                        far_plane=self.cfg.far_plane,
                        render_step_size=self.render_step_size,
                        alpha_thre=0.0,
                        stratified=self.randomized,
                        cone_angle=0.0,
                        early_stop_eps=0,
                    )
            else:

                def sigma_fn(t_starts, t_ends, ray_indices):
                    t_starts, t_ends = t_starts[..., None], t_ends[..., None]
                    t_origins = rays_o_flatten[ray_indices]
                    t_positions = (t_starts + t_ends) / 2.0
                    t_dirs = rays_d_flatten[ray_indices]
                    positions = t_origins + t_dirs * t_positions # shape（num_points, 3）
                    resolution = len(self.bound)
                    #expand the height and width to the resolution
                    enlarged_positions = deepcopy(positions)
                    enlarged_positions[:, :2] = positions[:, :2] * resolution//2 + resolution//2
                    enlarged_positions = enlarged_positions.to(torch.int64)
                    enlarged_positions = torch.clamp(enlarged_positions, 0, resolution-1)
                    upper_bound  = self.bound[enlarged_positions[:, 0], enlarged_positions[:, 1], 1]
                    lower_bound = self.bound[enlarged_positions[:, 0], enlarged_positions[:, 1], 0]
        
                    selector = (positions[:, 2] > lower_bound) & (positions[:, 2] < upper_bound)
                    # print(torch.max(positions), torch.min(positions)) # here is the place to change the density

                    if self.training:
                        sigma = self.geometry.forward_density(positions)[..., 0]
                        sigma = sigma * selector
                        stop = 1
                    else:
                        sigma = chunk_batch(
                            self.geometry.forward_density,
                            self.cfg.eval_chunk_size,
                            positions,
                        )[..., 0]
                        sigma = sigma * selector
                    #delete constraint and selector
                    

                    return sigma

                with torch.no_grad():
                    ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                        rays_o_flatten,
                        rays_d_flatten,
                        sigma_fn=sigma_fn if self.cfg.prune_alpha_threshold else None,
                        near_plane=self.cfg.near_plane,
                        far_plane=self.cfg.far_plane,
                        render_step_size=self.render_step_size,
                        alpha_thre=0.01 if self.cfg.prune_alpha_threshold else 0.0,
                        stratified=self.randomized,
                        cone_angle=0.0,
                    )

        elif self.cfg.estimator == "proposal":

            def prop_sigma_fn(
                t_starts: Float[Tensor, "Nr Ns"],
                t_ends: Float[Tensor, "Nr Ns"],
                proposal_network,
            ):
                t_origins: Float[Tensor, "Nr 1 3"] = rays_o_flatten.unsqueeze(-2)
                t_dirs: Float[Tensor, "Nr 1 3"] = rays_d_flatten.unsqueeze(-2)
                positions: Float[Tensor, "Nr Ns 3"] = (
                    t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
                )
                aabb_min, aabb_max = self.bbox[0], self.bbox[1]
                positions = (positions - aabb_min) / (aabb_max - aabb_min)
                selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
                density_before_activation = (
                    proposal_network(positions.view(-1, 3))
                    .view(*positions.shape[:-1], 1)
                    .to(positions)
                )
                density: Float[Tensor, "Nr Ns 1"] = (
                    get_activation("shifted_trunc_exp")(density_before_activation)
                    * selector[..., None]
                )
                return density.squeeze(-1)

            t_starts_, t_ends_ = self.estimator.sampling(
                prop_sigma_fns=[partial(prop_sigma_fn, proposal_network=self.prop_net)],
                prop_samples=[self.cfg.num_samples_per_ray_proposal],
                num_samples=self.cfg.num_samples_per_ray,
                n_rays=n_rays,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                sampling_type="uniform",
                stratified=self.randomized,
                requires_grad=self.vars_in_forward["requires_grad"],
            )
            ray_indices = (
                torch.arange(n_rays, device=rays_o_flatten.device)
                .unsqueeze(-1)
                .expand(-1, t_starts_.shape[1])
            )
            ray_indices = ray_indices.flatten()
            t_starts_ = t_starts_.flatten()
            t_ends_ = t_ends_.flatten()
        elif self.cfg.estimator == "importance":

            def prop_sigma_fn(
                t_starts: Float[Tensor, "Nr Ns"],
                t_ends: Float[Tensor, "Nr Ns"],
                proposal_network,
            ):
                t_origins: Float[Tensor, "Nr 1 3"] = rays_o_flatten.unsqueeze(-2)
                t_dirs: Float[Tensor, "Nr 1 3"] = rays_d_flatten.unsqueeze(-2)
                positions: Float[Tensor, "Nr Ns 3"] = (
                    t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
                )
                with torch.no_grad():
                    geo_out = chunk_batch(
                        proposal_network,
                        self.cfg.eval_chunk_size,
                        positions.reshape(-1, 3),
                        output_normal=False,
                    )
                    density = geo_out["density"]
                return density.reshape(positions.shape[:2])

            t_starts_, t_ends_ = self.estimator.sampling(
                prop_sigma_fns=[partial(prop_sigma_fn, proposal_network=self.geometry)],
                prop_samples=[self.cfg.num_samples_per_ray_importance],
                num_samples=self.cfg.num_samples_per_ray,
                n_rays=n_rays,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                sampling_type="uniform",
                stratified=self.randomized,
            )
            ray_indices = (
                torch.arange(n_rays, device=rays_o_flatten.device)
                .unsqueeze(-1)
                .expand(-1, t_starts_.shape[1])
            )
            ray_indices = ray_indices.flatten()
            t_starts_ = t_starts_.flatten()
            t_ends_ = t_ends_.flatten()
        else:
            raise NotImplementedError

        ray_indices, t_starts_, t_ends_ = validate_empty_rays(
            ray_indices, t_starts_, t_ends_
        )
        ray_indices = ray_indices.long()
        t_starts, t_ends = t_starts_[..., None], t_ends_[..., None]
        t_origins = rays_o_flatten[ray_indices]
        t_dirs = rays_d_flatten[ray_indices]
        t_light_positions = light_positions_flatten[ray_indices]
        t_positions = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * t_positions
        t_intervals = t_ends - t_starts

        if self.training:
            geo_out = self.geometry(
                positions, output_normal=self.material.requires_normal
            )
            rgb_fg_all = self.material(
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                **geo_out,
                **kwargs
            )
            comp_rgb_bg = self.background(dirs=rays_d)
        else:
            geo_out = chunk_batch(
                self.geometry,
                self.cfg.eval_chunk_size,
                positions,
                output_normal=self.material.requires_normal,
            )
            rgb_fg_all = chunk_batch(
                self.material,
                self.cfg.eval_chunk_size,
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                **geo_out
            )
            comp_rgb_bg = chunk_batch(
                self.background, self.cfg.eval_chunk_size, dirs=rays_d
            )

        weights: Float[Tensor, "Nr 1"]
        weights_, trans_, _ = nerfacc.render_weight_from_density(
            t_starts[..., 0],
            t_ends[..., 0],
            geo_out["density"][..., 0],
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        if self.training and self.cfg.estimator == "proposal":
            self.vars_in_forward["trans"] = trans_.reshape(n_rays, -1)

        weights = weights_[..., None]
        opacity: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=None, ray_indices=ray_indices, n_rays=n_rays
        )
        depth: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=t_positions, ray_indices=ray_indices, n_rays=n_rays
        )
        comp_rgb_fg: Float[Tensor, "Nr Nc"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=rgb_fg_all, ray_indices=ray_indices, n_rays=n_rays
        )

        # populate depth and opacity to each point
        weights_normalized = weights / opacity.clamp(min=1e-5)[ray_indices]  # num_pts
        # z-variance loss from HiFA: https://hifa-team.github.io/HiFA-site/
        z_mean: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights_normalized[..., 0],
            values=t_positions,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        z_variance_unmasked = nerfacc.accumulate_along_rays(
            weights_normalized[..., 0],
            values=(t_positions - z_mean[ray_indices]) ** 2,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        z_variance = z_variance_unmasked * (opacity > 0.5).float()

        if bg_color is None:
            bg_color = comp_rgb_bg
        else:
            if bg_color.shape[:-1] == (batch_size,):
                # e.g. constant random color used for Zero123
                # [bs,3] -> [bs, 1, 1, 3]):
                bg_color = bg_color.unsqueeze(1).unsqueeze(1)
                #        -> [bs, height, width, 3]):
                bg_color = bg_color.expand(-1, height, width, -1)

        if bg_color.shape[:-1] == (batch_size, height, width):
            bg_color = bg_color.reshape(batch_size * height * width, -1)

        comp_rgb = comp_rgb_fg + bg_color * (1.0 - opacity)

        out = {
            "comp_rgb": comp_rgb.view(batch_size, height, width, -1),
            "comp_rgb_fg": comp_rgb_fg.view(batch_size, height, width, -1),
            "comp_rgb_bg": comp_rgb_bg.view(batch_size, height, width, -1),
            "opacity": opacity.view(batch_size, height, width, 1),
            "depth": depth.view(batch_size, height, width, 1),
            "z_variance": z_variance.view(batch_size, height, width, 1),
        }

        if self.training:
            out.update(
                {
                    "weights": weights,
                    "t_points": t_positions,
                    "t_intervals": t_intervals,
                    "t_dirs": t_dirs,
                    "ray_indices": ray_indices,
                    "points": positions,
                    **geo_out,
                }
            )
            if "normal" in geo_out:
                if self.cfg.return_comp_normal:
                    comp_normal: Float[Tensor, "Nr 3"] = nerfacc.accumulate_along_rays(
                        weights[..., 0],
                        values=geo_out["normal"],
                        ray_indices=ray_indices,
                        n_rays=n_rays,
                    )
                    comp_normal = F.normalize(comp_normal, dim=-1)
                    comp_normal = (
                        (comp_normal + 1.0) / 2.0 * opacity
                    )  # for visualization
                    out.update(
                        {
                            "comp_normal": comp_normal.view(
                                batch_size, height, width, 3
                            ),
                        }
                    )
                if self.cfg.return_normal_perturb:
                    normal_perturb = self.geometry(
                        positions + torch.randn_like(positions) * 1e-2,
                        output_normal=self.material.requires_normal,
                    )["normal"]
                    out.update({"normal_perturb": normal_perturb})
        else:
            if "normal" in geo_out:
                comp_normal = nerfacc.accumulate_along_rays(
                    weights[..., 0],
                    values=geo_out["normal"],
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
                comp_normal = F.normalize(comp_normal, dim=-1)
                comp_normal = (comp_normal + 1.0) / 2.0 * opacity  # for visualization
                out.update(
                    {
                        "comp_normal": comp_normal.view(batch_size, height, width, 3),
                    }
                )

        return out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        if self.cfg.estimator == "occgrid":
            if self.cfg.grid_prune:

                def occ_eval_fn(x):
                    density = self.geometry.forward_density(x)
                    # approximate for 1 - torch.exp(-density * self.render_step_size) based on taylor series
                    return density * self.render_step_size

                if self.training and not on_load_weights:
                    self.estimator.update_every_n_steps(
                        step=global_step, occ_eval_fn=occ_eval_fn
                    )
        elif self.cfg.estimator == "proposal":
            if self.training:
                requires_grad = self.proposal_requires_grad_fn(global_step)
                self.vars_in_forward["requires_grad"] = requires_grad
            else:
                self.vars_in_forward["requires_grad"] = False

    def update_step_end(self, epoch: int, global_step: int) -> None:
        if self.cfg.estimator == "proposal" and self.training:
            self.estimator.update_every_n_steps(
                self.vars_in_forward["trans"],
                self.vars_in_forward["requires_grad"],
                loss_scaler=1.0,
            )

    def train(self, mode=True):
        self.randomized = mode and self.cfg.randomized
        if self.cfg.estimator == "proposal":
            self.prop_net.train()
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        if self.cfg.estimator == "proposal":
            self.prop_net.eval()
        return super().eval()






@threestudio.register("nerf-volume-renderer")
class NeRFVolumeRenderer(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        num_samples_per_ray: int = 512
        eval_chunk_size: int = 160000
        randomized: bool = True

        near_plane: float = 0.0
        far_plane: float = 1e10

        return_comp_normal: bool = False
        return_normal_perturb: bool = False

        # in ["occgrid", "proposal", "importance"]
        estimator: str = "occgrid"

        # for occgrid
        grid_prune: bool = True
        prune_alpha_threshold: bool = True

        # for proposal
        proposal_network_config: Optional[dict] = None
        prop_optimizer_config: Optional[dict] = None
        prop_scheduler_config: Optional[dict] = None
        num_samples_per_ray_proposal: int = 64

        # for importance
        num_samples_per_ray_importance: int = 64

    cfg: Config


    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        if self.cfg.estimator == "occgrid":
            self.estimator = nerfacc.OccGridEstimator(
                roi_aabb=self.bbox.view(-1), resolution=32, levels=1
            )
            if not self.cfg.grid_prune:
                self.estimator.occs.fill_(True)
                self.estimator.binaries.fill_(True)
            self.render_step_size = (
                1.732 * 2 * self.cfg.radius / self.cfg.num_samples_per_ray
            )
            self.randomized = self.cfg.randomized
        elif self.cfg.estimator == "importance":
            self.estimator = ImportanceEstimator()
        elif self.cfg.estimator == "proposal":
            self.prop_net = create_network_with_input_encoding(
                **self.cfg.proposal_network_config
            )
            self.prop_optim = parse_optimizer(
                self.cfg.prop_optimizer_config, self.prop_net
            )
            self.prop_scheduler = (
                parse_scheduler_to_instance(
                    self.cfg.prop_scheduler_config, self.prop_optim
                )
                if self.cfg.prop_scheduler_config is not None
                else None
            )
            self.estimator = nerfacc.PropNetEstimator(
                self.prop_optim, self.prop_scheduler
            )

            def get_proposal_requires_grad_fn(
                target: float = 5.0, num_steps: int = 1000
            ):
                schedule = lambda s: min(s / num_steps, 1.0) * target

                steps_since_last_grad = 0

                def proposal_requires_grad_fn(step: int) -> bool:
                    nonlocal steps_since_last_grad
                    target_steps_since_last_grad = schedule(step)
                    requires_grad = steps_since_last_grad > target_steps_since_last_grad
                    if requires_grad:
                        steps_since_last_grad = 0
                    steps_since_last_grad += 1
                    return requires_grad

                return proposal_requires_grad_fn

            self.proposal_requires_grad_fn = get_proposal_requires_grad_fn()
            self.randomized = self.cfg.randomized
        else:
            raise NotImplementedError(
                "Unknown estimator, should be one of ['occgrid', 'proposal', 'importance']."
            )

        # for proposal
        self.vars_in_forward = {}

    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        # import pdb; pdb.set_trace()
        batch_size, height, width = rays_o.shape[:3]
        rays_o_flatten: Float[Tensor, "Nr 3"] = rays_o.reshape(-1, 3)
        rays_d_flatten: Float[Tensor, "Nr 3"] = rays_d.reshape(-1, 3)
        light_positions_flatten: Float[Tensor, "Nr 3"] = (
            light_positions.reshape(-1, 1, 1, 3)
            .expand(-1, height, width, -1)
            .reshape(-1, 3)
        )
        n_rays = rays_o_flatten.shape[0]

        if self.cfg.estimator == "occgrid":
            if not self.cfg.grid_prune:
                with torch.no_grad():
                    ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                        rays_o_flatten,
                        rays_d_flatten,
                        sigma_fn=None,
                        near_plane=self.cfg.near_plane,
                        far_plane=self.cfg.far_plane,
                        render_step_size=self.render_step_size,
                        alpha_thre=0.0,
                        stratified=self.randomized,
                        cone_angle=0.0,
                        early_stop_eps=0,
                    )
            else:

                def sigma_fn(t_starts, t_ends, ray_indices):
                    t_starts, t_ends = t_starts[..., None], t_ends[..., None]
                    t_origins = rays_o_flatten[ray_indices]
                    t_positions = (t_starts + t_ends) / 2.0
                    t_dirs = rays_d_flatten[ray_indices]
                    positions = t_origins + t_dirs * t_positions

                    if self.training:
                        sigma = self.geometry.forward_density(positions)[..., 0]

                    else:
                        sigma = chunk_batch(
                            self.geometry.forward_density,
                            self.cfg.eval_chunk_size,
                            positions,
                        )[..., 0]
                    return sigma

                with torch.no_grad():
                    ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                        rays_o_flatten,
                        rays_d_flatten,
                        sigma_fn=sigma_fn if self.cfg.prune_alpha_threshold else None,
                        near_plane=self.cfg.near_plane,
                        far_plane=self.cfg.far_plane,
                        render_step_size=self.render_step_size,
                        alpha_thre=0.01 if self.cfg.prune_alpha_threshold else 0.0,
                        stratified=self.randomized,
                        cone_angle=0.0,
                    )

        elif self.cfg.estimator == "proposal":

            def prop_sigma_fn(
                t_starts: Float[Tensor, "Nr Ns"],
                t_ends: Float[Tensor, "Nr Ns"],
                proposal_network,
            ):
                t_origins: Float[Tensor, "Nr 1 3"] = rays_o_flatten.unsqueeze(-2)
                t_dirs: Float[Tensor, "Nr 1 3"] = rays_d_flatten.unsqueeze(-2)
                positions: Float[Tensor, "Nr Ns 3"] = (
                    t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
                )
                aabb_min, aabb_max = self.bbox[0], self.bbox[1]
                positions = (positions - aabb_min) / (aabb_max - aabb_min)
                selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
                density_before_activation = (
                    proposal_network(positions.view(-1, 3))
                    .view(*positions.shape[:-1], 1)
                    .to(positions)
                )
                density: Float[Tensor, "Nr Ns 1"] = (
                    get_activation("shifted_trunc_exp")(density_before_activation)
                    * selector[..., None]
                )
                return density.squeeze(-1)

            t_starts_, t_ends_ = self.estimator.sampling(
                prop_sigma_fns=[partial(prop_sigma_fn, proposal_network=self.prop_net)],
                prop_samples=[self.cfg.num_samples_per_ray_proposal],
                num_samples=self.cfg.num_samples_per_ray,
                n_rays=n_rays,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                sampling_type="uniform",
                stratified=self.randomized,
                requires_grad=self.vars_in_forward["requires_grad"],
            )
            ray_indices = (
                torch.arange(n_rays, device=rays_o_flatten.device)
                .unsqueeze(-1)
                .expand(-1, t_starts_.shape[1])
            )
            ray_indices = ray_indices.flatten()
            t_starts_ = t_starts_.flatten()
            t_ends_ = t_ends_.flatten()
        elif self.cfg.estimator == "importance":

            def prop_sigma_fn(
                t_starts: Float[Tensor, "Nr Ns"],
                t_ends: Float[Tensor, "Nr Ns"],
                proposal_network,
            ):
                t_origins: Float[Tensor, "Nr 1 3"] = rays_o_flatten.unsqueeze(-2)
                t_dirs: Float[Tensor, "Nr 1 3"] = rays_d_flatten.unsqueeze(-2)
                positions: Float[Tensor, "Nr Ns 3"] = (
                    t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
                )
                with torch.no_grad():
                    geo_out = chunk_batch(
                        proposal_network,
                        self.cfg.eval_chunk_size,
                        positions.reshape(-1, 3),
                        output_normal=False,
                    )
                    density = geo_out["density"]
                return density.reshape(positions.shape[:2])

            t_starts_, t_ends_ = self.estimator.sampling(
                prop_sigma_fns=[partial(prop_sigma_fn, proposal_network=self.geometry)],
                prop_samples=[self.cfg.num_samples_per_ray_importance],
                num_samples=self.cfg.num_samples_per_ray,
                n_rays=n_rays,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                sampling_type="uniform",
                stratified=self.randomized,
            )
            ray_indices = (
                torch.arange(n_rays, device=rays_o_flatten.device)
                .unsqueeze(-1)
                .expand(-1, t_starts_.shape[1])
            )
            ray_indices = ray_indices.flatten()
            t_starts_ = t_starts_.flatten()
            t_ends_ = t_ends_.flatten()
        else:
            raise NotImplementedError

        ray_indices, t_starts_, t_ends_ = validate_empty_rays(
            ray_indices, t_starts_, t_ends_
        )
        ray_indices = ray_indices.long()
        t_starts, t_ends = t_starts_[..., None], t_ends_[..., None]
        t_origins = rays_o_flatten[ray_indices]
        t_dirs = rays_d_flatten[ray_indices]
        t_light_positions = light_positions_flatten[ray_indices]
        t_positions = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * t_positions
        t_intervals = t_ends - t_starts

        if self.training:
            geo_out = self.geometry(
                positions, output_normal=self.material.requires_normal
            )
            rgb_fg_all = self.material(
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                **geo_out,
                **kwargs
            )
            comp_rgb_bg = self.background(dirs=rays_d)
        else:
            geo_out = chunk_batch(
                self.geometry,
                self.cfg.eval_chunk_size,
                positions,
                output_normal=self.material.requires_normal,
            )
            rgb_fg_all = chunk_batch(
                self.material,
                self.cfg.eval_chunk_size,
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                **geo_out
            )
            comp_rgb_bg = chunk_batch(
                self.background, self.cfg.eval_chunk_size, dirs=rays_d
            )

        weights: Float[Tensor, "Nr 1"]
        weights_, trans_, _ = nerfacc.render_weight_from_density(
            t_starts[..., 0],
            t_ends[..., 0],
            geo_out["density"][..., 0],
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        if self.training and self.cfg.estimator == "proposal":
            self.vars_in_forward["trans"] = trans_.reshape(n_rays, -1)

        weights = weights_[..., None]
        opacity: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=None, ray_indices=ray_indices, n_rays=n_rays
        )
        depth: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=t_positions, ray_indices=ray_indices, n_rays=n_rays
        )
        comp_rgb_fg: Float[Tensor, "Nr Nc"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=rgb_fg_all, ray_indices=ray_indices, n_rays=n_rays
        )

        # populate depth and opacity to each point
        weights_normalized = weights / opacity.clamp(min=1e-5)[ray_indices]  # num_pts
        # z-variance loss from HiFA: https://hifa-team.github.io/HiFA-site/
        z_mean: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights_normalized[..., 0],
            values=t_positions,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        z_variance_unmasked = nerfacc.accumulate_along_rays(
            weights_normalized[..., 0],
            values=(t_positions - z_mean[ray_indices]) ** 2,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        z_variance = z_variance_unmasked * (opacity > 0.5).float()

        if bg_color is None:
            bg_color = comp_rgb_bg
        else:
            if bg_color.shape[:-1] == (batch_size,):
                # e.g. constant random color used for Zero123
                # [bs,3] -> [bs, 1, 1, 3]):
                bg_color = bg_color.unsqueeze(1).unsqueeze(1)
                #        -> [bs, height, width, 3]):
                bg_color = bg_color.expand(-1, height, width, -1)

        if bg_color.shape[:-1] == (batch_size, height, width):
            bg_color = bg_color.reshape(batch_size * height * width, -1)

        comp_rgb = comp_rgb_fg + bg_color * (1.0 - opacity)

        out = {
            "comp_rgb": comp_rgb.view(batch_size, height, width, -1),
            "comp_rgb_fg": comp_rgb_fg.view(batch_size, height, width, -1),
            "comp_rgb_bg": comp_rgb_bg.view(batch_size, height, width, -1),
            "opacity": opacity.view(batch_size, height, width, 1),
            "depth": depth.view(batch_size, height, width, 1),
            "z_variance": z_variance.view(batch_size, height, width, 1),
        }

        if self.training:
            out.update(
                {
                    "weights": weights,
                    "t_points": t_positions,
                    "t_intervals": t_intervals,
                    "t_dirs": t_dirs,
                    "ray_indices": ray_indices,
                    "points": positions,
                    **geo_out,
                }
            )
            if "normal" in geo_out:
                if self.cfg.return_comp_normal:
                    comp_normal: Float[Tensor, "Nr 3"] = nerfacc.accumulate_along_rays(
                        weights[..., 0],
                        values=geo_out["normal"],
                        ray_indices=ray_indices,
                        n_rays=n_rays,
                    )
                    comp_normal = F.normalize(comp_normal, dim=-1)
                    comp_normal = (
                        (comp_normal + 1.0) / 2.0 * opacity
                    )  # for visualization
                    out.update(
                        {
                            "comp_normal": comp_normal.view(
                                batch_size, height, width, 3
                            ),
                        }
                    )
                if self.cfg.return_normal_perturb:
                    normal_perturb = self.geometry(
                        positions + torch.randn_like(positions) * 1e-2,
                        output_normal=self.material.requires_normal,
                    )["normal"]
                    out.update({"normal_perturb": normal_perturb})
        else:
            if "normal" in geo_out:
                comp_normal = nerfacc.accumulate_along_rays(
                    weights[..., 0],
                    values=geo_out["normal"],
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
                comp_normal = F.normalize(comp_normal, dim=-1)
                comp_normal = (comp_normal + 1.0) / 2.0 * opacity  # for visualization
                out.update(
                    {
                        "comp_normal": comp_normal.view(batch_size, height, width, 3),
                    }
                )

        return out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        if self.cfg.estimator == "occgrid":
            if self.cfg.grid_prune:

                def occ_eval_fn(x):
                    density = self.geometry.forward_density(x)
                    # approximate for 1 - torch.exp(-density * self.render_step_size) based on taylor series
                    return density * self.render_step_size

                if self.training and not on_load_weights:
                    self.estimator.update_every_n_steps(
                        step=global_step, occ_eval_fn=occ_eval_fn
                    )
        elif self.cfg.estimator == "proposal":
            if self.training:
                requires_grad = self.proposal_requires_grad_fn(global_step)
                self.vars_in_forward["requires_grad"] = requires_grad
            else:
                self.vars_in_forward["requires_grad"] = False

    def update_step_end(self, epoch: int, global_step: int) -> None:
        if self.cfg.estimator == "proposal" and self.training:
            self.estimator.update_every_n_steps(
                self.vars_in_forward["trans"],
                self.vars_in_forward["requires_grad"],
                loss_scaler=1.0,
            )

    def train(self, mode=True):
        self.randomized = mode and self.cfg.randomized
        if self.cfg.estimator == "proposal":
            self.prop_net.train()
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        if self.cfg.estimator == "proposal":
            self.prop_net.eval()
        return super().eval()
