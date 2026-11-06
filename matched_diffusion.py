"""CVCI2 gate12 clean-delta planner with the retained rolling future condition."""
import torch
from torch import nn
import torch.nn.functional as F
from cvci2_gate12.runtime import configure_attention_backend
from cvci2_gate12.baseline_transfuser import SinusoidalTime
from cvci2_gate12.order_model import AgentEgoSpatialLayer
from cvci2_gate12.xpred_canbus import XPredCanbusPlanner, ddim_xpred_update

from motion_gate import canonical_motion_gate_config, zero_motion_projection, MOTION_MONITOR_KEYS, INTEGRATED_MOTION_GATE_CONFIG

DEFAULT_FUSION_CONFIG = dict(future_dropout=.1)

def sampler_monitor_keys():
    keys=['sampler/valid_tokens_per_future_sample','sampler/token_norm']
    for h in range(5):
        keys += [f'sampler/{metric}/h{h+1}' for metric in
            ('valid_fraction','abs_tangent_m','abs_normal_m','offset_saturation','template_drift_m','window_tangent_m','window_normal_m')]
    return keys

def _stat(total, count):
    count = torch.as_tensor(count, device=total.device, dtype=total.dtype)
    return torch.stack((total.detach(), count.detach()))

def attention_monitor_keys():
    keys = ['attention/normalized_entropy', 'attention/same_bev_time_mass',
            'attention/same_waypoint_mass', 'fusion/point_gate', 'fusion/residual_ratio']
    keys += [f'attention/mass_bev/h{h+1}' for h in range(6)]
    keys += [f'attention/mass_reference/j{j+1}' for j in range(6)]
    keys += [f'attention/reference_bev/j{j+1}_h{h+1}' for j in range(6) for h in range(6)]
    keys += [f'fusion/global_gate/layer_{i}' for i in range(1, 12, 2)]
    return keys

def base_monitoring_keys():
    keys = sampler_monitor_keys() + attention_monitor_keys()
    keys += ['planning/outside_window_fraction']
    for h in range(6):
        keys += [f'planning/prior_gap_m/h{h+1}']
    for sec in (1, 2, 3):
        keys += [f'planning/train_batch_ddim_l2_m/{sec}s', f'planning/train_batch_prior_l2_m/{sec}s']
    keys += [f'timing/{name}_seconds' for name in
             ('conditions', 'world', 'future_encoder', 'diffusion_loss', 'ddim_sample')]
    keys += [f'gradient/{name}_norm' for name in ('dit', 'offset', 'pointwise_encoder', 'perception', 'spatial_gate', 'world_tuned')]
    keys += ['timing/data_loader_host_seconds']
    keys += list(MOTION_MONITOR_KEYS)
    from candidate_interaction import monitor_keys
    return sorted(keys + monitor_keys())

def prepare_future(condition, future_mask, heads):
    if condition is None:
        return None
    tokens, valid, log_prior = (condition[k] for k in ('tokens', 'valid', 'log_prior'))
    b, n, _ = tokens.shape
    if valid.shape != (b, n) or log_prior.shape != (b, n) or n < 1:
        raise ValueError('invalid sampled condition shapes')
    any_valid = valid.any(1)
    active = any_valid if future_mask is None else any_valid & future_mask.bool()
    safe_valid = valid.clone()
    safe_valid[:, 0] |= ~any_valid
    safe_tokens = tokens.masked_fill(~valid[..., None], 0)
    bias = log_prior.masked_fill(~valid, 0).masked_fill(~safe_valid, float('-inf'))
    # The dummy zero key only prevents an all-masked softmax. Its residual is off.
    bias = bias[:, None, None].expand(b, heads, 6, n).reshape(b*heads, 6, n)
    result = dict(tokens=safe_tokens, bias=bias, active=active, valid=valid,
                  map_ids=condition['map_ids'], waypoint_ids=condition['waypoint_ids'])
    from candidate_interaction import GEOMETRY_FIELDS
    result.update({key: condition[key] for key in GEOMETRY_FIELDS if key in condition})
    return result


class MatchedDiTLayer(AgentEgoSpatialLayer):
    def __init__(self, motion_gate_config=None):
        super().__init__(True)
        self.future_norm = nn.LayerNorm(256)
        self.future_attn = nn.MultiheadAttention(256, 8, batch_first=True)
        self.future_gate = nn.Parameter(torch.zeros(()))
        self.future_query_geometry = nn.Sequential(nn.Linear(6, 256), nn.SiLU(), nn.Linear(256, 256))
        self.motion_gate_config = canonical_motion_gate_config(motion_gate_config)
        self.integrated_motion = self.motion_gate_config == INTEGRATED_MOTION_GATE_CONFIG
        self.future_reliability = nn.Sequential(nn.Linear(282 if self.integrated_motion else 262, 128), nn.SiLU(), nn.Linear(128, 1))
        nn.init.zeros_(self.future_reliability[-1].weight)
        nn.init.zeros_(self.future_reliability[-1].bias)
        self.last_point_gate = None
        self.last_residual_ratio = None
        self.motion_gate_config = canonical_motion_gate_config(motion_gate_config)
        self.future_motion_projection = (zero_motion_projection()
            if self.motion_gate_config is not None and not self.integrated_motion else None)

        self.last_stats = {}

    def read_current(self, hidden, positions, dense_bev, monitor=False):
        grid = torch.stack((positions[..., 1]/32., positions[..., 0]/32.), -1)
        sampled = F.grid_sample(self.bev_value(dense_bev).float(), grid[:, :, None].float(),
                                 mode='bilinear', padding_mode='zeros', align_corners=False).to(hidden.dtype)
        return sampled.squeeze(-1).transpose(1, 2)

    def future_context(self, future, positions, prior_positions, monitor=False):
        return future

    def reliability_context(self, inputs, future):
        return inputs

    def forward(self, hidden, positions, dense_bev, agents, ego, time, status,
                future=None, prior_positions=None, diffusion_fraction=None, monitor=False, motion_features=None):
        if monitor:
            self.last_stats = {}
        hidden = self.norms[1](hidden + self.agent_attn(hidden, agents, agents, need_weights=False)[0])
        hidden = self.norms[2](hidden + self.ego_attn(hidden, ego, ego, need_weights=False)[0])
        sampled = self.read_current(hidden, positions, dense_bev, monitor)
        hidden = self.norms[0](hidden + sampled)
        return self._finish_interaction(hidden, positions, time, status, future,
            prior_positions, diffusion_fraction, monitor, motion_features)

    def _finish_interaction(self, hidden, positions, time, status, future,
                            prior_positions, diffusion_fraction, monitor, motion_features):
        if future is not None:
            future = self.future_context(future, positions, prior_positions, monitor)
            horizon = torch.arange(1, 7, device=hidden.device, dtype=hidden.dtype)[None, :, None]/6
            geometry = torch.cat(((prior_positions/32).clamp(-4, 4),
                                  ((positions-prior_positions)/32).clamp(-4, 4),
                                  horizon.expand(hidden.shape[0], -1, -1),
                                  diffusion_fraction[:, None, None].expand(-1, 6, -1)), -1)
            normalized = self.future_norm(hidden)
            query = self.future_norm(hidden + self.future_query_geometry(geometry))
            # Attention scores follow the projection dtype under CUDA autocast.
            bias_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else query.dtype
            attention_bias = future['bias'].to(bias_dtype)
            update, _ = self.future_attn(query, future['tokens'], future['tokens'],
                                         attn_mask=attention_bias, need_weights=False)
            if monitor:
                with torch.no_grad():
                    _, weights = self.future_attn(query.detach(), future['tokens'].detach(),
                        future['tokens'].detach(), attn_mask=attention_bias, need_weights=True)
            gate_input = torch.cat((normalized, geometry), -1)
            projection = None
            if self.integrated_motion:
                if motion_features is None or motion_features.shape != (*hidden.shape[:2], 20):
                    raise ValueError('Integrated gate requires observed [B,6,20] motion')
                gate_input = torch.cat((gate_input, motion_features.to(gate_input)), -1)
                point_gate = self.future_reliability(self.reliability_context(gate_input, future)).sigmoid()
            elif self.future_motion_projection is None:
                point_gate = self.future_reliability(gate_input).sigmoid()
            else:
                if motion_features is None or motion_features.shape != (*hidden.shape[:2], 20):
                    raise ValueError('Enabled motion gate requires [B,6,20] observed inputs')
                original = self.future_reliability[0](gate_input)
                projection = self.future_motion_projection(motion_features.to(original))
                point_gate = self.future_reliability[2](
                    self.future_reliability[1](original + projection)).sigmoid()
            contribution = self.future_gate.tanh()*point_gate*F.layer_norm(update, (256,))
            active = future['active'][:, None, None].to(hidden)
            contribution = contribution*active
            ratio = contribution.norm(dim=-1)/hidden.norm(dim=-1).clamp_min(1e-6)
            self.last_point_gate, self.last_residual_ratio = point_gate.detach(), ratio.detach()
            if monitor:
                with torch.no_grad():
                    frame = future['active'].to(hidden)
                    count = frame.sum()*6
                    self.last_stats['fusion/point_gate'] = _stat((point_gate[..., 0]*frame[:, None]).sum(), count)
                    self.last_stats['fusion/residual_ratio'] = _stat((ratio*frame[:, None]).sum(), count)
                    if self.motion_gate_config is not None:
                        observed = future['active'] & (motion_features[:, 0, -1] > 0)
                        speed = (motion_features[:, 0, 4:6]*15.).norm(dim=-1)
                        increasing = motion_features[:, 0, 16]*5. > .5
                        for label, group in [('low_speed', speed <= .5),
                                             ('increasing_speed', increasing), ('moving', speed >= 5.)]:
                            group_weight = (observed & group).to(point_gate)
                            self.last_stats['motion/gate_'+label] = _stat(
                                (point_gate[..., 0]*group_weight[:, None]).sum(), group_weight.sum()*6)
                        if projection is None:
                            projection = motion_features.to(normalized)
                        weight = observed.to(projection)
                        self.last_stats['motion/projection_norm'] = _stat(
                            (projection.norm(dim=-1)*weight[:, None]).sum(), weight.sum()*6)
                    entropy = -(weights.clamp_min(1e-12).log()*weights).sum(-1)
                    entropy = entropy/future['valid'].sum(1).clamp_min(2).to(hidden).log()[:, None]
                    self.last_stats['attention/normalized_entropy'] = _stat((entropy*frame[:, None]).sum(), count)
                    query_id = torch.arange(6, device=hidden.device)
                    for name, ids in (('same_bev_time_mass', future['map_ids']), ('same_waypoint_mass', future['waypoint_ids'])):
                        match = (query_id[:, None] == ids[None]).to(weights)
                        self.last_stats[f'attention/{name}'] = _stat((weights*match[None]*frame[:, None, None]).sum(), count)
                    for h in range(6):
                        select_h = future['map_ids'] == h
                        self.last_stats[f'attention/mass_bev/h{h+1}'] = _stat((weights[..., select_h]*frame[:, None, None]).sum(), count)
                    for j in range(6):
                        select_j = future['waypoint_ids'] == j
                        self.last_stats[f'attention/mass_reference/j{j+1}'] = _stat((weights[..., select_j]*frame[:, None, None]).sum(), count)
                        for h in range(6):
                            select = select_j & (future['map_ids'] == h)
                            self.last_stats[f'attention/reference_bev/j{j+1}_h{h+1}'] = _stat((weights[..., select]*frame[:, None, None]).sum(), count)
            hidden = hidden + contribution
        hidden = self.norms[3](hidden + self.ffn(hidden))
        scale, shift = self.modulation(torch.cat((time, status), -1)).chunk(2, -1)
        return hidden*(1+scale)+shift


class MatchedX0Planner(XPredCanbusPlanner):
    """Twelve equally supervised x0 blocks; extra future residuals start at zero."""
    def __init__(self, base_planner=None, motion_gate_config=None):
        backend = configure_attention_backend()
        super().__init__()
        self.attention_backend = backend
        self.motion_gate_config = canonical_motion_gate_config(motion_gate_config)
        self.layers = nn.ModuleList(list(self.layers)[:12])
        for index, layer in enumerate(self.layers):
            if layer.interaction:
                replacement = MatchedDiTLayer(self.motion_gate_config)
                replacement.load_state_dict(layer.state_dict(), strict=False)
                self.layers[index] = replacement
        if base_planner is not None:
            missing, unexpected = self.load_state_dict(base_planner.state_dict(), strict=False)
            if unexpected or any('.future_' not in name for name in missing):
                raise RuntimeError(f'gate12 initialization mismatch: {missing}, {unexpected}')
        self.monitor = False

    def clip_normalized(self, value):
        return value.clamp(-1, 1)

    def ddim_update(self, value, prediction, alpha, next_alpha):
        return ddim_xpred_update(value, prediction, alpha, next_alpha)

    def extend_loss_terms(self, terms, predictions, delta, mask, timestep):
        return terms

    def predict(self, noisy, timestep, ego, agents, dense, status,
                future_tokens=None, future_mask=None, future_prior_positions=None, canbus=None, motion_features=None,
                record_candidates=False):
        if canbus is None:
            raise ValueError('The separate CAN BUS embedding is required')
        time_embed = self.time(timestep)[:, None]
        time_action = SinusoidalTime(256).to(noisy.device)(timestep)[:, None].expand(-1, 6, -1)
        hidden = self.action_fuse(torch.cat((self.action(noisy.float()), time_action), -1))
        hidden = self.canbus_concat(torch.cat((hidden, canbus.expand(-1, 6, -1)), -1))
        hidden = hidden + self.position.weight[None] + self.ego_norm(ego).expand(-1, 6, -1)
        with torch.autocast(device_type=noisy.device.type, enabled=False):
            positions = self.denormalize(self.clip_normalized(noisy.float())).cumsum(1)
        prior = torch.zeros_like(positions) if future_prior_positions is None else future_prior_positions
        if prior.shape != positions.shape:
            raise ValueError('future prior must be [B,6,2]')
        condition = prepare_future(future_tokens, future_mask, 8)
        outputs = []
        previous_positions = positions
        candidate = getattr(self, 'candidate_config', None)
        if record_candidates:
            self.last_training_candidate_positions = None
        for layer in self.layers:
            layer_condition = condition
            if candidate is not None and condition is not None:
                layer_condition = dict(condition, candidate_previous_positions=previous_positions)
            if isinstance(layer, MatchedDiTLayer):
                if record_candidates and self.last_training_candidate_positions is None:
                    self.last_training_candidate_positions = positions.detach()
                hidden = layer(hidden, positions, dense, agents, ego, time_embed, status,
                               layer_condition, prior, timestep.to(hidden.dtype)/99, self.monitor, motion_features)
            else:
                hidden = layer(hidden, positions, dense, agents, ego, time_embed, status)
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                outputs.append(self.head(hidden.float()))
                if candidate is not None and candidate['dynamic_positions']:
                    previous_positions = positions
                    positions = self.denormalize(self.clip_normalized(outputs[-1].detach().float())).cumsum(1)
        return outputs

    def loss(self, delta, mask, ego, agents, dense, status, future_tokens=None, future_mask=None,
             future_prior_positions=None, return_details=False, canbus=None, motion_features=None):
        with torch.autocast(device_type=delta.device.type, enabled=False):
            clean = self.normalize(delta.float())
            timestep = torch.randint(0, self.train_timesteps, (delta.shape[0],), device=delta.device)
            fraction = float(getattr(self, 'training_t99_fraction', 0.))
            if fraction:
                selected = torch.rand(delta.shape[0], device=delta.device) < fraction
                timestep = torch.where(selected, torch.full_like(timestep, self.train_timesteps-1), timestep)
            noise = torch.randn_like(clean)
            alpha = self.alpha[timestep].float().view(-1, 1, 1)
            noisy = alpha.sqrt()*clean + (1-alpha).sqrt()*noise
        predictions = self.predict(noisy, timestep, ego, agents, dense, status,
                                   future_tokens, future_mask, future_prior_positions, canbus=canbus, motion_features=motion_features,
                                   record_candidates=bool(getattr(self, 'loss_diagnostics_enabled', False)))
        with torch.autocast(device_type=delta.device.type, enabled=False):
            valid = mask[..., None].float()
            denominator = (2*valid.sum()).clamp_min(1)
            layer_losses = torch.stack([((p.float()-clean).square()*valid).sum()/denominator
                                        for p in predictions])
            unweighted = layer_losses.mean()
            weights = getattr(self, 'layer_loss_weights', None)
            x0 = unweighted if weights is None else (layer_losses * layer_losses.new_tensor(weights)).sum()
            terms = dict(total=x0, x0=x0, x0_unweighted=unweighted, diffusion=x0, layer_losses=layer_losses)
            terms = self.extend_loss_terms(terms, predictions, delta, mask, timestep)
            if getattr(self, 'loss_diagnostics_enabled', False):
                from training_optimization import loss_diagnostics
                terms['diagnostics'] = loss_diagnostics(predictions, clean, valid, timestep,
                    denormalize=self.denormalize, target_positions=delta.float().cumsum(1),
                    candidate_positions=self.last_training_candidate_positions,
                    future=future_tokens, future_mask=future_mask, clip_normalized=self.clip_normalized)
        return terms if return_details else terms['total']

    @torch.no_grad()
    def sample(self, batch, ego, agents, dense, status, inference_steps=1,
               future_tokens=None, future_mask=None, initial_noise=None, timestep_indices=None,
               future_prior_positions=None, canbus=None, motion_features=None):
        if not 1 <= inference_steps <= self.train_timesteps:
            raise ValueError('DDIM steps must be in [1,100]')
        with torch.autocast(device_type=dense.device.type, enabled=False):
            value = (torch.randn(batch, self.steps, 2, device=dense.device, dtype=torch.float32)
                     if initial_noise is None else initial_noise.to(device=dense.device, dtype=torch.float32).clone())
            if value.shape != (batch, self.steps, 2):
                raise ValueError('initial noise must be [B,6,2]')
            indices = (torch.linspace(99, 0, inference_steps, device=dense.device).round().long()
                       if timestep_indices is None else torch.as_tensor(timestep_indices, device=dense.device, dtype=torch.long))
            if (indices.ndim != 1 or len(indices) != inference_steps or bool((indices < 0).any()) or
                    bool((indices >= 100).any()) or bool((indices[1:] >= indices[:-1]).any())):
                raise ValueError('invalid DDIM indices')
        for index, step in enumerate(indices):
            predictions = self.predict(value, step.expand(batch), ego, agents, dense, status,
                                      future_tokens, future_mask, future_prior_positions, canbus=canbus, motion_features=motion_features)
            prediction = predictions[-1]
            if getattr(self, 'collect_sample_layer_diagnostics', False):
                self.last_sample_layer_predictions = [p.detach() for p in predictions]
            with torch.autocast(device_type=dense.device.type, enabled=False):
                if index+1 == len(indices):
                    value = self.clip_normalized(prediction.float())
                else:
                    value = self.ddim_update(value, prediction, self.alpha[step], self.alpha[indices[index+1]])
        return self.denormalize(value.float()).cumsum(1)
