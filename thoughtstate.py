import math
import os
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    state_dim: int = 256
    num_layers: int = 6
    max_context_horizon: int = 256
    dropout: float = 0.05
    dt: float = 0.05


class ComplexLinear(nn.Module):
    bias_r: Optional[torch.Tensor]
    bias_i: Optional[torch.Tensor]

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_r = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_i = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias_r = nn.Parameter(torch.zeros(out_features))
            self.bias_i = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias_r = None
            self.bias_i = None

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(2.0 * self.in_features)
        nn.init.normal_(self.weight_r, std=std)
        nn.init.normal_(self.weight_i, std=std)

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out_r = F.linear(x_r, self.weight_r) - F.linear(x_i, self.weight_i)
        out_i = F.linear(x_i, self.weight_r) + F.linear(x_r, self.weight_i)

        b_r = self.bias_r
        b_i = self.bias_i
        if b_r is not None and b_i is not None:
            out_r = out_r + b_r
            out_i = out_i + b_i

        return out_r, out_i


class ComplexRMSNorm(nn.Module):
    def __init__(self, state_dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(state_dim))

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ms = torch.mean(x_r.pow(2) + x_i.pow(2), dim=-1, keepdim=True)
        scale = self.gamma / torch.sqrt(ms + self.eps)
        return x_r * scale, x_i * scale


def get_clock_freqs(
    state_dim: int,
    layer_idx: int,
    dt: float = 0.05,
    max_theta_deg: float = 45.0,
    min_theta_deg: float = 0.5,
) -> torch.Tensor:
    max_freq = math.radians(max_theta_deg) / dt
    min_freq = math.radians(min_theta_deg) / dt

    log_max, log_min = math.log(max_freq), math.log(min_freq)
    freq_scaling = torch.exp(torch.linspace(log_max, log_min, steps=state_dim))
    layer_scale = 1.0 / (1.0 + 0.5 * layer_idx)
    return freq_scaling * layer_scale


class SwiGLUFFN(nn.Module):
    def __init__(self, state_dim: int, expansion_factor: int = 2):
        super().__init__()
        hidden_dim = state_dim * expansion_factor
        self.w_gate = ComplexLinear(state_dim, hidden_dim)
        self.w_up   = ComplexLinear(state_dim, hidden_dim)
        self.w_down = ComplexLinear(hidden_dim, state_dim)

    def forward(self, phi_r: torch.Tensor, phi_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        g_r, g_i = self.w_gate(phi_r, phi_i)
        u_r, u_i = self.w_up(phi_r, phi_i)

        energy_gate = F.silu(g_r.pow(2) + g_i.pow(2))
        
        h_r = energy_gate * u_r
        h_i = energy_gate * u_i

        return self.w_down(h_r, h_i)


class SwiGLUPotentialOperator(nn.Module):
    def __init__(self, state_dim: int, expansion: int = 2):
        super().__init__()
        hidden_dim = state_dim * expansion
        self.w_gate = nn.Linear(state_dim, hidden_dim)
        self.w_up   = nn.Linear(state_dim, hidden_dim)
        self.w_down = nn.Linear(hidden_dim, state_dim)

        nn.init.normal_(self.w_down.weight, std=0.02)
        nn.init.zeros_(self.w_down.bias)

    def forward(self, density: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.w_gate(density)) * self.w_up(density)
        return self.w_down(gated)


class HamiltonianOperator(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.T_operator = nn.Linear(state_dim, state_dim, bias=False)
        nn.init.orthogonal_(self.T_operator.weight, gain=0.5)

        self.V_operator = SwiGLUPotentialOperator(state_dim)

    def forward(self, phi_r: torch.Tensor, phi_i: torch.Tensor) -> torch.Tensor:
        t_r = self.T_operator(phi_r)
        t_i = self.T_operator(phi_i)
        
        mag_sq = phi_r.pow(2) + phi_i.pow(2)
        v = self.V_operator(mag_sq)
        
        v_term_r = v * phi_r
        v_term_i = v * phi_i
        
        h_phi_r = t_r + v_term_r
        h_phi_i = t_i + v_term_i

        raw_energy = phi_r * h_phi_r + phi_i * h_phi_i
        return torch.tanh(raw_energy)


class UncertaintyGate(nn.Module):
    def __init__(self, state_dim: int, max_w_phase: float = 2.0, max_w_mag: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.max_w_phase = max_w_phase
        self.max_w_mag = max_w_mag

        init_phase_val = math.atanh(2.0 * 0.8 / max_w_phase - 1.0)
        init_mag_val   = math.atanh(2.0 * 0.2 / max_w_mag - 1.0)

        self.w_phase = nn.Parameter(torch.full((state_dim,), init_phase_val))
        self.w_mag   = nn.Parameter(torch.full((state_dim,), init_mag_val))

    def forward(
        self, 
        phi_r: torch.Tensor, phi_i: torch.Tensor, 
        psi_r: torch.Tensor, psi_i: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        mag_phi = torch.hypot(phi_r, phi_i)
        mag_psi = torch.hypot(psi_r, psi_i)

        dot_product = phi_r * psi_r + phi_i * psi_i
        mag_prod = mag_phi * mag_psi + self.eps
        phase_cos = torch.clamp(dot_product / mag_prod, min=-1.0, max=1.0)
        u_phase = 1.0 - phase_cos

        diff = mag_phi - mag_psi
        sum_mag = mag_phi + mag_psi + self.eps
        u_mag = (diff / sum_mag).pow(2)

        w_p = (self.max_w_phase / 2.0) * (1.0 + torch.tanh(self.w_phase))
        w_m = (self.max_w_mag / 2.0) * (1.0 + torch.tanh(self.w_mag))

        uncertainty = w_p * u_phase + w_m * u_mag

        a_gate = 1.0 / torch.sqrt(1.0 + uncertainty.pow(2))
        b_gate = 1.0 - a_gate

        return uncertainty, a_gate, b_gate


class WavePropagator(nn.Module):
    def __init__(self, state_dim: int, layer_idx: int = 0, dt: float = 0.05):
        super().__init__()
        self.dt = dt
        self.decay = 0.99
        self.rms_norm = ComplexRMSNorm(state_dim)
        self.register_buffer("clock_freqs", get_clock_freqs(state_dim, layer_idx, dt))

        self.gate_module = UncertaintyGate(state_dim)
        self.hamiltonian = HamiltonianOperator(state_dim)

    def forward(
        self, 
        phi_r: torch.Tensor, phi_i: torch.Tensor, 
        psi_r: torch.Tensor, psi_i: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        uncertainty, a_gate, b_gate = self.gate_module(phi_r, phi_i, psi_r, psi_i)

        h_energy = self.hamiltonian(phi_r, phi_i)
        theta = (self.clock_freqs + h_energy) * self.dt
        
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        phi_rot_r = (phi_r * cos_t - phi_i * sin_t) * self.decay
        phi_rot_i = (phi_r * sin_t + phi_i * cos_t) * self.decay

        next_r = a_gate * phi_rot_r + b_gate * psi_r
        next_i = a_gate * phi_rot_i + b_gate * psi_i
        
        phi_next_r, phi_next_i = self.rms_norm(next_r, next_i)
        return (phi_next_r, phi_next_i), uncertainty, a_gate


class HamiltonianSuperpositionLayer(nn.Module):
    def __init__(self, state_dim: int, layer_idx: int = 0, dt: float = 0.05):
        super().__init__()
        self.state_dim = state_dim
        self.rms_norm = ComplexRMSNorm(state_dim)
        self.propagator = WavePropagator(state_dim, layer_idx, dt)
        self.interfeature = ComplexLinear(state_dim, state_dim, bias=False)

    def forward(
        self,
        psi_r: torch.Tensor,
        psi_i: torch.Tensor,
        prev_state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        
        batch_size, seq_len, _ = psi_r.shape
        psi_r, psi_i = self.rms_norm(psi_r, psi_i)

        phi_r, phi_i = prev_state

        out_list_r: List[torch.Tensor] = []
        out_list_i: List[torch.Tensor] = []
        unc_list: List[torch.Tensor] = []
        gate_list: List[torch.Tensor] = []

        for t in range(seq_len):
            (phi_r, phi_i), unc_t, gate_t = self.propagator(
                phi_r, phi_i, psi_r[:, t, :], psi_i[:, t, :]
            )
            out_list_r.append(phi_r)
            out_list_i.append(phi_i)
            unc_list.append(unc_t)
            gate_list.append(gate_t)

        out_seq_r = torch.stack(out_list_r, dim=1)
        out_seq_i = torch.stack(out_list_i, dim=1)
        
        seq_unc = torch.stack(unc_list, dim=1)
        seq_gate = torch.stack(gate_list, dim=1)
        
        out_seq_r, out_seq_i = self.interfeature(out_seq_r, out_seq_i)

        return (out_seq_r, out_seq_i), (phi_r, phi_i), seq_unc, seq_gate


class ThoughtStateBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.h_layer = HamiltonianSuperpositionLayer(config.state_dim, layer_idx=layer_idx, dt=config.dt)
        self.ffn_norm = ComplexRMSNorm(config.state_dim)
        self.ffn_layer = SwiGLUFFN(config.state_dim)

    def forward(
        self,
        psi_r: torch.Tensor,
        psi_i: torch.Tensor,
        prev_state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        res_r, res_i = psi_r, psi_i
        
        # 1. Recurrent Wave Propagation
        (out_r, out_i), next_st, seq_unc, seq_gate = self.h_layer(psi_r, psi_i, prev_state=prev_state)
        
        # 2. First residual connection
        mix_r, mix_i = res_r + out_r, res_i + out_i
        
        # 3. Pre-Norm
        norm_r, norm_i = self.ffn_norm(mix_r, mix_i)
        
        # 4. FFN mapping
        ffn_r, ffn_i = self.ffn_layer(norm_r, norm_i)
        
        # 5. Second residual connection
        psi_out_r, psi_out_i = mix_r + ffn_r, mix_i + ffn_i
        
        return psi_out_r, psi_out_i, next_st, seq_unc, seq_gate


class ThoughtStateNetwork(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.state_dim = config.state_dim

        self.word_embed_real = nn.Embedding(config.vocab_size, config.state_dim)
        self.word_embed_imag = nn.Embedding(config.vocab_size, config.state_dim)
        self.embed_drop = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            ThoughtStateBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        self.collapse = ComplexLinear(config.state_dim, config.vocab_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor], List[torch.Tensor]]:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if states is None:
            batch_size = x.size(0)
            active_states: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for _ in range(len(self.layers)):
                z_r = torch.zeros(batch_size, self.state_dim, device=x.device)
                z_i = torch.zeros(batch_size, self.state_dim, device=x.device)
                active_states.append((z_r, z_i))
        else:
            active_states = states

        psi_r = self.embed_drop(self.word_embed_real(x))
        psi_i = self.embed_drop(self.word_embed_imag(x))

        next_states: List[Tuple[torch.Tensor, torch.Tensor]] = []
        layer_uncertainties: List[torch.Tensor] = []
        layer_gates: List[torch.Tensor] = []

        for idx, layer in enumerate(self.layers):
            psi_r, psi_i, next_st, seq_unc, seq_gate = layer(
                psi_r, psi_i, prev_state=active_states[idx]
            )
            next_states.append(next_st)
            layer_uncertainties.append(seq_unc)
            layer_gates.append(seq_gate)

        amps_r, amps_i = self.collapse(psi_r, psi_i)
        
        amps_sq = amps_r.pow(2) + amps_i.pow(2) + 1e-6 
        logits = torch.log(amps_sq)

        return logits, next_states, layer_uncertainties, layer_gates
