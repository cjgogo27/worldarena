from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 128,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(hidden_dim * 4 * 8 * 8, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(batch, -1)
        return self.fc(x)


class FrameDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 64,
        out_channels: int = 1,
    ):
        super().__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim * 4 * 8 * 8)
        self.deconv1 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.shape[0]
        x = self.fc(z)
        x = x.reshape(batch, -1, 8, 8)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        return self.deconv3(x)


class VideoPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_size: int = 256,
        latent_dim: int = 128,
        num_layers: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.encoder = FrameEncoder(in_channels=in_channels, hidden_dim=64, output_dim=latent_dim)
        self.decoder = FrameDecoder(latent_dim=latent_dim, hidden_dim=64, out_channels=out_channels)
        self.gru = nn.GRU(input_size=latent_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.hidden_to_latent = nn.Linear(hidden_size, latent_dim)

    def _init_zero_hidden(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype)

    def forward(
        self,
        frames: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _, _, _ = frames.shape
        hidden_state = hidden if hidden is not None else self._init_zero_hidden(batch, frames.device, frames.dtype)
        outputs: list[torch.Tensor] = []
        for step in range(steps):
            encoded = self.encoder(frames[:, step])
            _, hidden_state = self.gru(encoded.unsqueeze(1), hidden_state)
            outputs.append(self.decoder(self.hidden_to_latent(hidden_state[-1])))
        return torch.stack(outputs, dim=1), hidden_state

    def forward_with_teacher_forcing(
        self,
        frames: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _, _, _ = frames.shape
        hidden = self._init_zero_hidden(batch, frames.device, frames.dtype)
        current = frames[:, 0]
        outputs: list[torch.Tensor] = []
        for step in range(steps):
            encoded = self.encoder(current)
            _, hidden = self.gru(encoded.unsqueeze(1), hidden)
            recon = self.decoder(self.hidden_to_latent(hidden[-1]))
            outputs.append(recon)
            if step < steps - 1:
                use_teacher = torch.rand(1, device=frames.device).item() < teacher_forcing_ratio
                current = frames[:, step + 1] if use_teacher else recon.detach()
        return torch.stack(outputs, dim=1), hidden

    def rollout(self, initial_frames: torch.Tensor, num_steps: int) -> torch.Tensor:
        batch = initial_frames.shape[0]
        hidden = self._init_zero_hidden(batch, initial_frames.device, initial_frames.dtype)
        for step in range(initial_frames.shape[1]):
            encoded = self.encoder(initial_frames[:, step])
            _, hidden = self.gru(encoded.unsqueeze(1), hidden)
        generated: list[torch.Tensor] = []
        for _ in range(num_steps):
            recon = self.decoder(self.hidden_to_latent(hidden[-1]))
            generated.append(recon)
            encoded = self.encoder(recon)
            _, hidden = self.gru(encoded.unsqueeze(1), hidden)
        return torch.stack(generated, dim=1)

    def get_hidden_states(self, frames: torch.Tensor, return_all_layers: bool = False) -> torch.Tensor:
        batch, steps, _, _, _ = frames.shape
        hidden = self._init_zero_hidden(batch, frames.device, frames.dtype)
        all_states: list[torch.Tensor] = []
        for step in range(steps):
            encoded = self.encoder(frames[:, step])
            _, hidden = self.gru(encoded.unsqueeze(1), hidden)
            all_states.append(hidden.permute(1, 0, 2) if return_all_layers else hidden[-1])
        return torch.stack(all_states, dim=1)


class LocalTemporalAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, window_size: int = 5, dropout: float = 0.0):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def _mask(self, steps: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((steps, steps), float("-inf"), device=device)
        for target in range(steps):
            start = max(0, target - self.window_size + 1)
            mask[target, start : target + 1] = 0.0
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, attn_mask=self._mask(x.shape[1], x.device), need_weights=False)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ff(x))


class VideoPredictorLocal(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_size: int = 256,
        latent_dim: int = 128,
        attn_window_size: int = 5,
        num_heads: int = 4,
        num_attn_layers: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.attn_window_size = attn_window_size
        self.encoder = FrameEncoder(in_channels=in_channels, hidden_dim=64, output_dim=latent_dim)
        self.decoder = FrameDecoder(latent_dim=latent_dim, hidden_dim=64, out_channels=out_channels)
        self.input_proj = nn.Linear(latent_dim, hidden_size)
        self.layers = nn.ModuleList([
            LocalTemporalAttention(hidden_size, num_heads=num_heads, window_size=attn_window_size)
            for _ in range(num_attn_layers)
        ])
        self.output_proj = nn.Linear(hidden_size, latent_dim)

    def _run_stack(self, tokens: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        encoded = [self.input_proj(self.encoder(frames[:, step])) for step in range(frames.shape[1])]
        return torch.stack(encoded, dim=1)

    def _decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        outputs = [self.decoder(self.output_proj(tokens[:, step])) for step in range(tokens.shape[1])]
        return torch.stack(outputs, dim=1)

    def forward(self, frames: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        del hidden
        tokens = self._run_stack(self._encode_frames(frames))
        return self._decode_tokens(tokens), tokens[:, -1]

    def forward_with_teacher_forcing(
        self,
        frames: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del teacher_forcing_ratio
        return self.forward(frames)

    def rollout(self, initial_frames: torch.Tensor, num_steps: int) -> torch.Tensor:
        context = initial_frames
        generated: list[torch.Tensor] = []
        for _ in range(num_steps):
            tokens = self._run_stack(self._encode_frames(context))
            next_frame = self.decoder(self.output_proj(tokens[:, -1]))
            generated.append(next_frame)
            context = torch.cat([context, next_frame.unsqueeze(1).detach()], dim=1)
            if context.shape[1] > self.attn_window_size:
                context = context[:, -self.attn_window_size :]
        return torch.stack(generated, dim=1)

    def get_hidden_states(self, frames: torch.Tensor, return_all_layers: bool = False) -> torch.Tensor:
        tokens = self._encode_frames(frames)
        if return_all_layers:
            outputs: list[torch.Tensor] = []
            current = tokens
            for layer in self.layers:
                current = layer(current)
                outputs.append(current)
            return torch.stack(outputs, dim=1)
        return self._run_stack(tokens)


class StateBottleneckPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        embed_dim: int = 128,
        state_dim: int = 12,
        state_num_layers: int = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.state_num_layers = state_num_layers
        self.encoder = FrameEncoder(in_channels=in_channels, hidden_dim=64, output_dim=embed_dim)
        self.decoder = FrameDecoder(latent_dim=embed_dim, hidden_dim=64, out_channels=out_channels)
        self.encoder_to_state = nn.Linear(embed_dim, state_dim)
        self.state_gru = nn.GRU(input_size=state_dim, hidden_size=state_dim, num_layers=state_num_layers, batch_first=True)
        self.state_to_decoder = nn.Linear(state_dim, embed_dim)

    def _init_zero_hidden(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.zeros(self.state_num_layers, batch_size, self.state_dim, device=device, dtype=dtype)

    def forward(self, frames: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _, _, _ = frames.shape
        hidden_state = hidden if hidden is not None else self._init_zero_hidden(batch, frames.device, frames.dtype)
        outputs: list[torch.Tensor] = []
        for step in range(steps):
            state = self.encoder_to_state(self.encoder(frames[:, step]))
            _, hidden_state = self.state_gru(state.unsqueeze(1), hidden_state)
            outputs.append(self.decoder(self.state_to_decoder(hidden_state[-1])))
        return torch.stack(outputs, dim=1), hidden_state

    def forward_with_teacher_forcing(
        self,
        frames: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _, _, _ = frames.shape
        hidden = self._init_zero_hidden(batch, frames.device, frames.dtype)
        current = frames[:, 0]
        outputs: list[torch.Tensor] = []
        for step in range(steps):
            state = self.encoder_to_state(self.encoder(current))
            _, hidden = self.state_gru(state.unsqueeze(1), hidden)
            recon = self.decoder(self.state_to_decoder(hidden[-1]))
            outputs.append(recon)
            if step < steps - 1:
                use_teacher = torch.rand(1, device=frames.device).item() < teacher_forcing_ratio
                current = frames[:, step + 1] if use_teacher else recon.detach()
        return torch.stack(outputs, dim=1), hidden

    def rollout(self, initial_frames: torch.Tensor, num_steps: int) -> torch.Tensor:
        batch = initial_frames.shape[0]
        hidden = self._init_zero_hidden(batch, initial_frames.device, initial_frames.dtype)
        for step in range(initial_frames.shape[1]):
            state = self.encoder_to_state(self.encoder(initial_frames[:, step]))
            _, hidden = self.state_gru(state.unsqueeze(1), hidden)
        generated: list[torch.Tensor] = []
        for _ in range(num_steps):
            recon = self.decoder(self.state_to_decoder(hidden[-1]))
            generated.append(recon)
            state = self.encoder_to_state(self.encoder(recon))
            _, hidden = self.state_gru(state.unsqueeze(1), hidden)
        return torch.stack(generated, dim=1)

    def get_hidden_states(self, frames: torch.Tensor, return_all_layers: bool = False) -> torch.Tensor:
        batch, steps, _, _, _ = frames.shape
        hidden = self._init_zero_hidden(batch, frames.device, frames.dtype)
        all_states: list[torch.Tensor] = []
        for step in range(steps):
            state = self.encoder_to_state(self.encoder(frames[:, step]))
            _, hidden = self.state_gru(state.unsqueeze(1), hidden)
            all_states.append(hidden.permute(1, 0, 2) if return_all_layers else hidden[-1])
        return torch.stack(all_states, dim=1)

    def get_bottleneck_states(self, frames: torch.Tensor) -> torch.Tensor:
        states = [self.encoder_to_state(self.encoder(frames[:, step])) for step in range(frames.shape[1])]
        return torch.stack(states, dim=1)

    def predict_next_state(self, current_state: torch.Tensor) -> torch.Tensor:
        if current_state.ndim == 1:
            current_state = current_state.unsqueeze(0)
        hidden = self._init_zero_hidden(current_state.shape[0], current_state.device, current_state.dtype)
        _, next_hidden = self.state_gru(current_state.unsqueeze(1), hidden)
        return next_hidden[-1]


def create_model(
    in_channels: int = 1,
    out_channels: int = 1,
    hidden_size: int = 256,
    latent_dim: int = 128,
    num_layers: int = 1,
) -> VideoPredictor:
    return VideoPredictor(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_size=hidden_size,
        latent_dim=latent_dim,
        num_layers=num_layers,
    )


def create_model_local(
    in_channels: int = 1,
    out_channels: int = 1,
    hidden_size: int = 256,
    latent_dim: int = 128,
    attn_window_size: int = 5,
    num_heads: int = 4,
    num_attn_layers: int = 1,
) -> VideoPredictorLocal:
    return VideoPredictorLocal(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_size=hidden_size,
        latent_dim=latent_dim,
        attn_window_size=attn_window_size,
        num_heads=num_heads,
        num_attn_layers=num_attn_layers,
    )


def create_bottleneck_model(
    in_channels: int = 1,
    out_channels: int = 1,
    embed_dim: int = 128,
    state_dim: int = 12,
    state_num_layers: int = 1,
) -> StateBottleneckPredictor:
    return StateBottleneckPredictor(
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=embed_dim,
        state_dim=state_dim,
        state_num_layers=state_num_layers,
    )
