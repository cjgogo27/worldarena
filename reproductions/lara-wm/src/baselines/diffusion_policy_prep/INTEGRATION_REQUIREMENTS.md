# Diffusion Policy Integration Requirements

## Status: REPO CLONED - PREPARED FOR ADAPTATION

**Location**: `/data/alice/cjtest/lara-wm/third_party/diffusion_policy`
**Original repo**: https://github.com/real-stanford/diffusion_policy

---

## Key Format/Interface Mismatches with RoboTwin Pipeline

### 1. Data Storage Format (CRITICAL)
| Aspect | Diffusion Policy | RoboTwin (lara-wm) | Gap |
|--------|----------------|-------------------|-----|
| Storage | Zarr (`.zarr` directory or `.zarr.zip`) | HDF5 (`.h5`, `.hdf5`) | **MAJOR** - Need Zarr converter |
| Image compression | JPEG2K via `imagecodecs_numcodecs` | PNG/raw uint8 | Medium - Can handle |
| Episode structure | `data/demo_0`, `data/demo_1`, ... in HDF5 | `episodes/episode_0`, ... | Mapping needed |

### 2. Dataset Interface
```python
# Diffusion Policy expects:
class BaseLowdimDataset(torch.utils.data.Dataset):
    def get_normalizer(self) -> LinearNormalizer: ...
    def get_all_actions(self) -> torch.Tensor: ...  # (N, Da)
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # Returns: {'obs': (T, Do), 'action': (T, Da)}
        ...

# RoboTwin provides:
class RoboTwinEpisode:
    observations: dict[str, NDArray]  # {'agent_view': (T,H,W,C), 'joint_states': (T,DoF), ...}
    actions: dict[str, NDArray]    # {'joint_position': (T,DoF), 'gripper_position': (T,1)}
    rewards: NDArray             # (T,)
```

**Required adapter**: Create `RoboTwinDiffusionDataset(BaseImageDataset)` wrapping RoboTwinEpisode data to expose DiffPy interface.

### 3. Observation Key Mapping
| DiffPy Config | RoboTwin Key | Notes |
|--------------|-------------|-------|
| `obs_key: keypoint` | N/A (DiffPy-specific) | Push-T uses keypoint detection |
| `state_key: state` | `joint_states` | Maps to robot state |
| `action_key: action` | `joint_position`, `gripper_position` | Need concat + transform |

**RoboTwin native keys**: `agent_view`, `wrist_camera` (images); `joint_states`, `ee_pose` (state)

### 4. Action Representation
| Aspect | Diffusion Policy | RoboTwin | Gap |
|--------|---------------|----------|-----|
| Rotation | RotationTransformer (axis_angle → 6D) | raw axis_angle | Need conversion |
| Gripper | Included in action vector | separate `gripper_position` | Need concat |
| Absolute | Optional via `abs_action=True` | Currently relative? | Need investigation |

### 5. Normalization
- Diffusion Policy: `LinearNormalizer` with per-field stats (mean, std, min, max)
- RoboTwin: No built-in normalizer - handled per-model in training

**Required**: Implement `get_normalizer()` returning `LinearNormalizer` fitted to RoboTwinEpisode data.

### 6. Training Entry Point
```bash
# Diffusion Policy
python train.py --config-name=train_diffusion_unet_image_workspace

# lara-wm
python scripts/train_and_eval.py --tasks lift --seeds 42
```

**Integration options**:
1. Create DiffPy config pointing to RoboTwin dataset adapter (non-invasive)
2. Add Diffusion Policy as model option in train_and_eval.py (more invasive)

---

## Non-Overlapping Prep Tasks (Can proceed now)

### Done ✅
- [x] Clone real-stanford/diffusion_policy to third_party/
- [x] Inspect dataset interfaces
- [x] Identify key format mismatches

### To Do (Non-blocking) 📋
- [ ] Create `RoboTwinDiffusionDataset` adapter class (requires new file)
- [ ] Document exact Zarr conversion requirements
- [ ] Create minimal config for DiffPy training on RoboTwin data
- [ ] Test data loading pipeline

---

## Integration Decision Points

### Option A: Dataset Adapter (Recommended - Non-invasive)
1. Create `/data/alice/cjtest/lara-wm/src/baselines/diffusion_policy_prep/robottwin_diffusion_dataset.py`
   - Wraps `RoboTwinEpisode` → DiffPy interface
   - Implements `get_normalizer()` using RoboTwin episode statistics
2. Create YAML config for DiffPy training with RoboTwin data path
3. Keeps DiffPy and lara-wm evaluation independent

### Option B: Add as Trainable Model (More invasive)
- Add Diffusion Policy to `model_names` in `train_and_eval.py`
- Requires action normalization matching between models

**Recommendation**: Use Option A for baseline preparation - keeps ACT work non-overlapping.

---

## Dependencies Required
See `conda_environment.yaml` in cloned repo. Key packages:
- `hydra-core`, `omegaconf` - config management
- `zarr` - data storage
- `imagecodecs-numcodecs` - image compression
- `diffusers` - diffusion schedulers
- `robomimic` - dependency in some datasets

---

## Action Items for Next Steps
1. ✅ Repository cloned to `third_party/diffusion_policy`
2. ⏳ Create `RoboTwinDiffusionDataset` adapter (after ACT integration)
3. ⏳ Define exact dataset conversion from HDF5→ZARR (or use in-memory)
4. ⏳ Create test config to verify DiffPy runs on sample data