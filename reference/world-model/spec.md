================================================================================
AGY PROJECT SPECIFICATION: ACTION-CONDITIONED WORLD MODELS VS. LE WORLD MODEL (SIGREG)
================================================================================

PROJECT OBJECTIVE:
Implement, evaluate, and benchmark two contrasting world model paradigms on continuous-action 
visual control: a Generative Action-Conditioned Latent Diffusion Model (AC-LDM) paired with 
a Spatial VAE versus a Non-Generative, Teacher-Free Le World Model (LeWM) using 
Sketched Isotropic Gaussian Regularization (SIGReg) without an EMA target encoder.

--------------------------------------------------------------------------------
PHASE 1: ENVIRONMENT ROLLOUT & OFFLINE DATA INGESTION
--------------------------------------------------------------------------------
MILESTONE:
Establish a deterministic data harvesting pipeline and generate a verified offline 
dataset of continuous transitions with ground-truth physical telemetry.

STEPS:
1. Environment Wrapper Setup
   - Configure continuous visual control environment with center cropping and frame normalization.
   - Synchronize underlying simulator physics states alongside rendered observations.
2. Trajectory Rollout Harvesting
   - Implement an automated trajectory generator using a mixture of heuristic policy and random perturbation.
   - Stream and save transitions to disk with chunked storage to maintain minimal memory overhead.
3. Dataset Loader & Augmentation Engine
   - Build a high-throughput trajectory dataset loader yielding consecutive observation pairs, actions, rewards, terminals, and physical telemetry.
   - Implement synchronized spatial-action augmentations (horizontal flipping with inverted steering signs and lateral velocity).
4. Data Integrity Verification
   - Verify tensor shapes, value ranges, and continuity of physical state trajectories across train/val/test splits.

PHASE EXIT GATE:
Offline dataset contains at least 50,000 valid transitions with complete visual and physical telemetry, passing all automated data integrity checks.

--------------------------------------------------------------------------------
PHASE 2: SPATIAL REPRESENTATION COMPRESSION (VAE ENGINE)
--------------------------------------------------------------------------------
MILESTONE:
Train and freeze a continuous spatial Autoencoder capable of compressing high-dimensional 
pixel observations into compact spatial latents while preserving critical road and agent cues.

STEPS:
1. Autoencoder Architecture Construction
   - Build a convolutional encoder with residual downsampling stages mapping frames to continuous mean and variance latent parameters.
   - Build a symmetric residual decoder reconstructing spatial latent tensors back to observation space.
2. Objective Formulation & Regularization
   - Implement composite loss combining pixel-level L1 reconstruction, perceptual similarity (LPIPS), and Kullback-Leibler (KL) divergence.
   - Tune KL regularization weighting to balance reconstruction sharpness against latent manifold smoothness.
3. Training & Validation Execution
   - Train on offline transitions with cosine learning rate scheduling and early stopping.
   - Periodically evaluate reconstruction fidelity against an unseen validation set.
4. Representation Freezing & Checkpointing
   - Validate reconstruction quality benchmarks.
   - Freeze all encoder and decoder parameters for downstream latent world model training.

PHASE EXIT GATE:
Reconstruction quality exceeds target thresholds for PSNR, SSIM, and LPIPS; encoder-decoder weights are locked and validated.

--------------------------------------------------------------------------------
PHASE 3: ACTION-CONDITIONED LATENT DIFFUSION WORLD MODEL (AC-LDM)
--------------------------------------------------------------------------------
MILESTONE:
Implement and train a conditional latent diffusion network that generates future latent 
states conditioned on current visual latents and continuous action vectors.

STEPS:
1. Denoising Backbone Implementation
   - Build a spatial U-Net operating on noisy future latents conditioned on previous state latents.
   - Implement continuous action vector and diffusion timestep embeddings injected via Adaptive Normalization layers across residual blocks.
   - Add conditional dropout on action vectors during training to enable classifier-free guidance.
2. Diffusion Process & Schedule Design
   - Define forward Gaussian diffusion noise schedule across discrete time intervals.
   - Implement an accelerated sampling algorithm (DDIM) for multi-step latent unrolling with minimal inference steps.
3. Model Training & Optimization
   - Train the latent denoising network using mean-squared epsilon error objective.
   - Monitor gradient norms and validation denoising loss to ensure stable convergence.
4. Multi-Step Latent Rollout Validation
   - Unroll recursive state-action predictions over extended horizons in latent space.
   - Decode unrolled latents back to image space to qualitatively verify action-driven scene consistency.

PHASE EXIT GATE:
Fast sampler reliably generates multi-step unrolled latent transitions without numerical instability or visual collapse, demonstrating explicit controllability under opposing actions.

--------------------------------------------------------------------------------
PHASE 4: TEACHER-FREE LE WORLD MODEL BASELINE (SIGREG ENGINE)
--------------------------------------------------------------------------------
MILESTONE:
Implement and train a Teacher-Free Action-Conditioned World Model that predicts transitions 
directly in representation space using a single shared encoder and Sketched Isotropic 
Gaussian Regularization (SIGReg), eliminating EMA target networks and contrastive pairs.

STEPS:
1. Teacher-Free Single-Encoder Setup
   - Construct a single visual encoder mapping both context frames and target frames to latent representations.
   - Eliminate all teacher branches, momentum parameters, and stop-gradient operations to allow joint end-to-end backpropagation.
2. Latent Dynamics Predictor
   - Implement a residual predictor network taking the context representation and continuous action vector to estimate future state embeddings.
3. Sketched Isotropic Gaussian Regularization (SIGReg) Formulation
   - Implement random 1D directional slicing over the latent batch distribution via the Cramér-Wold device.
   - Compute the analytical Epps-Pulley characteristic function test statistic against a standard normal distribution to enforce isotropic Gaussianity and prevent representation collapse.
4. Joint End-to-End Optimization
   - Train the shared visual encoder and transition predictor jointly under the combined transition error and SIGReg loss penalty.
   - Track Gaussian goodness-of-fit statistics, singular value spectrum, and validation dynamics error to confirm representation richness without collapse.

PHASE EXIT GATE:
Latent embeddings match standard isotropic Gaussian properties with high effective rank across all dimensions; transition prediction error steadily decreases on unseen trajectories.

--------------------------------------------------------------------------------
PHASE 5: COMPARATIVE BENCHMARKING, STRESS-TESTING & LATENT MPC
--------------------------------------------------------------------------------
MILESTONE:
Execute rigorous head-to-head empirical evaluations comparing the generative diffusion 
model against the teacher-free SIGReg world model across physical fidelity, distractor resilience, 
latency, and planning.

STEPS:
1. Physical Telemetry Linear Probing
   - Train lightweight linear regression heads on frozen representations from both models to predict true physical state variables (velocity, heading, coordinates).
   - Compute and compare determination coefficients (R-squared) to evaluate dynamics preservation.
2. Visual Distractor Stress-Testing
   - Inject task-irrelevant visual corruption (high-frequency background noise and color shifts) into test observations.
   - Measure relative degradation in multi-step prediction error for both architectures to test the non-generative abstraction hypothesis.
3. Step Latency & Computational Profiling
   - Measure per-step transition latency and memory footprint comparing iterative diffusion sampling against single-pass forward inference in the teacher-free model.
4. Latent Model Predictive Control (MPC / CEM)
   - Implement a Cross-Entropy Method planner operating entirely within each model's latent representation space.
   - Train latent reward estimators and evaluate closed-loop agent control over multiple test episodes.
5. Report Asset & Matrix Compilation
   - Aggregate all quantitative metrics into a structured summary report.
   - Generate qualitative trajectory comparison plots, latent projection visualizations, and attention maps.

PHASE EXIT GATE:
All comparative tables, distractor degradation scores, inference latency profiles, and closed-loop latent MPC benchmark results are finalized and exported.
================================================================================
