# OpenLow

**An interactive, WebGPU-accelerated visual transformer builder.**

OpenLow is a browser-based playground for designing, training, and chatting with a real character-level GPT — all from a node-based visual canvas. Drag transformer building blocks onto a canvas, wire them into a forward pass, then train a genuinely backpropagating model on either the **CPU** or the **GPU (WebGPU)** and talk to it in a built-in chat panel.

> The visual node graph illustrates a live forward pass of the architecture, while a dedicated `CharTransformer` engine handles the real training and inference with mathematically correct backpropagation and Adam optimization.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Compute Backends (CPU & WebGPU)](#compute-backends-cpu--webgpu)
- [Hyperparameters](#hyperparameters)
- [Project Structure](#project-structure)
- [Engine Internals](#engine-internals)
- [Testing](#testing)
- [Browser Support](#browser-support)

---

## Features

- **Node-based visual canvas** — Pan, zoom, drag nodes, and draw edges between typed ports to build a GPT architecture by hand.
- **Real trainable transformer** — A single-block character-level GPT with full backpropagation and an Adam optimizer (not a simulation).
- **Dual compute backends** — Train and infer on the **CPU** (reference implementation) or the **GPU** via WebGPU WGSL compute kernels. GPU is the default.
- **Automatic GPU parity checking** — Before trusting the GPU, OpenLow runs a forward + backward pass against the CPU reference and transparently falls back to CPU if results diverge beyond `1e-2`.
- **Chat interface** — Talk to your trained model. Prompts are auto-formatted as `Q: … \nA:` and responses stream in token-by-token.
- **Auto-generation loop** — Watch the model generate text continuously, with the canvas updating live as tokens are sampled.
- **Preset blueprints** — Load ready-made graphs: `embeddings`, `layernorm`, and a complete `full` GPT layer.
- **Node inspector** — Select any node to edit its parameters (vocab size, embedding dim, heads, sampler temperature/top-k, etc.) and preview tensor outputs.
- **Built-in console** — Color-coded, timestamped logs for info, success, errors, warnings, and GPU events, with copy-to-clipboard.
- **QA dataset included** — One-click load of a bundled question/answer dataset to start training immediately.

---

## Tech Stack

| Component | Version | Purpose |
| --- | --- | --- |
| React | 19 | UI framework |
| TypeScript | ~6.0 | Type safety |
| Vite | 8 | Dev server & build tool (HMR) |
| Tailwind CSS | 4 | Utility-first styling |
| WebGPU (`@webgpu/types`) | 0.1.x | GPU compute kernels |
| Phosphor Icons | 2.1.x | Iconography |

---

## Getting Started

### Prerequisites

- **Node.js** 18+ (recommended 20+)
- A **WebGPU-capable browser** for GPU acceleration (see [Browser Support](#browser-support)). The CPU backend works everywhere.

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Open the printed local URL (default `http://localhost:5173`).

### Production Build

```bash
npm run build     # type-check (tsc -b) + vite build → dist/
npm run preview   # preview the production build locally
```

### Linting

```bash
npm run lint
```

### Deployment (GitHub Pages)

The repo includes a GitHub Actions workflow at [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) that builds the app and publishes it to GitHub Pages on every push to `main`.

To enable it:

1. Push the project to a GitHub repository named **`openlow`**.
2. In the repo, go to **Settings → Pages** and set **Source** to **GitHub Actions**.
3. Push to `main` (or run the workflow manually) — the site deploys to `https://<your-username>.github.io/openlow/`.

> The Vite `base` is set to `/openlow/` for production builds in [`vite.config.ts`](vite.config.ts). If you name the repository something other than `openlow`, update `base` to match `/<repo-name>/`.

---

## Usage

A typical end-to-end flow:

1. **Load an architecture.** Click a preset (`full` gives you a complete GPT layer) or drag nodes from the **Sidebar** and wire them together on the **Canvas**.
2. **Load training data.** Click **Load QA Dataset** to populate the Text Input node with the bundled `training.data`, or type your own corpus.
3. **(Optional) Tune hyperparameters.** Open **Settings** in the Control Panel to adjust `nEmbd`, `blockSize`, `mlpMult`, `lr`, `steps`, and `batch`.
4. **Pick a backend.** Toggle between **CPU** and **WebGPU** (default). GPU runs a parity check before use and falls back to CPU automatically if it diverges.
5. **Train.** Click **Train**. Loss is logged periodically in the console, followed by a short self-test generation.
6. **Chat.** Switch to the **Chat** tab and send a message — responses stream in character by character. Or enable **Auto-Generate** to watch continuous generation drive the canvas.

### Node Palette

| Category | Nodes |
| --- | --- |
| Inputs & Data | Text Input, Tokenizer |
| Embedding Layers | WTE (token embeddings), WPE (positional embeddings) |
| Transformer Math | Add, LayerNorm, Attention, MLP |
| Outputs & Samplers | LM Head, Softmax, Sampler |

### Canvas Controls

- **Pan** — scroll / drag the background
- **Zoom** — mouse wheel (roughly 50%–200%)
- **Move a node** — drag its header
- **Connect ports** — click an output port, then an input port to create an edge
- **Inspect / edit** — click a node to open the inspector panel

---

## Model Architecture

OpenLow trains a single-block, character-level GPT. The forward pass:

```
[tokens]    --WTE-->  [B, T, C]
[positions] --WPE-->  [B, T, C]
                 │
              (Add)
                 │
            LayerNorm
                 │
   Single-head causal self-attention   (Q,K,V proj → scaled dot-product → output proj)
                 │
              (Add residual)
                 │
            LayerNorm
                 │
        MLP: Linear(C → C·mlpMult) → GELU → Linear(C·mlpMult → C)
                 │
              (Add residual)
                 │
            LayerNorm
                 │
        LM Head: Linear(C → vocab)
                 │
             Softmax
                 │
       Sampler (temperature + top-k)
```

- **Vocabulary** is built dynamically from the unique characters in the training data.
- **Attention** is single-head and causal (lower-triangular mask), with scaled dot-product scoring.
- **Weight initialization** uses a Box-Muller normal distribution (std `0.02`); LayerNorm gains start at `1.0`, biases at `0.0`. Output projections start near zero so residual paths dominate early training.
- **Optimizer** is Adam (β₁ = 0.9, β₂ = 0.999, ε = 1e-8).

### Inference & Sampling

During chat/generation, the context is sliced to the last `blockSize` tokens and tokens are sampled with **temperature = 0.5** and **top-k = 6**. Generation stops at a newline or a length cap.

---

## Compute Backends (CPU & WebGPU)

OpenLow ships two numerically-aligned implementations of the same model:

- **CPU (`cpuTransformer.ts`)** — A readable reference implementation of the full forward + backward pass and Adam update. Always available.
- **WebGPU (`gpuTransformer.ts` + `webgpuEngine.ts`)** — Mirrors the CPU model with WGSL compute kernels. Weights, gradients, and Adam moments stay resident on the GPU; only scalars (loss, grad norm) are read back.

**Parity & fallback:** before each GPU training run, OpenLow executes one forward + backward pass on a fixed sequence and compares GPU output against the CPU reference. If the absolute difference exceeds `1e-2`, it logs a warning and transparently falls back to the CPU backend — so you never silently train on incorrect GPU math.

---

## Hyperparameters

Editable from the **Settings** panel. Defaults:

| Parameter | Default | Description |
| --- | --- | --- |
| `nEmbd` | `64` | Embedding / channel dimension (C) |
| `blockSize` | `64` | Context window (max sequence length) |
| `mlpMult` | `4` | MLP hidden expansion factor |
| `lr` | `0.003` | Adam learning rate |
| `steps` | `500` | Number of gradient steps |
| `batch` | `16` | Sequences per gradient step |

---

## Project Structure

```
openlow/
├── index.html
├── package.json
├── vite.config.ts
├── .github/
│   └── workflows/
│       └── deploy.yml             # GitHub Pages build & deploy
├── scripts/
│   └── test-cpu-transformer.mts      # offline CPU training/generation test
└── src/
    ├── App.tsx                        # app shell, training/inference orchestration
    ├── main.tsx
    ├── assets/
    │   └── training.data              # bundled QA dataset
    ├── components/
    │   ├── Canvas/Canvas.tsx          # SVG node-edge graph (pan/zoom/drag/connect)
    │   ├── ControlPanel/ControlPanel.tsx   # tabs: Chat | Raw Text | Settings, console
    │   ├── Inspector/NodeInspector.tsx     # per-node parameter editor + previews
    │   └── Sidebar/Sidebar.tsx        # categorized node palette
    └── engine/
        ├── cpuTransformer.ts          # reference CPU transformer (backprop + Adam)
        ├── gpuTransformer.ts          # WebGPU trainer mirroring the CPU model
        ├── webgpuEngine.ts            # device init, pipelines, buffers
        ├── shaders.ts                 # WGSL compute kernels
        ├── graphTypes.ts              # node/port/edge type definitions
        └── tokenizer.ts               # byte-level encode/decode
```

---

## Engine Internals

- **`graphTypes.ts`** — Defines the node model: input nodes (`text_input`, `tokenizer`), embeddings (`wte`, `wpe`), math ops (`add`, `layernorm`, `attention`, `mlp`), and outputs (`lm_head`, `softmax`, `sampler`). Each node carries typed ports (text / tokens / tensor), shape metadata, and an execution status.
- **`webgpuEngine.ts`** — Requests the GPU adapter/device, builds and caches compute pipelines per op, manages storage/uniform buffers and read-back mappings, and initializes weights.
- **`shaders.ts`** — WGSL kernels for token/positional embedding lookup, element-wise add, layer normalization, attention, MLP, GELU, and softmax (workgroup size 64).
- **`tokenizer.ts`** — Simple byte-level (0–255) `encode`/`decode`. The trainer additionally builds a character-level vocabulary from the dataset.
- **Stale-closure safety** — Async generation/training loops read graph state through refs to always see the latest nodes and edges.

---

## Testing

An offline script validates the CPU transformer end-to-end without a browser:

```bash
node scripts/test-cpu-transformer.mts
```

It loads the bundled QA dataset, builds a `nEmbd=64, blockSize=64, mlpMult=4, lr=0.003` model, runs training steps (logging loss periodically), and generates answers to sample prompts to confirm the model is learning.

---

## Browser Support

- **WebGPU backend** requires a browser with WebGPU enabled (recent Chrome, Edge, and Safari Technology Preview; Firefox behind a flag). On unsupported browsers the GPU toggle is disabled and the WebGPU status badge shows **OFF**.
- **CPU backend** works in any modern browser.

---

## License

This project does not currently specify a license. Add one before distribution.
