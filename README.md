# OpenLow

**An interactive, WebGPU-accelerated visual transformer builder.**

**🔗 Live demo: [halilugur.github.io/openlow](https://halilugur.github.io/openlow/)**

OpenLow is a browser-based playground for designing, training, and chatting with a real character-level GPT — all from a node-based visual canvas. Drag transformer building blocks onto a canvas, wire them into a forward pass, then train a genuinely backpropagating model on either the **CPU** or the **GPU (WebGPU)** and talk to it in a built-in chat panel.

> **The canvas IS the model.** When you press Train, the node graph you wired is compiled into a concrete architecture — the number of Transformer blocks, embedding dimension, context window, and MLP expansion are all read from your nodes — and a `CharTransformer` engine trains exactly that structure with mathematically correct backpropagation and Adam optimization. Change the canvas, retrain, and the real model changes with it.

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
- **Canvas-driven architecture** — The graph you wire is the model. A graph compiler reads the number of Attention/MLP blocks, embedding dimension, context window, and MLP expansion straight from your nodes, then trains that exact structure. Stack more Attention + MLP pairs to build a deeper model.
- **Topology validation** — Before training, OpenLow checks your wiring against a valid GPT topology. If something is missing or mis-wired (disconnected tokenizer, mismatched embedding dims, unequal Attention/MLP counts), it blocks training and logs exactly what to fix.
- **Real trainable transformer** — A multi-block character-level GPT with full backpropagation and an Adam optimizer (not a simulation).
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

1. **Build an architecture.** Click a preset (`full` gives you a complete GPT layer) or drag nodes from the **Sidebar** and wire them together on the **Canvas**. The structure you wire is what gets trained — add another Attention + MLP pair for a deeper model, or change a node's `n_embd` / `block_size` / `multiplier` to resize it.
2. **Load training data.** Click **Load QA Dataset** to populate the Text Input node with the bundled `training.data`, or type your own corpus.
3. **(Optional) Tune training knobs.** Open **Settings** in the Control Panel to adjust `lr`, `steps`, and `batch`. Architecture (`nEmbd`, `blockSize`, `mlpMult`, block count) comes from the canvas, not from Settings.
4. **Pick a backend.** Toggle between **CPU** and **WebGPU** (default). GPU runs a parity check before use and falls back to CPU automatically if it diverges. Multi-block models always train on CPU.
5. **Train.** Click **Train**. The graph is recompiled from the current canvas, validated, then trained. Loss is logged periodically in the console, followed by a short self-test generation. (Editing the canvas only changes the model on the next Train.)
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

OpenLow trains a character-level GPT whose shape is compiled from the canvas. Each **Causal Self-Attention + MLP pair you wire becomes one Transformer block**, so the depth is whatever you build. The forward pass:

```
[tokens]    --WTE-->  [B, T, C]
[positions] --WPE-->  [B, T, C]
                 │
              (Add)
                 │
   ┌─────────────────────┐  × nLayer (one per Attention+MLP pair)
   │        LayerNorm        │
   │           │            │
   │  Single-head causal    │  (Q,K,V proj → scaled dot-product → output proj)
   │   self-attention       │
   │           │            │
   │     (Add residual)     │
   │           │            │
   │        LayerNorm        │
   │           │            │
   │  MLP: Linear(C → C·mlpMult) → GELU → Linear(C·mlpMult → C)
   │           │            │
   │     (Add residual)     │
   └─────────────────────┘
                 │
            LayerNorm  (final)
                 │
        LM Head: Linear(C → vocab)
                 │
             Softmax
                 │
       Sampler (temperature + top-k)
```

- **Architecture is derived from the graph** — `nLayer` from the Attention node count, `nEmbd` from WTE/WPE `n_embd` (and Attention `n_head × head_size`), `blockSize` from the WPE node, and `mlpMult` from the MLP node. The graph compiler validates these are consistent before training.
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

**Parity & fallback:** before each GPU training run, OpenLow executes one forward + backward pass on a fixed sequence and compares GPU output against the CPU reference. If the absolute difference exceeds `1e-2`, it logs a warning and transparently falls back to the CPU backend — so you never silently train on incorrect GPU math. The WebGPU mirror implements a **single** Transformer block, so multi-block architectures (2+ Attention nodes) always train on the CPU engine.

---

## Hyperparameters

Architecture parameters come from the **canvas nodes**; the **Settings** panel holds the training knobs. Defaults:

| Parameter | Source | Default | Description |
| --- | --- | --- | --- |
| `nEmbd` | WTE/WPE `n_embd`, Attention `n_head×head_size` | `64` | Embedding / channel dimension (C) |
| `blockSize` | WPE `block_size` | `64` | Context window (max sequence length) |
| `mlpMult` | MLP `multiplier` | `4` | MLP hidden expansion factor |
| `nLayer` | number of Attention nodes | `1` | Stacked Transformer blocks |
| `lr` | Settings panel | `0.003` | Adam learning rate |
| `steps` | Settings panel | `500` | Number of gradient steps |
| `batch` | Settings panel | `16` | Sequences per gradient step |

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
        ├── cpuTransformer.ts          # reference CPU transformer (multi-block backprop + Adam)
        ├── gpuTransformer.ts          # WebGPU trainer mirroring a single CPU block
        ├── graphCompiler.ts           # compiles + validates the canvas graph into a model config
        ├── webgpuEngine.ts            # device init, pipelines, buffers
        ├── shaders.ts                 # WGSL compute kernels
        ├── graphTypes.ts              # node/port/edge type definitions
        └── tokenizer.ts               # byte-level encode/decode
```

---

## Engine Internals

- **`graphTypes.ts`** — Defines the node model: input nodes (`text_input`, `tokenizer`), embeddings (`wte`, `wpe`), math ops (`add`, `layernorm`, `attention`, `mlp`), and outputs (`lm_head`, `softmax`, `sampler`). Each node carries typed ports (text / tokens / tensor), shape metadata, and an execution status.
- **`graphCompiler.ts`** — Turns the wired node graph into a concrete model config (`nEmbd`, `blockSize`, `mlpMult`, `nLayer`). Validates the topology against a valid GPT shape and returns precise, blocking errors when the graph is incomplete or inconsistent.
- **`cpuTransformer.ts`** — Reference character-level GPT with full forward/backward and Adam. Supports an arbitrary number of stacked blocks (`nLayer`); per-block weights live in a `Block` class and the residual-stream gradient is threaded through the stack in reverse.
- **`webgpuEngine.ts`** — Requests the GPU adapter/device, builds and caches compute pipelines per op, manages storage/uniform buffers and read-back mappings, and initializes weights.
- **`shaders.ts`** — WGSL kernels for token/positional embedding lookup, element-wise add, layer normalization, attention, MLP, GELU, and softmax (workgroup size 64).
- **`tokenizer.ts`** — Simple byte-level (0–255) `encode`/`decode`. The trainer additionally builds a character-level vocabulary from the dataset.
- **Stale-closure safety** — Async generation/training loops read graph state through refs to always see the latest nodes and edges.

---

## Testing

Validate the project with the type-checker and linter:

```bash
npm run build   # tsc -b type-check + vite build
npm run lint    # ESLint
```

The CPU transformer's multi-block backpropagation has been verified against finite-difference numerical gradients, and training reduces loss for both single- and multi-block architectures.

---

## Browser Support

- **WebGPU backend** requires a browser with WebGPU enabled (recent Chrome, Edge, and Safari Technology Preview; Firefox behind a flag). On unsupported browsers the GPU toggle is disabled and the WebGPU status badge shows **OFF**.
- **CPU backend** works in any modern browser.

---

## License

Licensed under the [Apache License 2.0](LICENSE).
