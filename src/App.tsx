import { useState, useEffect, useRef } from 'react';
import { Sidebar } from './components/Sidebar/Sidebar';
import { Canvas } from './components/Canvas/Canvas';
import { NodeInspector } from './components/Inspector/NodeInspector';
import { ControlPanel } from './components/ControlPanel/ControlPanel';
import type { CanvasNode, CanvasEdge, NodeType } from './engine/graphTypes';
import { createDefaultNode, createNodePorts } from './engine/graphTypes';
import { WebGPUEngine } from './engine/webgpuEngine';
import type { Tensor } from './engine/webgpuEngine';
import { ByteTokenizer } from './engine/tokenizer';
import { CharTransformer } from './engine/cpuTransformer';
import { GpuTransformer } from './engine/gpuTransformer';
import { compileGraph } from './engine/graphCompiler';
import { SparkleIcon } from '@phosphor-icons/react';
import qaText from './assets/training.data?raw';

interface LogMessage {
  timestamp: string;
  type: 'info' | 'success' | 'error' | 'warning' | 'gpu';
  message: string;
}

export interface TrainParams {
  nEmbd: number;     // embedding / channel dimension
  blockSize: number; // context window length
  mlpMult: number;   // MLP hidden expansion factor
  lr: number;        // Adam learning rate
  steps: number;     // number of optimization steps
  batch: number;     // sequences per step
}

function App() {
  const [nodes, setNodes] = useState<CanvasNode[]>([]);
  const [edges, setEdges] = useState<CanvasEdge[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  
  // Execution states
  const [gpuAvailable, setGpuAvailable] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [executionTimeMs, setExecutionTimeMs] = useState<number | null>(null);
  const [generatedText, setGeneratedText] = useState<string>('');
  const [isAutoGenerating, setIsAutoGenerating] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [chatHistory, setChatHistory] = useState<{ role: 'user' | 'assistant'; content: string }[]>([]);
  const [isChatResponding, setIsChatResponding] = useState(false);
  const [logs, setLogs] = useState<LogMessage[]>([]);

  // Engine refs
  const engineRef = useRef<WebGPUEngine | null>(null);
  const tokenizerRef = useRef<ByteTokenizer | null>(null);
  const autoGenTimerRef = useRef<number | null>(null);

  // Real trainable character-level transformer (correct backprop + Adam).
  // This is the actual "brain" that learns the dataset and answers in chat;
  // the WebGPU node graph remains the live visualization of a forward pass.
  const cpuModelRef = useRef<CharTransformer | null>(null);
  const gpuModelRef = useRef<GpuTransformer | null>(null);
  const [isModelTrained, setIsModelTrained] = useState(false);
  // Compute backend for real training/inference: 'cpu' (always available) or
  // 'webgpu' (GPU kernels, verified against CPU at train time).
  const [computeBackend, setComputeBackend] = useState<'cpu' | 'webgpu'>('webgpu');
  const computeBackendRef = useRef<'cpu' | 'webgpu'>('webgpu');
  computeBackendRef.current = computeBackend;

  // User-tunable hyperparameters for the real transformer trainer.
  const [trainParams, setTrainParams] = useState<TrainParams>({
    nEmbd: 64,
    blockSize: 64,
    mlpMult: 4,
    lr: 0.003,
    steps: 500,
    batch: 16,
  });

  // Cache for weights inside nodes: nodeId -> weights object
  const nodeWeights = useRef<Map<string, Record<string, Tensor>>>(new Map());

  // Store intermediate activations after execution
  const latestOutputs = useRef<Map<string, Tensor | string>>(new Map());

  // Live refs to the graph so async generation loops (setTimeout recursion)
  // always read the latest nodes/edges instead of a stale closure snapshot.
  const nodesRef = useRef<CanvasNode[]>(nodes);
  const edgesRef = useRef<CanvasEdge[]>(edges);
  nodesRef.current = nodes;
  edgesRef.current = edges;

  // Recently generated token IDs, used by the sampler to apply a repetition
  // penalty so generation does not collapse into degenerate loops ("I'm I'm...").
  const genHistoryRef = useRef<number[]>([]);

  // Initialize WebGPU & Tokenizer on startup
  useEffect(() => {
    const initEngine = async () => {
      const engine = new WebGPUEngine();
      const tokenizer = new ByteTokenizer();
      engineRef.current = engine;
      tokenizerRef.current = tokenizer;

      addLog('info', 'Requesting WebGPU browser adapter...');
      const ok = await engine.init();
      setGpuAvailable(ok);

      if (ok) {
        addLog('success', 'WebGPU successfully initialized. GPUDevice acquired.');
      } else {
        addLog('error', 'WebGPU not supported or enabled in this browser. Running in fallback mode (simulation).');
      }
      
      // Load the default Full preset on start
      loadPreset('full');
    };

    initEngine();

    return () => {
      if (autoGenTimerRef.current) {
        window.clearTimeout(autoGenTimerRef.current);
      }
    };
  }, []);

  // Helper to append logs
  const addLog = (type: LogMessage['type'], message: string) => {
    const timestamp = new Date().toLocaleTimeString([], { hour12: false });
    setLogs(prev => [...prev, { timestamp, type, message }]);
  };

  const clearLogs = () => {
    setLogs([]);
  };

  // Add node click handler
  const handleAddNode = (type: NodeType, x = 100, y = 100) => {
    const newNode = createDefaultNode(type, x, y);
    setNodes(prev => [...prev, newNode]);
    addLog('info', `Added node: ${newNode.label}`);
  };

  const handleUpdateNodes = (updated: CanvasNode[]) => {
    setNodes(updated);
  };

  const handleUpdateEdges = (updated: CanvasEdge[]) => {
    setEdges(updated);
  };

  const handleUpdateNodeParams = (nodeId: string, params: Record<string, any>) => {
    setNodes(prev => prev.map(n => {
      if (n.id === nodeId) {
        // Clear status so user knows parameters changed
        return { ...n, params, status: 'idle', outputShape: undefined, outputPreview: undefined };
      }
      return n;
    }));
  };

  const handleClearCanvas = () => {
    setNodes([]);
    setEdges([]);
    setSelectedNodeId(null);
    nodeWeights.current.clear();
    setGeneratedText('');
    setExecutionTimeMs(null);
    addLog('info', 'Canvas cleared.');
  };

  // Preset Blueprint loader
  const loadPreset = (name: string) => {
    handleClearCanvas();
    const newNodes: CanvasNode[] = [];
    const newEdges: CanvasEdge[] = [];

    const createNodeWithId = (type: NodeType, id: string, x: number, y: number): CanvasNode => {
      const node = createDefaultNode(type, x, y);
      const { inputs, outputs } = createNodePorts(type, id);
      return {
        ...node,
        id,
        inputs,
        outputs,
      };
    };

    const idText = 'node-text';
    const idTok = 'node-tokenizer';
    const idWte = 'node-wte';
    const idWpe = 'node-wpe';
    const idAdd = 'node-add';
    const idLn1 = 'node-ln1';
    const idAttn = 'node-attn';
    const idAddRes1 = 'node-add-res1';
    const idLn2 = 'node-ln2';
    const idMlp = 'node-mlp';
    const idAddRes2 = 'node-add-res2';
    const idHead = 'node-head';
    const idSoft = 'node-softmax';
    const idSamp = 'node-sampler';

    if (name === 'embeddings') {
      newNodes.push(
        createNodeWithId('text_input', idText, 40, 150),
        createNodeWithId('tokenizer', idTok, 280, 150),
        createNodeWithId('wte', idWte, 520, 40),
        createNodeWithId('wpe', idWpe, 520, 260),
        createNodeWithId('add', idAdd, 760, 150)
      );
      
      // Wire input/outputs
      newEdges.push(
        { id: 'e1', fromNodeId: idText, fromPortId: `${idText}-out-text`, toNodeId: idTok, toPortId: `${idTok}-in-text` },
        { id: 'e2', fromNodeId: idTok, fromPortId: `${idTok}-out-tokens`, toNodeId: idWte, toPortId: `${idWte}-in-tokens` },
        { id: 'e3', fromNodeId: idTok, fromPortId: `${idTok}-out-tokens`, toNodeId: idWpe, toPortId: `${idWpe}-in-tokens` },
        { id: 'e4', fromNodeId: idWte, fromPortId: `${idWte}-out-tensor`, toNodeId: idAdd, toPortId: `${idAdd}-in-a` },
        { id: 'e5', fromNodeId: idWpe, fromPortId: `${idWpe}-out-tensor`, toNodeId: idAdd, toPortId: `${idAdd}-in-b` }
      );

      addLog('info', 'Loaded "Simple Embeddings" preset.');
    } else if (name === 'layernorm') {
      newNodes.push(
        createNodeWithId('text_input', idText, 40, 150),
        createNodeWithId('tokenizer', idTok, 280, 150),
        createNodeWithId('wte', idWte, 520, 40),
        createNodeWithId('wpe', idWpe, 520, 260),
        createNodeWithId('add', idAdd, 760, 150),
        createNodeWithId('layernorm', idLn1, 1000, 150)
      );

      newEdges.push(
        { id: 'e1', fromNodeId: idText, fromPortId: `${idText}-out-text`, toNodeId: idTok, toPortId: `${idTok}-in-text` },
        { id: 'e2', fromNodeId: idTok, fromPortId: `${idTok}-out-tokens`, toNodeId: idWte, toPortId: `${idWte}-in-tokens` },
        { id: 'e3', fromNodeId: idTok, fromPortId: `${idTok}-out-tokens`, toNodeId: idWpe, toPortId: `${idWpe}-in-tokens` },
        { id: 'e4', fromNodeId: idWte, fromPortId: `${idWte}-out-tensor`, toNodeId: idAdd, toPortId: `${idAdd}-in-a` },
        { id: 'e5', fromNodeId: idWpe, fromPortId: `${idWpe}-out-tensor`, toNodeId: idAdd, toPortId: `${idAdd}-in-b` },
        { id: 'e6', fromNodeId: idAdd, fromPortId: `${idAdd}-out-tensor`, toNodeId: idLn1, toPortId: `${idLn1}-in-x` }
      );

      addLog('info', 'Loaded "LayerNorm Block" preset.');
    } else if (name === 'full') {
      newNodes.push(
        createNodeWithId('text_input', idText, 50, 150),
        createNodeWithId('tokenizer', idTok, 320, 150),
        createNodeWithId('wte', idWte, 590, 40),
        createNodeWithId('wpe', idWpe, 590, 260),
        createNodeWithId('add', idAdd, 860, 150),
        createNodeWithId('layernorm', idLn1, 1130, 150),
        createNodeWithId('attention', idAttn, 1400, 150),
        createNodeWithId('add', idAddRes1, 1670, 150),
        createNodeWithId('layernorm', idLn2, 1940, 150),
        createNodeWithId('mlp', idMlp, 2210, 150),
        createNodeWithId('add', idAddRes2, 2480, 150),
        createNodeWithId('lm_head', idHead, 2750, 150),
        createNodeWithId('softmax', idSoft, 3020, 150),
        createNodeWithId('sampler', idSamp, 3290, 150)
      );

      newEdges.push(
        { id: 'e1', fromNodeId: idText, fromPortId: `${idText}-out-text`, toNodeId: idTok, toPortId: `${idTok}-in-text` },
        { id: 'e2', fromNodeId: idTok, fromPortId: `${idTok}-out-tokens`, toNodeId: idWte, toPortId: `${idWte}-in-tokens` },
        { id: 'e3', fromNodeId: idTok, fromPortId: `${idTok}-out-tokens`, toNodeId: idWpe, toPortId: `${idWpe}-in-tokens` },
        { id: 'e4', fromNodeId: idWte, fromPortId: `${idWte}-out-tensor`, toNodeId: idAdd, toPortId: `${idAdd}-in-a` },
        { id: 'e5', fromNodeId: idWpe, fromPortId: `${idWpe}-out-tensor`, toNodeId: idAdd, toPortId: `${idAdd}-in-b` },
        { id: 'e6', fromNodeId: idAdd, fromPortId: `${idAdd}-out-tensor`, toNodeId: idLn1, toPortId: `${idLn1}-in-x` },
        
        // Causal Attention wiring: x normalized goes to Q, K, and V
        { id: 'e7', fromNodeId: idLn1, fromPortId: `${idLn1}-out-tensor`, toNodeId: idAttn, toPortId: `${idAttn}-in-q` },
        { id: 'e8', fromNodeId: idLn1, fromPortId: `${idLn1}-out-tensor`, toNodeId: idAttn, toPortId: `${idAttn}-in-k` },
        { id: 'e9', fromNodeId: idLn1, fromPortId: `${idLn1}-out-tensor`, toNodeId: idAttn, toPortId: `${idAttn}-in-v` },
        
        // Residual 1: Add input to Attention (Add node input is Add [embeddings] output, plus Attention output)
        { id: 'e10', fromNodeId: idAdd, fromPortId: `${idAdd}-out-tensor`, toNodeId: idAddRes1, toPortId: `${idAddRes1}-in-a` },
        { id: 'e11', fromNodeId: idAttn, fromPortId: `${idAttn}-out-tensor`, toNodeId: idAddRes1, toPortId: `${idAddRes1}-in-b` },
        
        // LayerNorm 2 & MLP
        { id: 'e12', fromNodeId: idAddRes1, fromPortId: `${idAddRes1}-out-tensor`, toNodeId: idLn2, toPortId: `${idLn2}-in-x` },
        { id: 'e13', fromNodeId: idLn2, fromPortId: `${idLn2}-out-tensor`, toNodeId: idMlp, toPortId: `${idMlp}-in-x` },
        
        // Residual 2: Add input to MLP (Add node input is Residual 1 output, plus MLP output)
        { id: 'e14', fromNodeId: idAddRes1, fromPortId: `${idAddRes1}-out-tensor`, toNodeId: idAddRes2, toPortId: `${idAddRes2}-in-a` },
        { id: 'e15', fromNodeId: idMlp, fromPortId: `${idMlp}-out-tensor`, toNodeId: idAddRes2, toPortId: `${idAddRes2}-in-b` },
        
        // Output Classifier & Sampler
        { id: 'e16', fromNodeId: idAddRes2, fromPortId: `${idAddRes2}-out-tensor`, toNodeId: idHead, toPortId: `${idHead}-in-x` },
        { id: 'e17', fromNodeId: idHead, fromPortId: `${idHead}-out-tensor`, toNodeId: idSoft, toPortId: `${idSoft}-in-logits` },
        { id: 'e18', fromNodeId: idSoft, fromPortId: `${idSoft}-out-tensor`, toNodeId: idSamp, toPortId: `${idSamp}-in-probs` }
      );

      // Initialize default values for full parameters
      for (const n of newNodes) {
        if (n.params && n.params.n_embd) {
          n.params.n_embd = 64;
        }
      }
      addLog('info', 'Loaded "Full GPT Layer" preset.');
    }

    // Force ports creation update
    setNodes(newNodes.map(n => ({
      ...n,
      inputs: n.inputs.map(p => ({ ...p })),
      outputs: n.outputs.map(p => ({ ...p }))
    })));
    setEdges(newEdges);
  };

  const handleLoadTrainingData = (name: string) => {
    if (name === 'qa') {
      // Find text_input node and update its text
      const textNode = nodes.find(n => n.type === 'text_input');
      if (textNode) {
        setNodes(prev => prev.map(n => {
          if (n.type === 'text_input') {
            return { ...n, params: { ...n.params, text: qaText } };
          }
          return n;
        }));
        addLog('success', 'Loaded QA Training Dataset into Text Input node.');
      } else {
        addLog('error', 'Failed to load dataset: No Text Input node found on canvas.');
      }
    }
  };

  // Topological sorting helper
  const topologicalSort = (graphNodes: CanvasNode[], graphEdges: CanvasEdge[]): CanvasNode[] | null => {
    const sorted: CanvasNode[] = [];
    const visited = new Set<string>();
    const tempVisited = new Set<string>();

    const visit = (nodeId: string): boolean => {
      if (tempVisited.has(nodeId)) return false; // Cycle detected
      if (!visited.has(nodeId)) {
        tempVisited.add(nodeId);
        
        // Find nodes that connect to this node's inputs
        const incomingEdges = graphEdges.filter(e => e.toNodeId === nodeId);
        for (const edge of incomingEdges) {
          if (!visit(edge.fromNodeId)) return false;
        }
        
        tempVisited.delete(nodeId);
        visited.add(nodeId);
        const nodeObj = graphNodes.find(n => n.id === nodeId);
        if (nodeObj) sorted.push(nodeObj);
      }
      return true;
    };

    // Run topological sort from output endpoints (nodes with no outbound edges, or all nodes)
    for (const node of graphNodes) {
      if (!visit(node.id)) {
        addLog('error', 'Cycle detected in canvas blueprint. Execution halted.');
        return null;
      }
    }

    return sorted;
  };

  // Compile and run the pipeline
  const runPipeline = async (silent = false, promptOverride?: string): Promise<number | null> => {
    if (!engineRef.current || !tokenizerRef.current) {
      addLog('error', 'Cannot run: WebGPU engine not ready.');
      return null;
    }

    // Read the graph from refs so async generation loops always see the
    // latest nodes/edges (avoids stale-closure non-autoregressive bug).
    const graphNodes = nodesRef.current;
    const graphEdges = edgesRef.current;

    if (!silent) {
      setIsRunning(true);
      clearLogs();
    }
    if (!silent) {
      addLog('info', 'Compiling visual blueprint into WebGPU sequence...');
    }

    const sortedNodes = topologicalSort(graphNodes, graphEdges);
    if (!sortedNodes || sortedNodes.length === 0) {
      if (!silent) {
        setIsRunning(false);
      }
      return null;
    }

    if (!silent) {
      addLog('info', `Topological sorting completed. Found ${sortedNodes.length} active nodes.`);
    }
    const t0 = performance.now();

    // Map of executed port outputs: portId -> Tensor or string value
    const outputsMap = new Map<string, Tensor | string>();

    // Temporarily track updated node state to apply in one go at the end
    const nodeStateUpdates = new Map<string, Partial<CanvasNode>>();

    try {
      const engine = engineRef.current;
      const tokenizer = tokenizerRef.current;

      for (const node of sortedNodes) {
        if (!silent) {
          addLog('gpu', `[WebGPU] Dispatching compute shaders for: ${node.label}`);
        }
        nodeStateUpdates.set(node.id, { status: 'running' });

        // Retrieve inputs
        const inputValues: any[] = [];
        let missingInput = false;

        for (const inputPort of node.inputs) {
          const incomingEdge = graphEdges.find(e => e.toPortId === inputPort.id);
          if (!incomingEdge) {
            addLog('error', `Connection error: Input port "${inputPort.name}" on "${node.label}" is not connected.`);
            nodeStateUpdates.set(node.id, { status: 'error', errorMessage: `Missing link: ${inputPort.name}` });
            missingInput = true;
            break;
          }
          const val = outputsMap.get(incomingEdge.fromPortId);
          if (val === undefined) {
            addLog('error', `Execution error: Connected node failed to produce values for port "${inputPort.name}".`);
            nodeStateUpdates.set(node.id, { status: 'error' });
            missingInput = true;
            break;
          }
          inputValues.push(val);
        }

        if (missingInput) {
          throw new Error(`Execution halted on node ${node.label}`);
        }

        // Execute node logic
        let outputVal: Tensor | string | null = null;
        let previewText = '';
        let shape: number[] | undefined;

        switch (node.type) {
          case 'text_input': {
            outputVal = promptOverride !== undefined ? promptOverride : (node.params.text || '');
            previewText = `"${outputVal}"`;
            break;
          }

          case 'tokenizer': {
            const text = inputValues[0] as string;
            const tokens = tokenizer.encode(text);
            const gpuBuffer = gpuAvailable
              ? engine.createBuffer(new Uint32Array(tokens), GPUBufferUsage.STORAGE)
              : (null as any);
            
            outputVal = {
              shape: [1, tokens.length],
              buffer: gpuBuffer,
              dataType: 'uint32'
            };
            previewText = `[${tokens.join(', ')}]`;
            shape = [1, tokens.length];
            break;
          }

          case 'wte': {
            const tokens = inputValues[0] as Tensor;
            const vocabSize = parseInt(node.params.vocab_size) || 256;
            const n_embd = parseInt(node.params.n_embd) || 64;
            const seqLen = tokens.shape[1];

            shape = [1, seqLen, n_embd];

            if (gpuAvailable) {
              // Allocate/get embedding weights
              let cache = nodeWeights.current.get(node.id);
              if (!cache || !cache.wte || cache.wte.shape[0] !== vocabSize || cache.wte.shape[1] !== n_embd) {
                addLog('gpu', `[WebGPU] Allocating new Token Embedding weights matrix [${vocabSize}, ${n_embd}]`);
                const initialData = engine.randomNormal(vocabSize * n_embd, 0.0, 0.02);
                const weightBuffer = engine.createBuffer(initialData, GPUBufferUsage.STORAGE);
                cache = { wte: { shape: [vocabSize, n_embd], buffer: weightBuffer, dataType: 'float32' } };
                nodeWeights.current.set(node.id, cache);
              }

              const embeddingResult = engine.runWTE(tokens, cache.wte, vocabSize, n_embd);
              outputVal = embeddingResult;

              // Fetch a small slice of embedding output to preview in UI
              const sliceSize = Math.min(seqLen * n_embd, 12);
              const sliceData = await engine.readBuffer(embeddingResult.buffer, sliceSize * 4);
              previewText = Array.from(sliceData).map(v => v.toFixed(4)).join('\n') + '\n...';
            } else {
              outputVal = {
                shape,
                buffer: null as any,
                dataType: 'float32'
              };
              const sliceSize = Math.min(seqLen * n_embd, 12);
              previewText = Array.from({ length: sliceSize }, () => (Math.random() * 0.1 - 0.05).toFixed(4)).join('\n') + '\n...';
            }
            break;
          }

          case 'wpe': {
            const tokens = inputValues[0] as Tensor;
            const blockSize = parseInt(node.params.block_size) || 256;
            const n_embd = parseInt(node.params.n_embd) || 64;
            const seqLen = tokens.shape[1];

            shape = [1, seqLen, n_embd];

            if (gpuAvailable) {
              // Allocate/get positional weights
              let cache = nodeWeights.current.get(node.id);
              if (!cache || !cache.wpe || cache.wpe.shape[0] !== blockSize || cache.wpe.shape[1] !== n_embd) {
                addLog('gpu', `[WebGPU] Allocating new Position Embedding weights matrix [${blockSize}, ${n_embd}]`);
                const initialData = engine.randomNormal(blockSize * n_embd, 0.0, 0.02);
                const weightBuffer = engine.createBuffer(initialData, GPUBufferUsage.STORAGE);
                cache = { wpe: { shape: [blockSize, n_embd], buffer: weightBuffer, dataType: 'float32' } };
                nodeWeights.current.set(node.id, cache);
              }

              // Generate sequence positions array: 0, 1, 2, ..., seqLen-1
              const positions = new Uint32Array(seqLen);
              for (let t = 0; t < seqLen; t++) positions[t] = t % blockSize;
              const posTensor: Tensor = {
                shape: [seqLen],
                buffer: engine.createBuffer(positions, GPUBufferUsage.STORAGE),
                dataType: 'uint32'
              };

              const wpeResult = engine.runWPE(posTensor, cache.wpe, 1, seqLen, n_embd);
              outputVal = wpeResult;

              const sliceSize = Math.min(seqLen * n_embd, 12);
              const sliceData = await engine.readBuffer(wpeResult.buffer, sliceSize * 4);
              previewText = Array.from(sliceData).map(v => v.toFixed(4)).join('\n') + '\n...';
            } else {
              outputVal = {
                shape,
                buffer: null as any,
                dataType: 'float32'
              };
              const sliceSize = Math.min(seqLen * n_embd, 12);
              previewText = Array.from({ length: sliceSize }, () => (Math.random() * 0.1 - 0.05).toFixed(4)).join('\n') + '\n...';
            }
            break;
          }

          case 'add': {
            const a = inputValues[0] as Tensor;
            const b = inputValues[1] as Tensor;

            if (a.shape.join(',') !== b.shape.join(',')) {
              throw new Error(`Dimension mismatch: A ${JSON.stringify(a.shape)} does not match B ${JSON.stringify(b.shape)}`);
            }

            shape = a.shape;

            if (gpuAvailable) {
              const addResult = engine.runAdd(a, b);
              outputVal = addResult;

              const size = Math.min(addResult.shape.reduce((p,c) => p*c, 1), 12);
              const sliceData = await engine.readBuffer(addResult.buffer, size * 4);
              previewText = Array.from(sliceData).map(v => v.toFixed(4)).join('\n') + '\n...';
            } else {
              outputVal = {
                shape,
                buffer: null as any,
                dataType: 'float32'
              };
              const size = Math.min(shape.reduce((p,c) => p*c, 1), 12);
              previewText = Array.from({ length: size }, () => (Math.random() * 0.1 - 0.05).toFixed(4)).join('\n') + '\n...';
            }
            break;
          }

          case 'layernorm': {
            const x = inputValues[0] as Tensor;
            const epsilon = parseFloat(node.params.epsilon) || 1e-5;
            const C = x.shape[x.shape.length - 1];

            shape = x.shape;

            if (gpuAvailable) {
              // Setup gamma and beta parameter matrices
              let cache = nodeWeights.current.get(node.id);
              if (!cache || !cache.gamma || cache.gamma.shape[0] !== C) {
                addLog('gpu', `[WebGPU] Allocating LayerNorm weights (gamma=1.0, beta=0.0) of shape [${C}]`);
                const gammaData = new Float32Array(C).fill(1.0);
                const betaData = new Float32Array(C).fill(0.0);
                cache = {
                  gamma: { shape: [C], buffer: engine.createBuffer(gammaData, GPUBufferUsage.STORAGE), dataType: 'float32' },
                  beta: { shape: [C], buffer: engine.createBuffer(betaData, GPUBufferUsage.STORAGE), dataType: 'float32' }
                };
                nodeWeights.current.set(node.id, cache);
              }

              const normResult = engine.runLayerNorm(x, cache.gamma, cache.beta, epsilon);
              outputVal = normResult;

              const size = Math.min(normResult.shape.reduce((p,c) => p*c, 1), 12);
              const sliceData = await engine.readBuffer(normResult.buffer, size * 4);
              previewText = Array.from(sliceData).map(v => v.toFixed(4)).join('\n') + '\n...';
            } else {
              outputVal = {
                shape,
                buffer: null as any,
                dataType: 'float32'
              };
              const size = Math.min(shape.reduce((p,c) => p*c, 1), 12);
              previewText = Array.from({ length: size }, () => (Math.random() * 0.1 - 0.05).toFixed(4)).join('\n') + '\n...';
            }
            break;
          }

          case 'attention': {
            const q = inputValues[0] as Tensor;
            const k = inputValues[1] as Tensor;
            const v = inputValues[2] as Tensor;

            const n_head = parseInt(node.params.n_head) || 4;
            const head_size = parseInt(node.params.head_size) || 16;
            const C = q.shape[q.shape.length - 1];
            const B = q.shape[0] || 1;
            const T = q.shape[1];

            if (n_head * head_size !== C) {
              throw new Error(`Dimension config error: n_head (${n_head}) * head_size (${head_size}) must equal input channels C (${C}).`);
            }

            shape = [B, T, C];

            if (gpuAvailable) {
              // Project inputs (simulate QKV projections inside the module block)
              let cache = nodeWeights.current.get(node.id);
              if (!cache || !cache.wq || cache.wq.shape[0] !== C) {
                addLog('gpu', `[WebGPU] Allocating Attention QKV weights [${C}, ${C}]`);
                const wqData = engine.randomNormal(C * C, 0.0, 0.02);
                const wkData = engine.randomNormal(C * C, 0.0, 0.02);
                const wvData = engine.randomNormal(C * C, 0.0, 0.02);
                const woData = new Float32Array(C * C).fill(0.0); // near-zero so residual dominates at init

                cache = {
                  wq: { shape: [C, C], buffer: engine.createBuffer(wqData, GPUBufferUsage.STORAGE), dataType: 'float32' },
                  wk: { shape: [C, C], buffer: engine.createBuffer(wkData, GPUBufferUsage.STORAGE), dataType: 'float32' },
                  wv: { shape: [C, C], buffer: engine.createBuffer(wvData, GPUBufferUsage.STORAGE), dataType: 'float32' },
                  wo: { shape: [C, C], buffer: engine.createBuffer(woData, GPUBufferUsage.STORAGE), dataType: 'float32' }
                };
                nodeWeights.current.set(node.id, cache);
              }

              // Runs MatMuls on GPU to obtain Q, K, and V projections
              const qProj = engine.runMatMul(q, cache.wq, null, B * T, C, C);
              const kProj = engine.runMatMul(k, cache.wk, null, B * T, C, C);
              const vProj = engine.runMatMul(v, cache.wv, null, B * T, C, C);

              // Execute Attention Core WGSL Shader
              const attnResult = engine.runCausalSelfAttention(qProj, kProj, vProj, B, n_head, T, head_size);
              
              // Re-flatten attn output from [B, n_head, T, head_size] back to [B * T, C]
              const attnFlattened: Tensor = {
                shape: [B * T, C],
                buffer: attnResult.buffer,
                dataType: 'float32'
              };

              // Run output projection
              const finalProj = engine.runMatMul(attnFlattened, cache.wo, null, B * T, C, C);
              finalProj.shape = [B, T, C];
              
              outputVal = finalProj;

              const sliceSize = Math.min(B * T * C, 12);
              const sliceData = await engine.readBuffer(finalProj.buffer, sliceSize * 4);
              previewText = Array.from(sliceData).map(v => v.toFixed(4)).join('\n') + '\n...';
            } else {
              outputVal = {
                shape,
                buffer: null as any,
                dataType: 'float32'
              };
              const sliceSize = Math.min(B * T * C, 12);
              previewText = Array.from({ length: sliceSize }, () => (Math.random() * 0.1 - 0.05).toFixed(4)).join('\n') + '\n...';
            }
            break;
          }

          case 'mlp': {
            const x = inputValues[0] as Tensor;
            const C = x.shape[x.shape.length - 1];
            const B = x.shape[0] || 1;
            const T = x.shape[1];
            const mult = parseInt(node.params.multiplier) || 4;
            const hiddenDim = C * mult;

            shape = [B, T, C];

            if (gpuAvailable) {
              let cache = nodeWeights.current.get(node.id);
              if (!cache || !cache.w1 || cache.w1.shape[1] !== hiddenDim) {
                addLog('gpu', `[WebGPU] Allocating MLP Layer weights [${C}, ${hiddenDim}] and [${hiddenDim}, ${C}]`);
                const w1Data = engine.randomNormal(C * hiddenDim, 0.0, 0.02);
                const b1Data = new Float32Array(hiddenDim).fill(0.0);
                const w2Data = new Float32Array(hiddenDim * C).fill(0.0); // near-zero so residual dominates at init
                const b2Data = new Float32Array(C).fill(0.0);

                cache = {
                  w1: { shape: [C, hiddenDim], buffer: engine.createBuffer(w1Data, GPUBufferUsage.STORAGE), dataType: 'float32' },
                  b1: { shape: [hiddenDim], buffer: engine.createBuffer(b1Data, GPUBufferUsage.STORAGE), dataType: 'float32' },
                  w2: { shape: [hiddenDim, C], buffer: engine.createBuffer(w2Data, GPUBufferUsage.STORAGE), dataType: 'float32' },
                  b2: { shape: [C], buffer: engine.createBuffer(b2Data, GPUBufferUsage.STORAGE), dataType: 'float32' }
                };
                nodeWeights.current.set(node.id, cache);
              }

              // Layer 1 projection
              const proj1 = engine.runMatMul(x, cache.w1, cache.b1, B * T, hiddenDim, C);
              
              // Activation GeLU
              const act = engine.runGELU(proj1);

              // Layer 2 projection
              const proj2 = engine.runMatMul(act, cache.w2, cache.b2, B * T, C, hiddenDim);

              outputVal = proj2;

              const sliceSize = Math.min(B * T * C, 12);
              const sliceData = await engine.readBuffer(proj2.buffer, sliceSize * 4);
              previewText = Array.from(sliceData).map(v => v.toFixed(4)).join('\n') + '\n...';
            } else {
              outputVal = {
                shape,
                buffer: null as any,
                dataType: 'float32'
              };
              const sliceSize = Math.min(B * T * C, 12);
              previewText = Array.from({ length: sliceSize }, () => (Math.random() * 0.1 - 0.05).toFixed(4)).join('\n') + '\n...';
            }
            break;
          }

          case 'lm_head': {
            const x = inputValues[0] as Tensor;
            const C = x.shape[x.shape.length - 1];
            const B = x.shape[0] || 1;
            const T = x.shape[1];

            // Re-allocate separate head or tie weights to WTE if WTE exists
            const wteNode = graphNodes.find(n => n.type === 'wte');
            let vocabSize = 256;
            if (wteNode) {
              vocabSize = parseInt(wteNode.params.vocab_size) || 256;
            }

            shape = [B, T, vocabSize];

            if (gpuAvailable) {
              let cache = nodeWeights.current.get(node.id);
              if (!cache || !cache.w_head || cache.w_head.shape[1] !== vocabSize) {
                // Allocate custom weights matrix instead of reusing WTE directly to prevent scrambling due to transposition mismatch
                addLog('gpu', `[WebGPU] Allocating custom LM Head projection weight matrix [${C}, ${vocabSize}]`);
                const wData = engine.randomNormal(C * vocabSize, 0.0, 0.02);
                const wHeadTensor: Tensor = {
                  shape: [C, vocabSize],
                  buffer: engine.createBuffer(wData, GPUBufferUsage.STORAGE),
                  dataType: 'float32'
                };

                const newCache: Record<string, Tensor> = {
                  ...(cache || {}),
                  w_head: wHeadTensor
                };
                nodeWeights.current.set(node.id, newCache);
                cache = newCache;
              }

              const wHead = cache.w_head;
              // Logits MatMul
              const logits = engine.runMatMul(x, wHead, null, B * T, vocabSize, C);
              outputVal = logits;

              const sliceSize = Math.min(B * T * vocabSize, 12);
              const sliceData = await engine.readBuffer(logits.buffer, sliceSize * 4);
              previewText = Array.from(sliceData).map(v => v.toFixed(4)).join('\n') + '\n...';
            } else {
              outputVal = {
                shape,
                buffer: null as any,
                dataType: 'float32'
              };
              const sliceSize = Math.min(B * T * vocabSize, 12);
              previewText = Array.from({ length: sliceSize }, () => (Math.random() * 0.1 - 0.05).toFixed(4)).join('\n') + '\n...';
            }
            break;
          }

          case 'softmax': {
            const logits = inputValues[0] as Tensor;
            const B = logits.shape[0] || 1;
            const T = logits.shape[1];
            const vocabSize = logits.shape[logits.shape.length - 1];

            shape = [B, vocabSize];

            if (gpuAvailable) {
              let softmaxInput: Tensor = logits;

              // If logits shape is [B, T, vocabSize], slice the last sequence element [B, vocabSize]
              if (logits.shape.length === 3) {
                if (!silent) {
                  addLog('gpu', `[WebGPU] Slicing logits at sequence index T-1 (${T - 1}) to shape [${B}, ${vocabSize}] for classification`);
                }
                const sliceBuffer = engine.createEmptyBuffer(B * vocabSize * 4, GPUBufferUsage.STORAGE);
                const commandEncoder = engine.device!.createCommandEncoder();
                
                // Copy slice: Offset in bytes: (T - 1) * vocabSize * 4
                commandEncoder.copyBufferToBuffer(
                  logits.buffer,
                  (T - 1) * vocabSize * 4,
                  sliceBuffer,
                  0,
                  B * vocabSize * 4
                );
                engine.device!.queue.submit([commandEncoder.finish()]);

                softmaxInput = { shape: [B, vocabSize], buffer: sliceBuffer, dataType: 'float32' };
              }

              const probs = engine.runSoftmax(softmaxInput, B, vocabSize);
              outputVal = probs;

              // Preview top 10 probabilities
              const sliceData = await engine.readBuffer(probs.buffer, vocabSize * 4);
              const sortedWithIndex = Array.from(sliceData)
                .map((val, idx) => ({ val, idx }))
                .sort((a, b) => b.val - a.val)
                .slice(0, 5);

              previewText = sortedWithIndex
                .map(item => `token ${item.idx} (byte "${tokenizer.decode([item.idx])}"): ${(item.val * 100).toFixed(2)}%`)
                .join('\n');
            } else {
              outputVal = {
                shape,
                buffer: null as any,
                dataType: 'float32'
              };
              // Sim preview top probabilities
              const simProbs = Array.from({ length: 5 }, (_, idx) => ({
                idx: idx + 65, // ASCII 'A', 'B', etc.
                val: 0.8 - idx * 0.15
              }));
              previewText = simProbs
                .map(item => `token ${item.idx} (byte "${tokenizer.decode([item.idx])}"): ${(item.val * 100).toFixed(2)}%`)
                .join('\n');
            }
            break;
          }

          case 'sampler': {
            const probs = inputValues[0] as Tensor;
            const vocabSize = probs.shape[probs.shape.length - 1];
            shape = [1, 1];

            let sampledTokenId = 32; // Default space character

            if (gpuAvailable) {
              const temp = parseFloat(node.params.temperature) || 1.0;
              const topK = parseInt(node.params.top_k) || 50;

              // Download probabilities to CPU for sampling
              const probsData = await engine.readBuffer(probs.buffer, vocabSize * 4);

              // Apply Top-K sampling restricted to printable ASCII range & newlines
              const indexedProbs = Array.from(probsData)
                .map((p, idx) => ({ p, idx }))
                .filter(item => item.idx === 10 || item.idx === 13 || (item.idx >= 32 && item.idx <= 126));

              // Repetition penalty: down-weight tokens generated in the recent
              // window so the model cannot lock into a degenerate repeating loop.
              const recent = genHistoryRef.current;
              if (recent.length > 0) {
                const window = recent.slice(-24);
                const counts = new Map<number, number>();
                for (const t of window) counts.set(t, (counts.get(t) || 0) + 1);
                const repPenalty = 1.6;
                for (const item of indexedProbs) {
                  const c = counts.get(item.idx);
                  if (c) item.p /= Math.pow(repPenalty, c);
                }
              }

              indexedProbs.sort((a, b) => b.p - a.p);
              
              // Keep only Top-K
              const topKProbs = indexedProbs.slice(0, topK);
              
              // Apply temperature scaling: p_new = p^(1/temp)
              const scaledProbs = topKProbs.map(item => ({ p: Math.pow(item.p, 1 / temp), idx: item.idx }));
              let sum = scaledProbs.reduce((acc, item) => acc + item.p, 0);
              
              // Renormalize
              const normProbs = scaledProbs.map(item => ({ p: item.p / (sum || 1), idx: item.idx }));

              // Sample token using multinomial choice
              const r = Math.random();
              let acc = 0;
              sampledTokenId = normProbs[0].idx;
              for (const item of normProbs) {
                acc += item.p;
                if (r <= acc) {
                  sampledTokenId = item.idx;
                  break;
                }
              }

              // Allocate sampled token buffer
              const outputBuffer = engine.createBuffer(new Uint32Array([sampledTokenId]), GPUBufferUsage.STORAGE);
              
              outputVal = {
                shape: [1, 1],
                buffer: outputBuffer,
                dataType: 'uint32'
              };
            }

            previewText = `token ID: ${sampledTokenId}\nbyte: "${tokenizer.decode([sampledTokenId])}"`;
            
            // Save generation output to state
            const charOutput = tokenizer.decode([sampledTokenId]);
            setGeneratedText(prev => prev + charOutput);
            break;
          }
        }

        // Register output
        if (outputVal) {
          for (const outPort of node.outputs) {
            outputsMap.set(outPort.id, outputVal);
          }
        }

        nodeStateUpdates.set(node.id, {
          status: 'success',
          outputShape: shape,
          outputPreview: previewText,
        });
      }

      // Store intermediate outputs for training access
      latestOutputs.current = outputsMap;

      const t1 = performance.now();
      const elapsed = t1 - t0;
      if (!silent) {
        setExecutionTimeMs(elapsed);
      }
      if (!silent) {
        addLog('success', `${gpuAvailable ? 'WebGPU inference' : 'Simulated CPU'} pass executed successfully in ${elapsed.toFixed(2)} ms.`);
      }
      if (!silent) {
        setIsRunning(false);
      }

      // Apply all visual node states updates in one batch
      setNodes(prev => prev.map(n => {
        const up = nodeStateUpdates.get(n.id);
        if (up) return { ...n, ...up };
        return n;
      }));

      return elapsed;
    } catch (e: any) {
      console.error(e);
      addLog('error', `Execution crashed: ${e.message || e}`);
      if (!silent) {
        setIsRunning(false);
      }
      
      // Update nodes states to error on crash
      setNodes(prev => prev.map(n => {
        const up = nodeStateUpdates.get(n.id);
        if (up) {
          return {
            ...n,
            status: up.status === 'running' ? 'error' : (up.status || n.status),
            errorMessage: up.status === 'running' ? 'Execution failed' : up.errorMessage,
          };
        }
        return n;
      }));

      return null;
    }
  };

  // Real next-token training: learns the Text Input dataset with a correct
  // character-level Transformer (full backprop + Adam). This is what makes the
  // chat actually answer from the data. The WebGPU node graph stays as the
  // live visualization of a forward pass.
  const runTraining = async () => {
    if (!tokenizerRef.current) {
      addLog('error', 'Cannot train: engine not ready.');
      return;
    }

    const textNode = nodes.find(n => n.type === 'text_input');
    if (!textNode) {
      addLog('error', 'Training requires a Text Input node on the canvas.');
      return;
    }

    const text = (textNode.params.text || '').trim();
    if (text.length < 8) {
      addLog('error', 'Load a dataset into the Text Input node first (use "Load QA Dataset").');
      return;
    }

    // The canvas IS the model: compile the node graph you wired into a concrete
    // architecture and validate its topology. If the graph is incomplete or
    // mis-wired we block training and explain exactly what to fix.
    const compiled = compileGraph(nodes, edges);
    if (!compiled.config) {
      addLog('error', 'Cannot train — your canvas architecture is incomplete:');
      for (const msg of compiled.errors) addLog('error', `  • ${msg}`);
      return;
    }
    const arch = compiled.config;

    setIsTraining(true);
    setIsModelTrained(false);
    clearLogs();

    addLog('info', `Architecture compiled from canvas — nEmbd ${arch.nEmbd} | block ${arch.blockSize} | mlp x${arch.mlpMult} | blocks ${arch.nLayer}.`);
    for (const w of compiled.warnings) addLog('warning', w);

    // (Re)build a fresh model whose architecture is read from the canvas graph.
    const model = new CharTransformer({
      nEmbd: arch.nEmbd,
      blockSize: arch.blockSize,
      mlpMult: arch.mlpMult,
      nLayer: arch.nLayer,
      lr: trainParams.lr,
    });
    model.setText(text);
    cpuModelRef.current = model;
    gpuModelRef.current = null;

    const steps = Math.max(1, Math.floor(trainParams.steps));
    const batch = Math.max(1, Math.floor(trainParams.batch));

    // Decide backend. WebGPU is verified against the CPU math before use; if it
    // diverges or errors, we transparently fall back to the CPU trainer.
    let backend = computeBackendRef.current;
    let gpu: GpuTransformer | null = null;

    // The WebGPU mirror only implements a single Transformer block. For deeper
    // architectures wired on the canvas, train on the (block-agnostic) CPU engine.
    if (backend === 'webgpu' && arch.nLayer > 1) {
      addLog('warning', `WebGPU supports single-block models only — training this ${arch.nLayer}-block architecture on CPU.`);
      backend = 'cpu';
    }

    if (backend === 'webgpu') {
      const device = engineRef.current?.device ?? null;
      if (!gpuAvailable || !device) {
        addLog('warning', 'WebGPU device unavailable — falling back to CPU backend.');
        backend = 'cpu';
      } else {
        try {
          gpu = new GpuTransformer(device, model.cfg, model.vocab);
          gpu.loadFromCPU(model); // identical initial weights for a fair parity check

          // Parity self-check: one fixed sequence, compare GPU vs CPU logits+grad.
          const probe = model.tokens.slice(0, Math.min(model.cfg.blockSize, model.tokens.length));
          const inputs = probe.slice(0, probe.length - 1);
          const targets = probe.slice(1);
          const cpuRef = model.debugForwardBackward(inputs, targets);
          const gpuRef = await gpu.debugForwardBackward(inputs, targets);

          let maxLogitDiff = 0;
          for (let i = 0; i < cpuRef.logits.length; i++) {
            maxLogitDiff = Math.max(maxLogitDiff, Math.abs(cpuRef.logits[i] - gpuRef.logits[i]));
          }
          let maxGradDiff = 0;
          for (let i = 0; i < cpuRef.wheadGrad.length; i++) {
            maxGradDiff = Math.max(maxGradDiff, Math.abs(cpuRef.wheadGrad[i] - gpuRef.wheadGrad[i]));
          }
          addLog('info', `WebGPU parity check | logits Δ=${maxLogitDiff.toExponential(2)} | grad Δ=${maxGradDiff.toExponential(2)} | loss CPU ${cpuRef.loss.toFixed(4)} / GPU ${gpuRef.loss.toFixed(4)}`);

          if (maxLogitDiff > 1e-2 || maxGradDiff > 1e-2 || !isFinite(gpuRef.loss)) {
            addLog('warning', 'WebGPU kernels diverged from CPU reference — falling back to CPU backend for correctness.');
            backend = 'cpu';
            gpu = null;
          } else {
            addLog('success', 'WebGPU kernels match CPU reference. Training on GPU.');
            gpu.loadFromCPU(model); // reset weights/moments after the probe step
            gpuModelRef.current = gpu;
          }
        } catch (e: any) {
          console.error(e);
          addLog('warning', `WebGPU init failed (${e.message || e}) — falling back to CPU backend.`);
          backend = 'cpu';
          gpu = null;
        }
      }
    }

    addLog('info', `Training real transformer on ${text.length} chars | vocab ${model.vocab} | ${steps} steps | backend: ${backend.toUpperCase()}`);
    addLog('info', `Hyperparameters: nEmbd ${arch.nEmbd} | block ${arch.blockSize} | mlp x${arch.mlpMult} | blocks ${arch.nLayer} | lr ${trainParams.lr} | batch ${batch}`);
    addLog('info', `Architecture: WTE+WPE -> [LN -> Self-Attention -> MLP(GELU) -> LN] x${arch.nLayer} -> LN -> Head | Optimizer: Adam`);

    try {
      const t0 = performance.now();
      let lastLoss = 0;
      for (let step = 1; step <= steps; step++) {
        const isLogStep = step === 1 || step % 100 === 0 || step === steps;
        // On GPU, only read the loss back on log steps so the pipeline never stalls.
        const loss = gpu ? await gpu.trainStep(model.tokens, batch, isLogStep) : model.trainStep(batch);
        if (loss >= 0) lastLoss = loss;

        if (isLogStep) {
          addLog('gpu', `[Training] Step ${step}/${steps} | Loss: ${lastLoss.toFixed(4)}`);
          // Yield to the UI so the log + spinner update and the tab stays responsive.
          await new Promise(r => setTimeout(r, 0));
        }
      }
      const secs = ((performance.now() - t0) / 1000).toFixed(1);

      setIsModelTrained(true);
      addLog('success', `Training complete in ${secs}s on ${backend.toUpperCase()} (final loss ${lastLoss.toFixed(4)}). Ask a question in Chat to talk to your model!`);

      // Quick self-test so the user immediately sees it learned.
      const sample = await generateAnswer('Q: who are you?\nA:', 120);
      addLog('info', `Self-test "who are you?" -> ${sample.trim()}`);
    } catch (err: any) {
      console.error(err);
      addLog('error', `Training failed: ${err.message || err}`);
    } finally {
      setIsTraining(false);
    }
  };

  // Generate an answer using whichever backend trained the model. Returns the
  // full answer string (used for the post-training self-test).
  const generateAnswer = async (prompt: string, maxNew = 160): Promise<string> => {
    const cpu = cpuModelRef.current;
    if (!cpu) return '';
    const gpu = gpuModelRef.current;
    if (!gpu) {
      return cpu.generate(prompt, maxNew, 0.6, 8, '\n');
    }
    // GPU forward for logits, sampling on the CPU side using the shared vocab.
    const T = cpu.cfg.blockSize;
    let ids = cpu.encode(prompt);
    let out = '';
    for (let s = 0; s < maxNew; s++) {
      const ctx = ids.slice(Math.max(0, ids.length - T));
      const logits = await gpu.logitsForContext(ctx);
      const V = cpu.vocab;
      const last = (ctx.length - 1) * V;
      const ch = sampleFromLogits(logits, last, V, cpu, 0.6, 8);
      out += ch.char;
      ids.push(ch.id);
      if (ch.char === '\n' || out.length > 200) break;
    }
    return out;
  };

  // Top-k temperature sampling from a logits slice; returns char + token id.
  const sampleFromLogits = (logits: Float32Array, off: number, V: number, cpu: CharTransformer, temperature: number, topK: number) => {
    const scaled = new Float32Array(V);
    let maxL = -Infinity;
    for (let j = 0; j < V; j++) { scaled[j] = logits[off + j] / Math.max(1e-6, temperature); if (scaled[j] > maxL) maxL = scaled[j]; }
    let sum = 0;
    const probs = new Float32Array(V);
    for (let j = 0; j < V; j++) { probs[j] = Math.exp(scaled[j] - maxL); sum += probs[j]; }
    for (let j = 0; j < V; j++) probs[j] /= sum;
    const idx = Array.from({ length: V }, (_, j) => j).sort((a, b) => probs[b] - probs[a]).slice(0, Math.min(topK, V));
    let pSum = 0;
    for (const j of idx) pSum += probs[j];
    let r = Math.random() * pSum;
    let chosen = idx[0];
    for (const j of idx) { r -= probs[j]; if (r <= 0) { chosen = j; break; } }
    return { id: chosen, char: cpu.decode([chosen]) };
  };

  // Chat with the trained character-level Transformer. Generates the answer
  // token-by-token from the real learned weights, streaming it into the UI.
  const runChatInference = async (userMsg: string) => {
    const model = cpuModelRef.current;
    if (!model || !isModelTrained) {
      addLog('error', 'No trained model yet. Click "Load QA Dataset", then "Train" before chatting.');
      return;
    }

    setIsChatResponding(true);

    // Format prompt to match the training Q&A structure.
    const formattedPrompt = userMsg.includes('Q:') ? userMsg : `Q: ${userMsg}\nA:`;

    // Add user message and an assistant placeholder.
    setChatHistory(prev => [
      ...prev,
      { role: 'user', content: userMsg },
      { role: 'assistant', content: '' },
    ]);

    // Reflect the prompt in the canvas Text Input node and kick off a visual
    // forward pass so the node graph animates while the model thinks.
    setNodes(prev => prev.map(n =>
      n.type === 'text_input' ? { ...n, params: { ...n.params, text: formattedPrompt } } : n
    ));
    if (gpuAvailable) {
      runPipeline(true, formattedPrompt).catch(() => {});
    }

    try {
      let answer = '';
      const T = model.cfg.blockSize;
      let ids = model.encode(formattedPrompt);
      const gpu = gpuModelRef.current;

      const maxNew = 160;
      for (let step = 0; step < maxNew; step++) {
        const ctx = ids.slice(Math.max(0, ids.length - T));

        // Generate exactly one character from the active backend.
        let ch: string;
        let tokenId: number;
        if (gpu) {
          const logits = await gpu.logitsForContext(ctx);
          const V = model.vocab;
          const s = sampleFromLogits(logits, (ctx.length - 1) * V, V, model, 0.5, 6);
          ch = s.char; tokenId = s.id;
        } else {
          const V = model.vocab;
          const logits = model.logitsForContextCPU(ctx);
          const s = sampleFromLogits(logits, (ctx.length - 1) * V, V, model, 0.5, 6);
          ch = s.char; tokenId = s.id;
        }
        if (!ch) break;
        answer += ch;
        ids.push(tokenId);

        // Stream into the chat bubble.
        setChatHistory(prev => {
          const next = [...prev];
          if (next.length > 0) next[next.length - 1] = { role: 'assistant', content: answer };
          return next;
        });

        // Stop at end of the answer line.
        if (ch === '\n' || answer.length > 200) break;
        // Small delay so the user sees it typing.
        await new Promise(r => setTimeout(r, 18));
      }

      addLog('success', `Model replied: ${answer.trim()}`);
    } catch (err: any) {
      console.error(err);
      addLog('error', `Chat failed: ${err.message || err}`);
    } finally {
      setIsChatResponding(false);
    }
  };

  // Iterative generation loops (Auto-Gen)
  const handleAutoGenerateToggle = (enabled: boolean) => {
    setIsAutoGenerating(enabled);
    if (!enabled) {
      if (autoGenTimerRef.current) {
        window.clearTimeout(autoGenTimerRef.current);
        autoGenTimerRef.current = null;
      }
      addLog('info', 'Auto-generation stopped.');
      return;
    }

    if (!gpuAvailable) {
      addLog('error', 'Auto-Gen requires WebGPU. No GPU device available.');
      setIsAutoGenerating(false);
      return;
    }

    const hasSampler = nodes.some(n => n.type === 'sampler');
    if (!hasSampler) {
      addLog('error', 'Auto-Gen requires a Sampler node. Load the Full GPT Layer preset first.');
      setIsAutoGenerating(false);
      return;
    }

    setGeneratedText('');
    addLog('info', 'Starting iterative generation loop...');

    // Find the text input node to read/append tokens
    const textNode = nodes.find(n => n.type === 'text_input');
    if (!textNode) {
      addLog('error', 'Auto-Gen error: No Text Input node found on the canvas.');
      setIsAutoGenerating(false);
      return;
    }

    let currentPrompt = textNode.params.text || '';
    setGeneratedText(currentPrompt);

    // Seed repetition-penalty history with the prompt tokens.
    genHistoryRef.current = tokenizerRef.current
      ? tokenizerRef.current.encode(currentPrompt)
      : [];

    const loop = async () => {
      // Execute real WebGPU forward pass, feeding back the growing prompt so
      // each step is conditioned on previously generated tokens.
      const elapsed = await runPipeline(true, currentPrompt);
      if (elapsed === null) {
        setIsAutoGenerating(false);
        return;
      }

      // Read back the real sampled token ID directly from the GPU buffer.
      const samplerNode = nodesRef.current.find(n => n.type === 'sampler');
      const samplerOutputTensor = samplerNode
        ? (latestOutputs.current.get(`${samplerNode.id}-out-tokens`) as Tensor | undefined)
        : undefined;

      if (samplerOutputTensor && tokenizerRef.current && engineRef.current) {
        const tokenArray = await engineRef.current.readBufferUint(samplerOutputTensor.buffer, 4);
        const tokenId = tokenArray[0];
        const nextChar = tokenizerRef.current.decode([tokenId]);

        // Track for repetition penalty.
        genHistoryRef.current.push(tokenId);

        currentPrompt += nextChar;

        // Limit length to prevent GPU context overflows
        if (currentPrompt.length > 150) {
          addLog('warning', 'Prompt reached maximum safety limit (150 chars). Auto-Gen completed.');
          setGeneratedText(currentPrompt);
          setIsAutoGenerating(false);
          return;
        }

        // Reflect generated text in the UI and the canvas text node.
        setGeneratedText(currentPrompt);
        setNodes(prev => prev.map(n => {
          if (n.type === 'text_input') {
            return { ...n, params: { ...n.params, text: currentPrompt } };
          }
          return n;
        }));

        // Recurse next token after a brief dynamic delay for visual rendering
        autoGenTimerRef.current = window.setTimeout(loop, 120);
      } else {
        setIsAutoGenerating(false);
      }
    };

    loop();
  };

  return (
    <div className="w-full h-full flex flex-col bg-zinc-950 text-zinc-100 antialiased overflow-hidden">
      {/* Header bar */}
      <header className="h-[52px] border-b border-zinc-800 flex items-center justify-between px-4 bg-zinc-900 shrink-0 select-none">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center">
            <SparkleIcon size={18} className="text-emerald-400" />
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tight text-white flex items-center gap-1.5 leading-none">
              OpenLow
              <span className="text-[9px] bg-zinc-800 border border-zinc-700 text-zinc-400 px-1.5 py-0.5 rounded font-mono font-medium">
                v1.0.0
              </span>
            </h1>
            <p className="text-[10px] text-zinc-500 font-mono mt-0.5">WEBGPU DRAG & DROP LLM ARCHITECT</p>
          </div>
        </div>

        {/* Short top bar instruction */}
        <div className="text-[10px] text-zinc-500 font-mono hidden md:block max-w-[400px] truncate">
          Graph traverses and executes WGSL shaders: WTE &rarr; WPE &rarr; LN &rarr; Multi-Head Attention &rarr; MLP &rarr; LM Head
        </div>
      </header>

      {/* Main workspace */}
      <div className="flex-1 flex overflow-hidden min-h-0 relative">
        <Sidebar onAddNodeClick={(type) => handleAddNode(type, 80, 80)} />
        
        <Canvas
          nodes={nodes}
          edges={edges}
          selectedNodeId={selectedNodeId}
          onSelectNode={setSelectedNodeId}
          onUpdateNodes={handleUpdateNodes}
          onUpdateEdges={handleUpdateEdges}
          onAddNode={(type, x, y) => handleAddNode(type, x, y)}
        />

        <NodeInspector
          node={nodes.find(n => n.id === selectedNodeId) || null}
          onUpdateNodeParams={handleUpdateNodeParams}
        />
      </div>

      {/* Bottom Logger & Run Console */}
      <ControlPanel
        onRun={runPipeline}
        onTrain={runTraining}
        isTraining={isTraining}
        onAutoGenerateToggle={handleAutoGenerateToggle}
        isAutoGenerating={isAutoGenerating}
        onClearCanvas={handleClearCanvas}
        onLoadPreset={loadPreset}
        onLoadTrainingData={handleLoadTrainingData}
        logs={logs}
        gpuAvailable={gpuAvailable}
        executionTimeMs={executionTimeMs}
        isRunning={isRunning}
        generatedText={generatedText}
        chatHistory={chatHistory}
        isChatResponding={isChatResponding}
        onSendChatMessage={runChatInference}
        computeBackend={computeBackend}
        onSetComputeBackend={setComputeBackend}
        trainParams={trainParams}
        onSetTrainParams={setTrainParams}
      />
    </div>
  );
}

export default App;
