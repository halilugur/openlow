/// <reference types="@webgpu/types" />
// WebGPU Execution Engine for Visual LLM Builder
import * as Shaders from './shaders';

export interface Tensor {
  shape: number[];
  buffer: GPUBuffer;
  dataType: 'float32' | 'uint32';
}

export class WebGPUEngine {
  device: GPUDevice | null = null;
  pipelineCache: Map<string, GPUComputePipeline> = new Map();
  weightsCache: Map<string, Tensor> = new Map();

  async init(): Promise<boolean> {
    if (!navigator.gpu) {
      console.warn("WebGPU not supported in this browser.");
      return false;
    }
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return false;
      this.device = await adapter.requestDevice();
      return true;
    } catch (e) {
      console.error("Failed to initialize WebGPU:", e);
      return false;
    }
  }

  // Box-Muller transform for normal distribution weight initialization
  randomNormal(size: number, mean = 0.0, std = 0.02): Float32Array {
    const arr = new Float32Array(size);
    for (let i = 0; i < size; i += 2) {
      const u1 = Math.random() || 0.0001; // Avoid 0
      const u2 = Math.random();
      const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
      
      arr[i] = z0 * std + mean;
      if (i + 1 < size) {
        arr[i + 1] = z1 * std + mean;
      }
    }
    return arr;
  }

  createBuffer(data: Float32Array | Uint32Array, usage: number): GPUBuffer {
    if (!this.device) throw new Error("WebGPU device not initialized");
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: false,
    });
    this.device.queue.writeBuffer(buffer, 0, data as any);
    return buffer;
  }

  createEmptyBuffer(sizeBytes: number, usage: number): GPUBuffer {
    if (!this.device) throw new Error("WebGPU device not initialized");
    return this.device.createBuffer({
      size: sizeBytes,
      usage: usage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }

  getOrCreatePipeline(label: string, shaderCode: string): GPUComputePipeline {
    if (!this.device) throw new Error("WebGPU device not initialized");
    if (this.pipelineCache.has(label)) {
      return this.pipelineCache.get(label)!;
    }

    const shaderModule = this.device.createShaderModule({
      label,
      code: shaderCode,
    });
    const pipeline = this.device.createComputePipeline({
      label,
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });
    this.pipelineCache.set(label, pipeline);
    return pipeline;
  }

  async readBuffer(buffer: GPUBuffer, sizeBytes: number): Promise<Float32Array> {
    if (!this.device) throw new Error("WebGPU device not initialized");
    
    // Create mapping buffer
    const readBuffer = this.device.createBuffer({
      size: sizeBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, sizeBytes);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const copyArray = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    readBuffer.destroy();
    
    return copyArray;
  }

  async readBufferUint(buffer: GPUBuffer, sizeBytes: number): Promise<Uint32Array> {
    if (!this.device) throw new Error("WebGPU device not initialized");
    
    const readBuffer = this.device.createBuffer({
      size: sizeBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, sizeBytes);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const copyArray = new Uint32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    readBuffer.destroy();
    
    return copyArray;
  }

  // --- Compute Operations ---

  runWTE(tokens: Tensor, weights: Tensor, vocabSize: number, n_embd: number): Tensor {
    if (!this.device) throw new Error("WebGPU device not initialized");
    const B = tokens.shape[0];
    const T = tokens.shape[1];
    const outputShape = [B, T, n_embd];
    const outputSizeBytes = B * T * n_embd * 4;
    const outputBuffer = this.createEmptyBuffer(outputSizeBytes, GPUBufferUsage.STORAGE);

    const pipeline = this.getOrCreatePipeline('wte_compute', Shaders.WTE_SHADER);

    const paramsBuffer = this.createBuffer(
      new Uint32Array([n_embd, vocabSize]),
      GPUBufferUsage.UNIFORM
    );

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tokens.buffer } },
        { binding: 1, resource: { buffer: weights.buffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil((B * T * n_embd) / 64));
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    return { shape: outputShape, buffer: outputBuffer, dataType: 'float32' };
  }

  runWPE(positions: Tensor, weights: Tensor, B: number, T: number, n_embd: number): Tensor {
    if (!this.device) throw new Error("WebGPU device not initialized");
    const outputShape = [B, T, n_embd];
    const outputSizeBytes = B * T * n_embd * 4;
    const outputBuffer = this.createEmptyBuffer(outputSizeBytes, GPUBufferUsage.STORAGE);

    const pipeline = this.getOrCreatePipeline('wpe_compute', Shaders.WPE_SHADER);

    const paramsBuffer = this.createBuffer(
      new Uint32Array([n_embd, B, T]),
      GPUBufferUsage.UNIFORM
    );

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: positions.buffer } },
        { binding: 1, resource: { buffer: weights.buffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil((B * T * n_embd) / 64));
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    return { shape: outputShape, buffer: outputBuffer, dataType: 'float32' };
  }

  runAdd(a: Tensor, b: Tensor): Tensor {
    if (!this.device) throw new Error("WebGPU device not initialized");
    const outputShape = [...a.shape];
    const numElements = a.shape.reduce((p, c) => p * c, 1);
    const outputSizeBytes = numElements * 4;
    const outputBuffer = this.createEmptyBuffer(outputSizeBytes, GPUBufferUsage.STORAGE);

    const pipeline = this.getOrCreatePipeline('add_compute', Shaders.ADD_SHADER);

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: a.buffer } },
        { binding: 1, resource: { buffer: b.buffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(numElements / 64));
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    return { shape: outputShape, buffer: outputBuffer, dataType: 'float32' };
  }

  runLayerNorm(x: Tensor, gamma: Tensor, beta: Tensor, epsilon = 1e-5): Tensor {
    if (!this.device) throw new Error("WebGPU device not initialized");
    const outputShape = [...x.shape];
    const C = x.shape[x.shape.length - 1];
    const totalElements = x.shape.reduce((p, c) => p * c, 1);
    const numRows = totalElements / C;
    const outputSizeBytes = totalElements * 4;
    const outputBuffer = this.createEmptyBuffer(outputSizeBytes, GPUBufferUsage.STORAGE);

    const pipeline = this.getOrCreatePipeline('layernorm_compute', Shaders.LAYERNORM_SHADER);

    const paramsBuffer = this.device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([C]) as any);
    this.device.queue.writeBuffer(paramsBuffer, 4, new Float32Array([epsilon]) as any);

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: x.buffer } },
        { binding: 1, resource: { buffer: gamma.buffer } },
        { binding: 2, resource: { buffer: beta.buffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(numRows / 64));
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    return { shape: outputShape, buffer: outputBuffer, dataType: 'float32' };
  }

  runMatMul(x: Tensor, w: Tensor, bias: Tensor | null, M: number, N: number, K: number): Tensor {
    if (!this.device) throw new Error("WebGPU device not initialized");
    
    // Output shape logic
    let outputShape = [M, N];
    if (x.shape.length === 3) {
      outputShape = [x.shape[0], x.shape[1], N];
    }
    
    const outputSizeBytes = M * N * 4;
    const outputBuffer = this.createEmptyBuffer(outputSizeBytes, GPUBufferUsage.STORAGE);

    const pipeline = this.getOrCreatePipeline('matmul_compute', Shaders.MATMUL_SHADER);

    const hasBias = bias ? 1 : 0;
    const paramsBuffer = this.createBuffer(
      new Uint32Array([M, N, K, hasBias]),
      GPUBufferUsage.UNIFORM
    );

    // If no bias, we supply a dummy buffer to satisfy the binding
    const biasBuffer = bias ? bias.buffer : this.createBuffer(new Float32Array(N), GPUBufferUsage.STORAGE);

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: x.buffer } },
        { binding: 1, resource: { buffer: w.buffer } },
        { binding: 2, resource: { buffer: biasBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(N / 16), Math.ceil(M / 16));
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    return { shape: outputShape, buffer: outputBuffer, dataType: 'float32' };
  }

  runGELU(x: Tensor): Tensor {
    if (!this.device) throw new Error("WebGPU device not initialized");
    const outputShape = [...x.shape];
    const numElements = x.shape.reduce((p, c) => p * c, 1);
    const outputSizeBytes = numElements * 4;
    const outputBuffer = this.createEmptyBuffer(outputSizeBytes, GPUBufferUsage.STORAGE);

    const pipeline = this.getOrCreatePipeline('gelu_compute', Shaders.GELU_SHADER);

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: x.buffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(numElements / 64));
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    return { shape: outputShape, buffer: outputBuffer, dataType: 'float32' };
  }

  runCausalSelfAttention(q: Tensor, k: Tensor, v: Tensor, B: number, n_head: number, T: number, head_size: number): Tensor {
    if (!this.device) throw new Error("WebGPU device not initialized");
    const outputShape = [B, n_head, T, head_size];
    const outputSizeBytes = B * n_head * T * head_size * 4;
    const outputBuffer = this.createEmptyBuffer(outputSizeBytes, GPUBufferUsage.STORAGE);

    const pipeline = this.getOrCreatePipeline('attention_compute', Shaders.ATTENTION_SHADER);

    const paramsBuffer = this.createBuffer(
      new Uint32Array([B, n_head, T, head_size]),
      GPUBufferUsage.UNIFORM
    );

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: q.buffer } },
        { binding: 1, resource: { buffer: k.buffer } },
        { binding: 2, resource: { buffer: v.buffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    // Grid: columns (T) x rows (Batch * Heads)
    passEncoder.dispatchWorkgroups(Math.ceil(T / 16), Math.ceil((B * n_head) / 16));
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    return { shape: outputShape, buffer: outputBuffer, dataType: 'float32' };
  }

  runSoftmax(logits: Tensor, B: number, vocab_size: number): Tensor {
    if (!this.device) throw new Error("WebGPU device not initialized");
    const outputShape = [B, vocab_size];
    const outputSizeBytes = B * vocab_size * 4;
    const outputBuffer = this.createEmptyBuffer(outputSizeBytes, GPUBufferUsage.STORAGE);

    const pipeline = this.getOrCreatePipeline('softmax_compute', Shaders.SOFTMAX_SHADER);

    const paramsBuffer = this.createBuffer(
      new Uint32Array([B, vocab_size]),
      GPUBufferUsage.UNIFORM
    );

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: logits.buffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(B / 64));
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    return { shape: outputShape, buffer: outputBuffer, dataType: 'float32' };
  }
}
