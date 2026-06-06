/// <reference types="@webgpu/types" />
// Full WebGPU character-level Transformer trainer.
//
// This mirrors the math in cpuTransformer.ts (CharTransformer) EXACTLY, but
// runs the forward pass, the backward pass (real backprop) and the Adam update
// entirely on the GPU via WGSL compute kernels. Weights, gradients and Adam
// moments all stay resident in GPU buffers; only tiny scalars (loss, grad norm)
// are read back per step.
//
// Architecture (single head, matching the CPU reference):
//   x0 = wte[tok] + wpe[pos]
//   a  = x0 + Wo( attn( LN1(x0) ) )
//   y  = a  + W2( gelu( W1( LN2(a) ) ) )
//   logits = Whead( LNf(y) )
//
// Because WebGPU cannot be exercised in the build environment, GpuTransformer
// is designed to be checked against the CPU model at runtime (see App's parity
// self-check) and the app falls back to CPU automatically on any divergence.

import { CharTransformer, type TransformerConfig } from './cpuTransformer';

interface Buf {
  buffer: GPUBuffer;
  size: number; // element count (f32)
}

const WG = 64;

export class GpuTransformer {
  device: GPUDevice;
  cfg: TransformerConfig;
  V: number;
  C: number;
  T: number;
  H: number;

  private pipelines = new Map<string, GPUComputePipeline>();
  private params: { name: string; data: Buf; grad: Buf; m: Buf; v: Buf }[] = [];
  private p: Record<string, { data: Buf; grad: Buf; m: Buf; v: Buf }> = {};
  private inter: Record<string, Buf> = {};
  private seqBuf!: Buf;       // u32 token ids
  private tgtBuf!: Buf;       // u32 targets
  private lossBuf!: Buf;      // per-row loss [T]
  private accumLoss!: Buf;    // running loss sum across the batch [1]
  private normPartials!: Buf; // scratch for grad-norm reduction
  private clipScaleBuf!: Buf; // GPU-resident grad-clip scale [1]
  stepCount = 0;

  constructor(device: GPUDevice, cfg: TransformerConfig, vocab: number) {
    this.device = device;
    this.cfg = cfg;
    this.V = vocab;
    this.C = cfg.nEmbd;
    this.T = cfg.blockSize;
    this.H = cfg.nEmbd * cfg.mlpMult;
    this.allocate();
  }

  // ---- buffer helpers ----
  private mk(size: number, storage = true): Buf {
    const usage = (storage ? GPUBufferUsage.STORAGE : GPUBufferUsage.UNIFORM) |
      GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    return { buffer: this.device.createBuffer({ size: Math.max(4, size * 4), usage }), size };
  }
  private upload(b: Buf, arr: Float32Array | Uint32Array) {
    this.device.queue.writeBuffer(b.buffer, 0, arr as any);
  }
  async read(b: Buf, count = b.size): Promise<Float32Array> {
    const rb = this.device.createBuffer({ size: count * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(b.buffer, 0, rb, 0, count * 4);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap();
    rb.destroy();
    return out;
  }

  private param(name: string, size: number) {
    const rec = { data: this.mk(size), grad: this.mk(size), m: this.mk(size), v: this.mk(size) };
    this.device.queue.writeBuffer(rec.m.buffer, 0, new Float32Array(size));
    this.device.queue.writeBuffer(rec.v.buffer, 0, new Float32Array(size));
    this.p[name] = rec;
    this.params.push({ name, ...rec });
  }

  private allocate() {
    const { C, T, V, H } = this;
    this.param('wte', V * C); this.param('wpe', T * C);
    this.param('ln1g', C); this.param('ln1b', C);
    this.param('Wq', C * C); this.param('bq', C);
    this.param('Wk', C * C); this.param('bk', C);
    this.param('Wv', C * C); this.param('bv', C);
    this.param('Wo', C * C); this.param('bo', C);
    this.param('ln2g', C); this.param('ln2b', C);
    this.param('W1', C * H); this.param('b1', H);
    this.param('W2', H * C); this.param('b2', C);
    this.param('lnfg', C); this.param('lnfb', C);
    this.param('Whead', C * V); this.param('bhead', V);

    const I = (n: number) => this.mk(n);
    // forward intermediates (max size n=T)
    Object.assign(this.inter, {
      x0: I(T * C),
      ln1y: I(T * C), ln1xhat: I(T * C), ln1rstd: I(T),
      q: I(T * C), k: I(T * C), v: I(T * C),
      att: I(T * T), attOut: I(T * C), proj: I(T * C), x1: I(T * C),
      ln2y: I(T * C), ln2xhat: I(T * C), ln2rstd: I(T),
      fc: I(T * H), act: I(T * H), mlpOut: I(T * C), x2: I(T * C),
      lnfy: I(T * C), lnfxhat: I(T * C), lnfrstd: I(T),
      logits: I(T * V),
      // backward
      dlogits: I(T * V), dLnfY: I(T * C), dx2: I(T * C),
      dAct: I(T * H), dFc: I(T * H), dLn2Y: I(T * C), dx1: I(T * C),
      dAttOut: I(T * C), dq: I(T * C), dk: I(T * C), dv: I(T * C), dLn1Y: I(T * C),
      dx0: I(T * C),
      dScratchX: I(T * C),
      da: I(T * T), sa: I(T), ds: I(T * T),
      // scratch for linear backward (transpose + matmul temps)
      sA: I(Math.max(C, H) * Math.max(C, V, T)),
      sB: I(Math.max(C, H) * Math.max(C, V, T)),
    });
    this.seqBuf = this.mk(T);
    this.tgtBuf = this.mk(T);
    this.lossBuf = this.mk(T);
    this.accumLoss = this.mk(1);
    this.normPartials = I(4096);
    this.clipScaleBuf = this.mk(1);
  }

  // Copy weights from a CPU model (used to seed + for parity checks).
  loadFromCPU(m: CharTransformer) {
    const set = (name: string, d: Float32Array) => this.upload(this.p[name].data, d);
    set('wte', m.wte.data); set('wpe', m.wpe.data);
    set('ln1g', m.ln1g.data); set('ln1b', m.ln1b.data);
    set('Wq', m.Wq.data); set('bq', m.bq.data);
    set('Wk', m.Wk.data); set('bk', m.bk.data);
    set('Wv', m.Wv.data); set('bv', m.bv.data);
    set('Wo', m.Wo.data); set('bo', m.bo.data);
    set('ln2g', m.ln2g.data); set('ln2b', m.ln2b.data);
    set('W1', m.W1.data); set('b1', m.b1.data);
    set('W2', m.W2.data); set('b2', m.b2.data);
    set('lnfg', m.lnfg.data); set('lnfb', m.lnfb.data);
    set('Whead', m.Whead.data); set('bhead', m.bhead.data);
    this.stepCount = m.stepCount;
  }

  // ---- pipeline dispatch ----
  private pipe(name: string, code: string): GPUComputePipeline {
    let pl = this.pipelines.get(name);
    if (!pl) {
      const mod = this.device.createShaderModule({ code });
      pl = this.device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
      this.pipelines.set(name, pl);
    }
    return pl;
  }

  private run(enc: GPUCommandEncoder, name: string, code: string, bufs: Buf[], groups: number, uni?: Uint32Array | Float32Array) {
    const pl = this.pipe(name, code);
    const entries: GPUBindGroupEntry[] = bufs.map((b, i) => ({ binding: i, resource: { buffer: b.buffer } }));
    if (uni) {
      const ub = this.device.createBuffer({ size: Math.max(16, uni.byteLength), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(ub, 0, uni as any);
      entries.push({ binding: bufs.length, resource: { buffer: ub } });
    }
    const bg = this.device.createBindGroup({ layout: pl.getBindGroupLayout(0), entries });
    const pass = enc.beginComputePass();
    pass.setPipeline(pl);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(groups);
    pass.end();
  }

  private run2(enc: GPUCommandEncoder, name: string, code: string, bufs: Buf[], gx: number, gy: number, uni?: Uint32Array | Float32Array) {
    const pl = this.pipe(name, code);
    const entries: GPUBindGroupEntry[] = bufs.map((b, i) => ({ binding: i, resource: { buffer: b.buffer } }));
    if (uni) {
      const ub = this.device.createBuffer({ size: Math.max(16, uni.byteLength), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(ub, 0, uni as any);
      entries.push({ binding: bufs.length, resource: { buffer: ub } });
    }
    const bg = this.device.createBindGroup({ layout: pl.getBindGroupLayout(0), entries });
    const pass = enc.beginComputePass();
    pass.setPipeline(pl);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(gx, gy);
    pass.end();
  }

  // ===================== KERNELS =====================
  private static K = {
    zero: `@group(0)@binding(0) var<storage,read_write> a: array<f32>;
      struct P{n:u32}; @group(0)@binding(1) var<uniform> p:P;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g:vec3<u32>){ if(g.x<p.n){a[g.x]=0.0;} }`,

    embedFwd: `@group(0)@binding(0) var<storage,read> seq: array<u32>;
      @group(0)@binding(1) var<storage,read> wte: array<f32>;
      @group(0)@binding(2) var<storage,read> wpe: array<f32>;
      @group(0)@binding(3) var<storage,read_write> x0: array<f32>;
      struct P{n:u32,C:u32}; @group(0)@binding(4) var<uniform> p:P;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g:vec3<u32>){
        let idx=g.x; if(idx>=p.n*p.C){return;}
        let t=idx/p.C; let c=idx%p.C; let tok=seq[t];
        x0[idx]=wte[tok*p.C+c]+wpe[t*p.C+c];
      }`,

    lnFwd: `@group(0)@binding(0) var<storage,read> x: array<f32>;
      @group(0)@binding(1) var<storage,read> g: array<f32>;
      @group(0)@binding(2) var<storage,read> b: array<f32>;
      @group(0)@binding(3) var<storage,read_write> y: array<f32>;
      @group(0)@binding(4) var<storage,read_write> xhat: array<f32>;
      @group(0)@binding(5) var<storage,read_write> rstd: array<f32>;
      struct P{n:u32,C:u32}; @group(0)@binding(6) var<uniform> p:P;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) gid:vec3<u32>){
        let i=gid.x; if(i>=p.n){return;} let off=i*p.C;
        var mean=0.0; for(var c=0u;c<p.C;c++){mean+=x[off+c];} mean/=f32(p.C);
        var vs=0.0; for(var c=0u;c<p.C;c++){let d=x[off+c]-mean; vs+=d*d;}
        let rs=1.0/sqrt(vs/f32(p.C)+1e-5); rstd[i]=rs;
        for(var c=0u;c<p.C;c++){let xh=(x[off+c]-mean)*rs; xhat[off+c]=xh; y[off+c]=xh*g[c]+b[c];}
      }`,

    // Y[M,N] = X[M,K] @ W[K,N]
    matmul: `@group(0)@binding(0) var<storage,read> X: array<f32>;
      @group(0)@binding(1) var<storage,read> W: array<f32>;
      @group(0)@binding(2) var<storage,read_write> Y: array<f32>;
      struct P{M:u32,N:u32,K:u32}; @group(0)@binding(3) var<uniform> p:P;
      @compute @workgroup_size(16,16) fn main(@builtin(global_invocation_id) id:vec3<u32>){
        let row=id.y; let col=id.x; if(row>=p.M||col>=p.N){return;}
        var s=0.0; for(var i=0u;i<p.K;i++){ s+=X[row*p.K+i]*W[i*p.N+col]; }
        Y[row*p.N+col]=s;
      }`,

    biasAdd: `@group(0)@binding(0) var<storage,read_write> Y: array<f32>;
      @group(0)@binding(1) var<storage,read> b: array<f32>;
      struct P{M:u32,N:u32}; @group(0)@binding(2) var<uniform> p:P;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g:vec3<u32>){
        let idx=g.x; if(idx>=p.M*p.N){return;} Y[idx]+=b[idx%p.N];
      }`,

    transpose: `@group(0)@binding(0) var<storage,read> A: array<f32>;
      @group(0)@binding(1) var<storage,read_write> O: array<f32>;
      struct P{M:u32,N:u32}; @group(0)@binding(2) var<uniform> p:P;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g:vec3<u32>){
        let idx=g.x; if(idx>=p.M*p.N){return;} let i=idx/p.N; let j=idx%p.N; O[j*p.M+i]=A[idx];
      }`,

    addInto: `@group(0)@binding(0) var<storage,read_write> dst: array<f32>;
      @group(0)@binding(1) var<storage,read> src: array<f32>;
      struct P{n:u32}; @group(0)@binding(2) var<uniform> p:P;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g:vec3<u32>){ if(g.x<p.n){dst[g.x]+=src[g.x];} }`,

    colsumInto: `@group(0)@binding(0) var<storage,read_write> db: array<f32>;
      @group(0)@binding(1) var<storage,read> dy: array<f32>;
      struct P{M:u32,N:u32}; @group(0)@binding(2) var<uniform> p:P;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g:vec3<u32>){
        let j=g.x; if(j>=p.N){return;} var s=0.0; for(var i=0u;i<p.M;i++){s+=dy[i*p.N+j];} db[j]+=s;
      }`,

    residual: `@group(0)@binding(0) var<storage,read> a: array<f32>;
      @group(0)@binding(1) var<storage,read> b: array<f32>;
      @group(0)@binding(2) var<storage,read_write> o: array<f32>;
      struct P{n:u32}; @group(0)@binding(3) var<uniform> p:P;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g:vec3<u32>){ if(g.x<p.n){o[g.x]=a[g.x]+b[g.x];} }`,

    attnFwd: `@group(0)@binding(0) var<storage,read> q: array<f32>;
      @group(0)@binding(1) var<storage,read> k: array<f32>;
      @group(0)@binding(2) var<storage,read> v: array<f32>;
      @group(0)@binding(3) var<storage,read_write> att: array<f32>;
      @group(0)@binding(4) var<storage,read_write> attOut: array<f32>;
      struct P{n:u32,C:u32,scale:f32}; @group(0)@binding(5) var<uniform> p:P;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) gid:vec3<u32>){
        let i=gid.x; if(i>=p.n){return;}
        var maxs=-1e30;
        for(var j=0u;j<=i;j++){ var d=0.0; for(var c=0u;c<p.C;c++){d+=q[i*p.C+c]*k[j*p.C+c];} d*=p.scale; att[i*p.n+j]=d; if(d>maxs){maxs=d;} }
        var sum=0.0; for(var j=0u;j<=i;j++){ let e=exp(att[i*p.n+j]-maxs); att[i*p.n+j]=e; sum+=e; }
        for(var j=0u;j<=i;j++){ att[i*p.n+j]=att[i*p.n+j]/sum; }
        for(var c=0u;c<p.C;c++){ var s=0.0; for(var j=0u;j<=i;j++){ s+=att[i*p.n+j]*v[j*p.C+c]; } attOut[i*p.C+c]=s; }
      }`,

    geluFwd: `@group(0)@binding(0) var<storage,read> x: array<f32>;
      @group(0)@binding(1) var<storage,read_write> o: array<f32>;
      struct P{n:u32}; @group(0)@binding(2) var<uniform> p:P;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g:vec3<u32>){
        if(g.x>=p.n){return;} let val=x[g.x]; let inner=0.7978845608028654*(val+0.044715*val*val*val);
        o[g.x]=0.5*val*(1.0+tanh(inner));
      }`,

    softmaxCE: `@group(0)@binding(0) var<storage,read> logits: array<f32>;
      @group(0)@binding(1) var<storage,read> tgt: array<u32>;
      @group(0)@binding(2) var<storage,read_write> dlogits: array<f32>;
      @group(0)@binding(3) var<storage,read_write> loss: array<f32>;
      struct P{n:u32,V:u32,invCount:f32}; @group(0)@binding(4) var<uniform> p:P;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) gid:vec3<u32>){
        let i=gid.x; if(i>=p.n){return;} let off=i*p.V; let t=tgt[i];
        var maxl=-1e30; for(var j=0u;j<p.V;j++){ if(logits[off+j]>maxl){maxl=logits[off+j];} }
        var sum=0.0; for(var j=0u;j<p.V;j++){ sum+=exp(logits[off+j]-maxl); }
        let pt=exp(logits[off+t]-maxl)/sum; loss[i]=-log(max(pt,1e-12));
        for(var j=0u;j<p.V;j++){ var pj=exp(logits[off+j]-maxl)/sum; dlogits[off+j]=pj*p.invCount; }
        dlogits[off+t]-=p.invCount;
      }`,

    geluBwd: `@group(0)@binding(0) var<storage,read> dact: array<f32>;
      @group(0)@binding(1) var<storage,read> fc: array<f32>;
      @group(0)@binding(2) var<storage,read_write> dfc: array<f32>;
      struct P{n:u32}; @group(0)@binding(3) var<uniform> p:P;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g:vec3<u32>){
        if(g.x>=p.n){return;} let x=fc[g.x]; let x3=x*x*x;
        let inner=0.7978845608028654*(x+0.044715*x3); let tnh=tanh(inner);
        let dInner=0.7978845608028654*(1.0+3.0*0.044715*x*x);
        let gg=0.5*(1.0+tnh)+0.5*x*(1.0-tnh*tnh)*dInner; dfc[g.x]=dact[g.x]*gg;
      }`,

    // LayerNorm backward: dx (per row). Accumulate dg,db separately.
    lnBwdDx: `@group(0)@binding(0) var<storage,read> dy: array<f32>;
      @group(0)@binding(1) var<storage,read> xhat: array<f32>;
      @group(0)@binding(2) var<storage,read> rstd: array<f32>;
      @group(0)@binding(3) var<storage,read> g: array<f32>;
      @group(0)@binding(4) var<storage,read_write> dx: array<f32>;
      struct P{n:u32,C:u32}; @group(0)@binding(5) var<uniform> p:P;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) gid:vec3<u32>){
        let i=gid.x; if(i>=p.n){return;} let off=i*p.C;
        var mdx=0.0; var mdxx=0.0;
        for(var c=0u;c<p.C;c++){ let dh=dy[off+c]*g[c]; mdx+=dh; mdxx+=dh*xhat[off+c]; }
        mdx/=f32(p.C); mdxx/=f32(p.C); let rs=rstd[i];
        for(var c=0u;c<p.C;c++){ let dh=dy[off+c]*g[c]; dx[off+c]=rs*(dh-mdx-xhat[off+c]*mdxx); }
      }`,

    lnBwdParam: `@group(0)@binding(0) var<storage,read> dy: array<f32>;
      @group(0)@binding(1) var<storage,read> xhat: array<f32>;
      @group(0)@binding(2) var<storage,read_write> dg: array<f32>;
      @group(0)@binding(3) var<storage,read_write> db: array<f32>;
      struct P{n:u32,C:u32}; @group(0)@binding(4) var<uniform> p:P;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) gid:vec3<u32>){
        let c=gid.x; if(c>=p.C){return;} var sg=0.0; var sb=0.0;
        for(var i=0u;i<p.n;i++){ let d=dy[i*p.C+c]; sg+=d*xhat[i*p.C+c]; sb+=d; }
        dg[c]+=sg; db[c]+=sb;
      }`,

    // Attention backward pieces.
    attnDa: `@group(0)@binding(0) var<storage,read> dAttOut: array<f32>;
      @group(0)@binding(1) var<storage,read> v: array<f32>;
      @group(0)@binding(2) var<storage,read_write> da: array<f32>;
      struct P{n:u32,C:u32}; @group(0)@binding(3) var<uniform> p:P;
      @compute @workgroup_size(16,16) fn main(@builtin(global_invocation_id) id:vec3<u32>){
        let i=id.y; let j=id.x; if(i>=p.n||j>=p.n||j>i){return;}
        var s=0.0; for(var c=0u;c<p.C;c++){ s+=dAttOut[i*p.C+c]*v[j*p.C+c]; } da[i*p.n+j]=s;
      }`,

    attnSa: `@group(0)@binding(0) var<storage,read> att: array<f32>;
      @group(0)@binding(1) var<storage,read> da: array<f32>;
      @group(0)@binding(2) var<storage,read_write> sa: array<f32>;
      struct P{n:u32}; @group(0)@binding(3) var<uniform> p:P;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) gid:vec3<u32>){
        let i=gid.x; if(i>=p.n){return;} var s=0.0; for(var j=0u;j<=i;j++){ s+=att[i*p.n+j]*da[i*p.n+j]; } sa[i]=s;
      }`,

    attnDs: `@group(0)@binding(0) var<storage,read> att: array<f32>;
      @group(0)@binding(1) var<storage,read> da: array<f32>;
      @group(0)@binding(2) var<storage,read> sa: array<f32>;
      @group(0)@binding(3) var<storage,read_write> ds: array<f32>;
      struct P{n:u32,scale:f32}; @group(0)@binding(4) var<uniform> p:P;
      @compute @workgroup_size(16,16) fn main(@builtin(global_invocation_id) id:vec3<u32>){
        let i=id.y; let j=id.x; if(i>=p.n||j>=p.n||j>i){return;}
        ds[i*p.n+j]=att[i*p.n+j]*(da[i*p.n+j]-sa[i])*p.scale;
      }`,

    attnDq: `@group(0)@binding(0) var<storage,read> ds: array<f32>;
      @group(0)@binding(1) var<storage,read> k: array<f32>;
      @group(0)@binding(2) var<storage,read_write> dq: array<f32>;
      struct P{n:u32,C:u32}; @group(0)@binding(3) var<uniform> p:P;
      @compute @workgroup_size(16,16) fn main(@builtin(global_invocation_id) id:vec3<u32>){
        let i=id.y; let c=id.x; if(i>=p.n||c>=p.C){return;}
        var s=0.0; for(var j=0u;j<=i;j++){ s+=ds[i*p.n+j]*k[j*p.C+c]; } dq[i*p.C+c]=s;
      }`,

    attnDk: `@group(0)@binding(0) var<storage,read> ds: array<f32>;
      @group(0)@binding(1) var<storage,read> q: array<f32>;
      @group(0)@binding(2) var<storage,read_write> dk: array<f32>;
      struct P{n:u32,C:u32}; @group(0)@binding(3) var<uniform> p:P;
      @compute @workgroup_size(16,16) fn main(@builtin(global_invocation_id) id:vec3<u32>){
        let j=id.y; let c=id.x; if(j>=p.n||c>=p.C){return;}
        var s=0.0; for(var i=j;i<p.n;i++){ s+=ds[i*p.n+j]*q[i*p.C+c]; } dk[j*p.C+c]=s;
      }`,

    attnDv: `@group(0)@binding(0) var<storage,read> att: array<f32>;
      @group(0)@binding(1) var<storage,read> dAttOut: array<f32>;
      @group(0)@binding(2) var<storage,read_write> dv: array<f32>;
      struct P{n:u32,C:u32}; @group(0)@binding(3) var<uniform> p:P;
      @compute @workgroup_size(16,16) fn main(@builtin(global_invocation_id) id:vec3<u32>){
        let j=id.y; let c=id.x; if(j>=p.n||c>=p.C){return;}
        var s=0.0; for(var i=j;i<p.n;i++){ s+=att[i*p.n+j]*dAttOut[i*p.C+c]; } dv[j*p.C+c]=s;
      }`,

    embedBwd: `@group(0)@binding(0) var<storage,read> seq: array<u32>;
      @group(0)@binding(1) var<storage,read> dx0: array<f32>;
      @group(0)@binding(2) var<storage,read_write> dwte: array<f32>;
      @group(0)@binding(3) var<storage,read_write> dwpe: array<f32>;
      struct P{n:u32,C:u32}; @group(0)@binding(4) var<uniform> p:P;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) gid:vec3<u32>){
        let c=gid.x; if(c>=p.C){return;}
        for(var t=0u;t<p.n;t++){ let g=dx0[t*p.C+c]; dwte[seq[t]*p.C+c]+=g; dwpe[t*p.C+c]+=g; }
      }`,

    scale: `@group(0)@binding(0) var<storage,read_write> a: array<f32>;
      struct P{n:u32,s:f32}; @group(0)@binding(1) var<uniform> p:P;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g:vec3<u32>){ if(g.x<p.n){a[g.x]*=p.s;} }`,

    sumsq: `@group(0)@binding(0) var<storage,read> a: array<f32>;
      @group(0)@binding(1) var<storage,read_write> partial: array<f32>;
      struct P{n:u32,off:u32}; @group(0)@binding(2) var<uniform> p:P;
      var<workgroup> tmp: array<f32,${WG}>;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) gid:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>, @builtin(workgroup_id) wid:vec3<u32>){
        var s=0.0; if(gid.x<p.n){ s=a[gid.x]*a[gid.x]; } tmp[lid.x]=s; workgroupBarrier();
        var stride=${WG / 2}u; loop{ if(stride==0u){break;} if(lid.x<stride){tmp[lid.x]+=tmp[lid.x+stride];} workgroupBarrier(); stride=stride/2u; }
        if(lid.x==0u){ partial[p.off+wid.x]=tmp[0]; }
      }`,

    adam: `@group(0)@binding(0) var<storage,read_write> data: array<f32>;
      @group(0)@binding(1) var<storage,read> grad: array<f32>;
      @group(0)@binding(2) var<storage,read_write> m: array<f32>;
      @group(0)@binding(3) var<storage,read_write> v: array<f32>;
      struct P{n:u32,lr:f32,bc1:f32,bc2:f32}; @group(0)@binding(4) var<uniform> p:P;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g:vec3<u32>){
        let i=g.x; if(i>=p.n){return;} let gi=grad[i];
        m[i]=0.9*m[i]+0.1*gi; v[i]=0.999*v[i]+0.001*gi*gi;
        let mh=m[i]/p.bc1; let vh=v[i]/p.bc2; data[i]-=p.lr*mh/(sqrt(vh)+1e-8);
      }`,

    accumLoss: `@group(0)@binding(0) var<storage,read> loss: array<f32>;
      @group(0)@binding(1) var<storage,read_write> accum: array<f32>;
      struct P{n:u32}; @group(0)@binding(2) var<uniform> p:P;
      @compute @workgroup_size(1) fn main(){ var s=0.0; for(var i=0u;i<p.n;i++){ s+=loss[i]; } accum[0]+=s; }`,

    // Reduce all partial sum-of-squares into a single clip scale = min(1, 1/sqrt(sum)).
    clipScale: `@group(0)@binding(0) var<storage,read> partial: array<f32>;
      @group(0)@binding(1) var<storage,read_write> outScale: array<f32>;
      struct P{n:u32}; @group(0)@binding(2) var<uniform> p:P;
      @compute @workgroup_size(1) fn main(){ var s=0.0; for(var i=0u;i<p.n;i++){ s+=partial[i]; } let nrm=sqrt(s); outScale[0]=select(1.0, 1.0/nrm, nrm>1.0); }`,

    // Scale a buffer in place by a scalar stored in scale[0] (GPU-resident, no readback).
    scaleByBuf: `@group(0)@binding(0) var<storage,read_write> a: array<f32>;
      @group(0)@binding(1) var<storage,read> scale: array<f32>;
      struct P{n:u32}; @group(0)@binding(2) var<uniform> p:P;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g:vec3<u32>){ if(g.x<p.n){ a[g.x]*=scale[0]; } }`,
  };

  // Pack a uniform with mixed u32/f32 fields (WGSL reads them by declared type).
  private pk(spec: Array<['u' | 'f', number]>): Uint32Array {
    const buf = new ArrayBuffer(spec.length * 4);
    const dv = new DataView(buf);
    spec.forEach(([t, v], i) => {
      if (t === 'u') dv.setUint32(i * 4, v >>> 0, true);
      else dv.setFloat32(i * 4, v, true);
    });
    return new Uint32Array(buf);
  }

  // ---- linear forward: Y[n,o] = X[n,k] @ W[k,o] + b ----
  private linFwd(enc: GPUCommandEncoder, X: Buf, n: number, k: number, W: Buf, b: Buf, o: number, Y: Buf) {
    const K = GpuTransformer.K;
    this.run2(enc, 'matmul', K.matmul, [X, W, Y], Math.ceil(o / 16), Math.ceil(n / 16), new Uint32Array([n, o, k]));
    this.run(enc, 'biasAdd', K.biasAdd, [Y, b], Math.ceil((n * o) / WG), new Uint32Array([n, o]));
  }

  // ---- linear backward: returns dX into provided buffer; accumulates dW,db ----
  private linBwd(enc: GPUCommandEncoder, dY: Buf, X: Buf, n: number, k: number, o: number, W: Buf, dW: Buf, db: Buf, dX: Buf) {
    const K = GpuTransformer.K;
    const { sA, sB } = this.inter;
    // dX = dY[n,o] @ W^T[o,k]
    this.run(enc, 'transpose', K.transpose, [W, sA], Math.ceil((k * o) / WG), new Uint32Array([k, o])); // sA = Wt[o,k]
    this.run2(enc, 'matmul', K.matmul, [dY, sA, dX], Math.ceil(k / 16), Math.ceil(n / 16), new Uint32Array([n, k, o]));
    // dW += X^T[k,n] @ dY[n,o]
    this.run(enc, 'transpose', K.transpose, [X, sB], Math.ceil((n * k) / WG), new Uint32Array([n, k])); // sB = Xt[k,n]
    this.run2(enc, 'matmul', K.matmul, [sB, dY, sA], Math.ceil(o / 16), Math.ceil(k / 16), new Uint32Array([k, o, n])); // sA = dWtmp[k,o]
    this.run(enc, 'addInto', K.addInto, [dW, sA], Math.ceil((k * o) / WG), new Uint32Array([k * o]));
    // db += colsum(dY)
    this.run(enc, 'colsumInto', K.colsumInto, [db, dY], Math.ceil(o / WG), new Uint32Array([n, o]));
  }

  private zero(enc: GPUCommandEncoder, b: Buf, n: number) {
    this.run(enc, 'zero', GpuTransformer.K.zero, [b], Math.ceil(n / WG), new Uint32Array([n]));
  }

  // Build the forward graph into the encoder for a sequence of length n.
  private encodeForward(enc: GPUCommandEncoder, n: number) {
    const K = GpuTransformer.K;
    const { C, H, V } = this;
    const I = this.inter;
    const scale = 1 / Math.sqrt(C);
    // embeddings
    this.run(enc, 'embedFwd', K.embedFwd, [this.seqBuf, this.p.wte.data, this.p.wpe.data, I.x0], Math.ceil((n * C) / WG), new Uint32Array([n, C]));
    // ln1
    this.run(enc, 'lnFwd', K.lnFwd, [I.x0, this.p.ln1g.data, this.p.ln1b.data, I.ln1y, I.ln1xhat, I.ln1rstd], Math.ceil(n / WG), new Uint32Array([n, C]));
    // qkv
    this.linFwd(enc, I.ln1y, n, C, this.p.Wq.data, this.p.bq.data, C, I.q);
    this.linFwd(enc, I.ln1y, n, C, this.p.Wk.data, this.p.bk.data, C, I.k);
    this.linFwd(enc, I.ln1y, n, C, this.p.Wv.data, this.p.bv.data, C, I.v);
    // attention
    this.zero(enc, I.att, n * n);
    this.run(enc, 'attnFwd', K.attnFwd, [I.q, I.k, I.v, I.att, I.attOut], Math.ceil(n / WG), this.pk([['u', n], ['u', C], ['f', scale]]));
    // output proj + residual1
    this.linFwd(enc, I.attOut, n, C, this.p.Wo.data, this.p.bo.data, C, I.proj);
    this.run(enc, 'residual', K.residual, [I.x0, I.proj, I.x1], Math.ceil((n * C) / WG), new Uint32Array([n * C]));
    // ln2 + mlp
    this.run(enc, 'lnFwd', K.lnFwd, [I.x1, this.p.ln2g.data, this.p.ln2b.data, I.ln2y, I.ln2xhat, I.ln2rstd], Math.ceil(n / WG), new Uint32Array([n, C]));
    this.linFwd(enc, I.ln2y, n, C, this.p.W1.data, this.p.b1.data, H, I.fc);
    this.run(enc, 'geluFwd', K.geluFwd, [I.fc, I.act], Math.ceil((n * H) / WG), new Uint32Array([n * H]));
    this.linFwd(enc, I.act, n, H, this.p.W2.data, this.p.b2.data, C, I.mlpOut);
    this.run(enc, 'residual', K.residual, [I.x1, I.mlpOut, I.x2], Math.ceil((n * C) / WG), new Uint32Array([n * C]));
    // final ln + head
    this.run(enc, 'lnFwd', K.lnFwd, [I.x2, this.p.lnfg.data, this.p.lnfb.data, I.lnfy, I.lnfxhat, I.lnfrstd], Math.ceil(n / WG), new Uint32Array([n, C]));
    this.linFwd(enc, I.lnfy, n, C, this.p.Whead.data, this.p.bhead.data, V, I.logits);
  }

  private encodeBackward(enc: GPUCommandEncoder, n: number, invCount: number) {
    const K = GpuTransformer.K;
    const { C, H, V } = this;
    const I = this.inter;
    const scale = 1 / Math.sqrt(C);
    // softmax + CE -> dlogits, loss
    this.run(enc, 'softmaxCE', K.softmaxCE, [I.logits, this.tgtBuf, I.dlogits, this.lossBuf], Math.ceil(n / WG), this.pk([['u', n], ['u', V], ['f', invCount]]));
    // head backward -> dLnfY ; dWhead,dbhead
    this.linBwd(enc, I.dlogits, I.lnfy, n, C, V, this.p.Whead.data, this.p.Whead.grad, this.p.bhead.grad, I.dLnfY);
    // final ln backward -> dx2
    this.run(enc, 'lnBwdDx', K.lnBwdDx, [I.dLnfY, I.lnfxhat, I.lnfrstd, this.p.lnfg.data, I.dx2], Math.ceil(n / WG), new Uint32Array([n, C]));
    this.run(enc, 'lnBwdParam', K.lnBwdParam, [I.dLnfY, I.lnfxhat, this.p.lnfg.grad, this.p.lnfb.grad], Math.ceil(C / WG), new Uint32Array([n, C]));
    // residual2: dmlpOut = dx2 ; dx1 += dx2 (start dx1 = dx2)
    // mlp backward
    this.linBwd(enc, I.dx2, I.act, n, H, C, this.p.W2.data, this.p.W2.grad, this.p.b2.grad, I.dAct);
    this.run(enc, 'geluBwd', K.geluBwd, [I.dAct, I.fc, I.dFc], Math.ceil((n * H) / WG), new Uint32Array([n * H]));
    this.linBwd(enc, I.dFc, I.ln2y, n, C, H, this.p.W1.data, this.p.W1.grad, this.p.b1.grad, I.dLn2Y);
    // ln2 backward -> dx1 contribution
    this.run(enc, 'lnBwdDx', K.lnBwdDx, [I.dLn2Y, I.ln2xhat, I.ln2rstd, this.p.ln2g.data, I.dx1], Math.ceil(n / WG), new Uint32Array([n, C]));
    this.run(enc, 'lnBwdParam', K.lnBwdParam, [I.dLn2Y, I.ln2xhat, this.p.ln2g.grad, this.p.ln2b.grad], Math.ceil(C / WG), new Uint32Array([n, C]));
    // dx1 += dx2 (residual)
    this.run(enc, 'addInto', K.addInto, [I.dx1, I.dx2], Math.ceil((n * C) / WG), new Uint32Array([n * C]));
    // residual1: dproj = dx1 ; output proj backward -> dAttOut
    this.linBwd(enc, I.dx1, I.attOut, n, C, C, this.p.Wo.data, this.p.Wo.grad, this.p.bo.grad, I.dAttOut);
    // attention backward
    this.zero(enc, I.da, n * n);
    this.zero(enc, I.ds, n * n);
    this.run2(enc, 'attnDa', K.attnDa, [I.dAttOut, I.v, I.da], Math.ceil(n / 16), Math.ceil(n / 16), new Uint32Array([n, C]));
    this.run(enc, 'attnSa', K.attnSa, [I.att, I.da, I.sa], Math.ceil(n / WG), new Uint32Array([n]));
    this.run2(enc, 'attnDs', K.attnDs, [I.att, I.da, I.sa, I.ds], Math.ceil(n / 16), Math.ceil(n / 16), this.pk([['u', n], ['f', scale]]));
    this.run2(enc, 'attnDq', K.attnDq, [I.ds, I.k, I.dq], Math.ceil(C / 16), Math.ceil(n / 16), new Uint32Array([n, C]));
    this.run2(enc, 'attnDk', K.attnDk, [I.ds, I.q, I.dk], Math.ceil(C / 16), Math.ceil(n / 16), new Uint32Array([n, C]));
    this.run2(enc, 'attnDv', K.attnDv, [I.att, I.dAttOut, I.dv], Math.ceil(C / 16), Math.ceil(n / 16), new Uint32Array([n, C]));
    // qkv backward -> dLn1Y (accumulate)
    this.zero(enc, I.dLn1Y, n * C);
    this.linBwd(enc, I.dq, I.ln1y, n, C, C, this.p.Wq.data, this.p.Wq.grad, this.p.bq.grad, I.dScratchX);
    this.run(enc, 'addInto', K.addInto, [I.dLn1Y, I.dScratchX], Math.ceil((n * C) / WG), new Uint32Array([n * C]));
    this.linBwd(enc, I.dk, I.ln1y, n, C, C, this.p.Wk.data, this.p.Wk.grad, this.p.bk.grad, I.dScratchX);
    this.run(enc, 'addInto', K.addInto, [I.dLn1Y, I.dScratchX], Math.ceil((n * C) / WG), new Uint32Array([n * C]));
    this.linBwd(enc, I.dv, I.ln1y, n, C, C, this.p.Wv.data, this.p.Wv.grad, this.p.bv.grad, I.dScratchX);
    this.run(enc, 'addInto', K.addInto, [I.dLn1Y, I.dScratchX], Math.ceil((n * C) / WG), new Uint32Array([n * C]));
    // ln1 backward -> dx0 ; dx0 += dx1 (residual)
    this.run(enc, 'lnBwdDx', K.lnBwdDx, [I.dLn1Y, I.ln1xhat, I.ln1rstd, this.p.ln1g.data, I.dx0], Math.ceil(n / WG), new Uint32Array([n, C]));
    this.run(enc, 'lnBwdParam', K.lnBwdParam, [I.dLn1Y, I.ln1xhat, this.p.ln1g.grad, this.p.ln1b.grad], Math.ceil(C / WG), new Uint32Array([n, C]));
    this.run(enc, 'addInto', K.addInto, [I.dx0, I.dx1], Math.ceil((n * C) / WG), new Uint32Array([n * C]));
    // embeddings backward
    this.run(enc, 'embedBwd', K.embedBwd, [this.seqBuf, I.dx0, this.p.wte.grad, this.p.wpe.grad], Math.ceil(C / WG), new Uint32Array([n, C]));
  }

  // ---- one Adam mini-batch step over random windows; returns avg loss ----
  // When needLoss is false, the per-step loss read-back is skipped entirely so
  // the GPU pipelines all sequences without stalling (much faster); the return
  // value is then -1 and callers should only read it on logging steps.
  async trainStep(tokens: number[], batch = 16, needLoss = true): Promise<number> {
    const T = this.T;
    const N = tokens.length;
    if (N < 2) return 0;

    let totalCount = 0;
    let valid = 0;
    const seqs: number[][] = [];
    for (let b = 0; b < batch; b++) {
      const start = Math.floor(Math.random() * Math.max(1, N - 1));
      const end = Math.min(start + T + 1, N);
      const w = tokens.slice(start, end);
      if (w.length < 2) continue;
      seqs.push(w);
    }

    // zero grads + loss accumulator (one submit)
    {
      const enc = this.device.createCommandEncoder();
      for (const pr of this.params) this.zero(enc, pr.grad, pr.grad.size);
      this.zero(enc, this.accumLoss, 1);
      this.device.queue.submit([enc.finish()]);
    }

    // forward + backward for every sequence WITHOUT reading anything back, so
    // the GPU keeps the pipeline full instead of stalling on mapAsync each time.
    for (const w of seqs) {
      const inputs = w.slice(0, w.length - 1);
      const targets = w.slice(1);
      const n = inputs.length;
      this.upload(this.seqBuf, new Uint32Array(inputs));
      this.upload(this.tgtBuf, new Uint32Array(targets));

      const enc = this.device.createCommandEncoder();
      this.encodeForward(enc, n);
      this.encodeBackward(enc, n, 1 / n);
      // accumulate this sequence's loss on the GPU (no CPU read)
      this.run(enc, 'accumLoss', GpuTransformer.K.accumLoss, [this.lossBuf, this.accumLoss], 1, new Uint32Array([n]));
      this.device.queue.submit([enc.finish()]);

      totalCount += n;
      valid++;
    }
    if (valid === 0) return 0;

    // average grads across batch
    {
      const enc = this.device.createCommandEncoder();
      for (const pr of this.params) this.run(enc, 'scale', GpuTransformer.K.scale, [pr.grad], Math.ceil(pr.grad.size / WG), this.pk([['u', pr.grad.size], ['f', 1 / valid]]));
      this.device.queue.submit([enc.finish()]);
    }

    // Global grad-norm clip to 1.0 — computed and applied entirely on the GPU
    // (no per-step read-back stall). partial sum-of-squares -> scale -> apply.
    {
      const enc = this.device.createCommandEncoder();
      let off = 0;
      for (const pr of this.params) {
        const groups = Math.ceil(pr.grad.size / WG);
        this.run(enc, 'sumsq', GpuTransformer.K.sumsq, [pr.grad, this.normPartials], groups, new Uint32Array([pr.grad.size, off]));
        off += groups;
      }
      this.run(enc, 'clipScale', GpuTransformer.K.clipScale, [this.normPartials, this.clipScaleBuf], 1, new Uint32Array([off]));
      for (const pr of this.params) this.run(enc, 'scaleByBuf', GpuTransformer.K.scaleByBuf, [pr.grad, this.clipScaleBuf], Math.ceil(pr.grad.size / WG), new Uint32Array([pr.grad.size]));
      this.device.queue.submit([enc.finish()]);
    }

    // Adam
    this.stepCount++;
    const bc1 = 1 - Math.pow(0.9, this.stepCount);
    const bc2 = 1 - Math.pow(0.999, this.stepCount);
    {
      const enc = this.device.createCommandEncoder();
      for (const pr of this.params) {
        this.run(enc, 'adam', GpuTransformer.K.adam, [pr.data, pr.grad, pr.m, pr.v], Math.ceil(pr.data.size / WG),
          this.pk([['u', pr.data.size], ['f', this.cfg.lr], ['f', bc1], ['f', bc2]]));
      }
      this.device.queue.submit([enc.finish()]);
    }

    if (!needLoss || totalCount === 0) return -1;
    const accum = await this.read(this.accumLoss, 1);
    return accum[0] / totalCount;
  }

  // ---- inference: returns logits[n*V] for a context ----
  async logitsForContext(ctx: number[]): Promise<Float32Array> {
    const n = ctx.length;
    this.upload(this.seqBuf, new Uint32Array(ctx));
    const enc = this.device.createCommandEncoder();
    this.encodeForward(enc, n);
    this.device.queue.submit([enc.finish()]);
    return this.read(this.inter.logits, n * this.V);
  }

  // Debug hook mirroring CharTransformer.debugForwardBackward: forward + backward
  // on one fixed sequence (no Adam), returning loss, logits and the head grad so
  // the app can numerically compare GPU vs CPU and fall back on divergence.
  async debugForwardBackward(inputs: number[], targets: number[]): Promise<{ loss: number; logits: Float32Array; wheadGrad: Float32Array }> {
    const n = inputs.length;
    {
      const enc = this.device.createCommandEncoder();
      for (const pr of this.params) this.zero(enc, pr.grad, pr.grad.size);
      this.device.queue.submit([enc.finish()]);
    }
    this.upload(this.seqBuf, new Uint32Array(inputs));
    this.upload(this.tgtBuf, new Uint32Array(targets));
    const enc = this.device.createCommandEncoder();
    this.encodeForward(enc, n);
    this.encodeBackward(enc, n, 1 / n);
    this.device.queue.submit([enc.finish()]);
    const lossRow = await this.read(this.lossBuf, n);
    let loss = 0;
    for (let i = 0; i < n; i++) loss += lossRow[i];
    const logits = await this.read(this.inter.logits, n * this.V);
    const wheadGrad = await this.read(this.p.Whead.grad, this.C * this.V);
    return { loss: loss / n, logits, wheadGrad };
  }
}
