// pcm-downsampler.js
class PCMDownsampler extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions || {};
    this.targetRate = opts.targetSampleRate || 16000;
    this.chunkMs = opts.chunkMs || 50;
    this.inputRate = sampleRate; // context rate, e.g., 48000
    this.buffer = [];
    this.samplesPerChunk = Math.floor(this.targetRate * (this.chunkMs / 1000));
    this.port.onmessage = (ev) => {
      if (ev.data && ev.data.type === 'stop') {
        this.buffer = [];
      }
    };
  }

  resample(inFloat32) {
    const ratio = this.inputRate / this.targetRate;
    const outLen = Math.floor(inFloat32.length / ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const x = i * ratio;
      const x0 = Math.floor(x);
      const x1 = Math.min(x0 + 1, inFloat32.length - 1);
      const t = x - x0;
      out[i] = inFloat32[x0] * (1 - t) + inFloat32[x1] * t;
    }
    return out;
  }

  floatToInt16PCM(float32) {
    const out = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      let s = Math.max(-1, Math.min(1, float32[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return out;
  }

  process(inputs, outputs, parameters) {
    const input = inputs;
    if (!input || input.length === 0) return true;
    const ch0 = input;
    if (!ch0) return true;

    const resampled = this.resample(ch0);
    this.buffer.push(resampled);

    let total = 0;
    for (const b of this.buffer) total += b.length;

    if (total >= this.samplesPerChunk) {
      const mono = new Float32Array(total);
      let o = 0;
      for (const b of this.buffer) { mono.set(b, o); o += b.length; }
      this.buffer = [];

      let start = 0;
      while (start + this.samplesPerChunk <= mono.length) {
        const slice = mono.subarray(start, start + this.samplesPerChunk);
        start += this.samplesPerChunk;
        const int16 = this.floatToInt16PCM(slice);
        this.port.postMessage(int16.buffer, [int16.buffer]);
      }
      if (start < mono.length) {
        this.buffer.push(mono.subarray(start));
      }
    }
    return true;
  }
}

registerProcessor('pcm-downsampler', PCMDownsampler);
