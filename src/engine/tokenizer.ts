// Byte-level Tokenizer for interactive WebGPU LLM Builder

export class ByteTokenizer {
  // Encode string to token IDs (0 - 255 for raw bytes)
  encode(text: string): number[] {
    const encoder = new TextEncoder();
    const bytes = encoder.encode(text);
    return Array.from(bytes);
  }

  // Decode token IDs back to string
  decode(tokens: number[]): string {
    const decoder = new TextDecoder();
    const bytes = new Uint8Array(tokens);
    return decoder.decode(bytes);
  }
}
