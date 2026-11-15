// Free-model aggregator — make.fun free-model-aggregator-01.
//
// One long-running Node program. It listens on 0.0.0.0 at a port it picks,
// registers with the trial env, and answers the env client's twenty
// requests, each with exactly one call to one of the ten free providers.
//
// How it picks a model, per request:
//   * only free models that meet the request's requirements (tools, vision,
//     json, minimum context) are candidates;
//   * a JSON-mode request from an OpenAI client may only go to an
//     OpenAI-wire or Gemini-wire provider, because an Anthropic-wire
//     provider echoes json_mode false;
//   * candidates are tried best-rated first; a provider that refuses
//     with a scheduled failure is retried twice more on the spot —
//     refusals are usually one scheduled failure, not exhaustion —
//     while a refusal that says the budget is spent or an outage has
//     begun retires that provider for the rest of the trial, so no
//     later request wastes calls on it; then the next candidate is
//     tried, until one answers 2xx.
//   * the provider is always called non-streaming; the answer text, usage
//     and stop reason are then rendered back in the wire the client used,
//     streamed or not as the client asked. That keeps every stream the
//     client sees well-formed and its usage exactly the provider's.
//
// Nothing here depends on the wall clock or on unseeded randomness: every
// choice is a function of the request and the providers' answers so far,
// so a replay leaves the same state. The provider directory, the quality
// table and the catalog below are the competition's sealed tables — the
// same on every trial; only each trial's budgets, failure schedules,
// outages and request bodies vary.

import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { readFile } from "node:fs/promises";

const FIXED_CREATED = 1767225600;
const PROTOCOL_VERSION = 1;
const UPSTREAM_TIMEOUT_MS = 10_000;
const WORK_POLL_MS = 100;
const MAX_ATTEMPTS_PER_PROVIDER = 3;

type Wire = "openai" | "anthropic" | "gemini";
type Auth = "bearer" | "x-api-key" | "x-goog-api-key" | "none";

interface Provider {
  id: string;
  wire: Wire;
  auth: Auth;
  keyEnv: string;
  key: string;
  path: string;
}

interface ModelRow {
  id: string;
  provider: string;
  rating: number;
  context: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

// The provider directory, as the brief publishes it. base_url in the
// catalog is the provider's own path (never joined to $MAKEFUN_ENV_URL);
// the env origin is joined only when dialling.
const PROVIDERS: Provider[] = [
  { id: "groq", wire: "openai", auth: "bearer", keyEnv: "GROQ_API_KEY", key: "sim-groq-0001", path: "/p/groq" },
  { id: "cerebras", wire: "openai", auth: "bearer", keyEnv: "CEREBRAS_API_KEY", key: "sim-cerebras-0002", path: "/p/cerebras" },
  { id: "gemini", wire: "gemini", auth: "x-goog-api-key", keyEnv: "GEMINI_API_KEY", key: "sim-gemini-0003", path: "/p/gemini" },
  { id: "zai", wire: "anthropic", auth: "bearer", keyEnv: "ZAI_API_KEY", key: "sim-zai-0004", path: "/p/zai" },
  { id: "openrouter", wire: "openai", auth: "bearer", keyEnv: "OPENROUTER_API_KEY", key: "sim-openrouter-0005", path: "/p/openrouter" },
  { id: "sambanova", wire: "openai", auth: "bearer", keyEnv: "SAMBANOVA_API_KEY", key: "sim-sambanova-0006", path: "/p/sambanova" },
  { id: "pollinations", wire: "openai", auth: "none", keyEnv: "", key: "", path: "/p/pollinations" },
  { id: "cloudflare", wire: "openai", auth: "bearer", keyEnv: "CLOUDFLARE_API_TOKEN", key: "sim-cloudflare-0007", path: "/p/cloudflare" },
  { id: "ovh", wire: "openai", auth: "none", keyEnv: "", key: "", path: "/p/ovh" },
  { id: "kilol", wire: "openai", auth: "none", keyEnv: "", key: "", path: "/p/kilol" },
];

// The quality table (free models only), in the brief's provider order.
// rating is milli-points; context is the model's context window in tokens.
const MODELS: ModelRow[] = [
  { id: "llama-3.1-8b-instant", provider: "groq", rating: 1210, context: 131072, tools: true, vision: false, json: true, free: true },
  { id: "gpt-oss-120b", provider: "groq", rating: 1352, context: 131072, tools: true, vision: false, json: true, free: true },
  { id: "qwen3.8-27b", provider: "groq", rating: 1434, context: 131072, tools: true, vision: false, json: true, free: true },
  { id: "gpt-oss-20b", provider: "groq", rating: 1318, context: 131072, tools: true, vision: false, json: false, free: true },
  { id: "llama-4-scout-17b", provider: "cerebras", rating: 1250, context: 131072, tools: true, vision: false, json: false, free: true },
  { id: "qwen-3.8-27b", provider: "cerebras", rating: 1434, context: 131072, tools: true, vision: false, json: true, free: true },
  { id: "gpt-oss-120b", provider: "cerebras", rating: 1352, context: 131072, tools: true, vision: false, json: true, free: true },
  { id: "gemini-3.8-flash", provider: "gemini", rating: 1458, context: 1048576, tools: true, vision: true, json: true, free: true },
  { id: "gemma-4-31b", provider: "gemini", rating: 1452, context: 131072, tools: true, vision: true, json: false, free: true },
  { id: "gemma-4-26b", provider: "gemini", rating: 1410, context: 32768, tools: false, vision: true, json: false, free: true },
  { id: "glm-4.7-flash", provider: "zai", rating: 1470, context: 204800, tools: true, vision: false, json: true, free: true },
  { id: "glm-4.6v-flash", provider: "zai", rating: 1432, context: 204800, tools: true, vision: true, json: false, free: true },
  { id: "z-ai/glm-5.3-flash:free", provider: "openrouter", rating: 1475, context: 204800, tools: true, vision: false, json: true, free: true },
  { id: "deepseek/deepseek-v4.1:free", provider: "openrouter", rating: 1462, context: 163840, tools: true, vision: false, json: true, free: true },
  { id: "minimax/m3:free", provider: "openrouter", rating: 1436, context: 196608, tools: true, vision: false, json: true, free: true },
  { id: "deepseek-v3.2", provider: "sambanova", rating: 1465, context: 163840, tools: true, vision: false, json: true, free: true },
  { id: "gemma-4-31b", provider: "sambanova", rating: 1452, context: 131072, tools: false, vision: true, json: false, free: true },
  { id: "mistral-nemo", provider: "pollinations", rating: 1180, context: 32768, tools: true, vision: false, json: false, free: true },
  { id: "openai-fast", provider: "pollinations", rating: 1390, context: 32768, tools: true, vision: false, json: false, free: true },
  { id: "glm-4.7-flash", provider: "cloudflare", rating: 1470, context: 204800, tools: true, vision: false, json: true, free: true },
  { id: "llama-3.3-70b", provider: "cloudflare", rating: 1330, context: 131072, tools: true, vision: false, json: false, free: true },
  { id: "nemotron-3.5-super-vl", provider: "cloudflare", rating: 1414, context: 131072, tools: false, vision: true, json: false, free: true },
  { id: "qwen3.5-397b", provider: "ovh", rating: 1440, context: 262144, tools: true, vision: false, json: false, free: true },
  { id: "llama-3.3-70b", provider: "ovh", rating: 1330, context: 131072, tools: true, vision: false, json: false, free: true },
  { id: "kilo-fast", provider: "kilol", rating: 1140, context: 131072, tools: true, vision: false, json: false, free: true },
  { id: "kilo-auto/free", provider: "kilol", rating: 1420, context: 131072, tools: true, vision: false, json: false, free: true },
];

// Paid models each provider lists. They are catalogued (free: false) and
// never picked as a serving model.
const PAID_MODELS: Record<string, Array<{ id: string; context: number | null }>> = {
  openrouter: [{ id: "anthropic/claude-opus-5.5", context: null }],
};

const PROVIDER_BY_ID = new Map(PROVIDERS.map((p) => [p.id, p]));

function buildCatalog(): Record<string, unknown> {
  const providers = PROVIDERS.map((p) => {
    const free = MODELS.filter((m) => m.provider === p.id).map((m) => ({
      id: m.id,
      free: true,
      context_length: m.context,
    }));
    const paid = (PAID_MODELS[p.id] ?? []).map((m) => ({
      id: m.id,
      free: false,
      context_length: m.context,
    }));
    return {
      id: p.id,
      base_url: p.path,
      key_env: p.keyEnv,
      wire: p.wire,
      models: [...free, ...paid],
    };
  });
  return { providers };
}

function freeModelIds(): string[] {
  return MODELS.map((m) => m.id);
}

// ---------------------------------------------------------------------------
// Request normalisation.
//
// A client request, in either wire, is reduced to one canonical shape.
// The fields below are exactly the ones the provider echoes back and the
// client compares, so a translation that keeps them keeps the echo:
// system text, the last user turn's text, the last image, the tool names
// in order, max_tokens, the stop sequences, and JSON mode.
// ---------------------------------------------------------------------------

interface ImageRef {
  mediaType: string;
  data: string;
}

type Image = { kind: "base64"; ref: ImageRef } | { kind: "url"; url: string };

interface Turn {
  role: "user" | "assistant";
  text: string;
  image: Image | null;
}

interface Tool {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

interface Req {
  api: "openai" | "anthropic";
  stream: boolean;
  systemText: string;
  turns: Turn[];
  tools: Tool[];
  toolChoice: unknown;
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | undefined;
  topP: number | undefined;
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") parts.push(block.text);
  }
  return parts.join("\n");
}

function parseDataUrl(url: string): ImageRef | null {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mediaType: m[1] === "" ? "image/png" : m[1], data: m[2] };
}

function imageFromOpenAI(url: unknown): Image | null {
  if (typeof url !== "string" || url === "") return null;
  const ref = parseDataUrl(url);
  return ref !== null ? { kind: "base64", ref } : { kind: "url", url };
}

function imageFromAnthropic(source: unknown): Image | null {
  if (!isRecord(source)) return null;
  if (source.type === "base64" && typeof source.data === "string") {
    return { kind: "base64", ref: { mediaType: typeof source.media_type === "string" ? source.media_type : "image/png", data: source.data } };
  }
  if (source.type === "url" && typeof source.url === "string") return { kind: "url", url: source.url };
  return null;
}

function dataUrlOf(image: Image): string {
  if (image.kind === "url") return image.url;
  return `data:${image.ref.mediaType};base64,${image.ref.data}`;
}

function normalize(body: Record<string, unknown>, api: "openai" | "anthropic"): Req {
  let systemText = "";
  const turns: Turn[] = [];
  const tools: Tool[] = [];
  if (api === "openai") {
    const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
    const systemParts: string[] = [];
    for (const message of messages) {
      if (message.role === "system") {
        const text = textOf(message.content);
        if (text !== "") systemParts.push(text);
      } else if (message.role === "user" || message.role === "assistant") {
        let image: Image | null = null;
        if (Array.isArray(message.content)) {
          for (const block of message.content) {
            if (isRecord(block) && block.type === "image_url" && isRecord(block.image_url)) {
              image = imageFromOpenAI(block.image_url.url) ?? image;
            }
          }
        }
        turns.push({ role: message.role, text: textOf(message.content), image });
      }
    }
    systemText = systemParts.join("\n");
    if (Array.isArray(body.tools)) {
      for (const tool of body.tools) {
        if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === "string") {
          tools.push({
            name: tool.function.name,
            description: typeof tool.function.description === "string" ? tool.function.description : "",
            parameters: isRecord(tool.function.parameters) ? tool.function.parameters : {},
          });
        }
      }
    }
    const format = isRecord(body.response_format) ? body.response_format.type : undefined;
    return {
      api,
      stream: body.stream === true,
      systemText,
      turns,
      tools,
      toolChoice: body.tool_choice,
      maxTokens: typeof body.max_tokens === "number" ? body.max_tokens : typeof body.max_completion_tokens === "number" ? body.max_completion_tokens : null,
      stops: Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === "string") : [],
      jsonMode: format === "json_object" || format === "json_schema",
      temperature: typeof body.temperature === "number" ? body.temperature : undefined,
      topP: typeof body.top_p === "number" ? body.top_p : undefined,
    };
  }
  systemText = textOf(body.system);
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  for (const message of messages) {
    if (message.role !== "user" && message.role !== "assistant") continue;
    let image: Image | null = null;
    if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (isRecord(block) && block.type === "image" && isRecord(block.source)) {
          image = imageFromAnthropic(block.source) ?? image;
        }
      }
    }
    turns.push({ role: message.role, text: textOf(message.content), image });
  }
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && typeof tool.name === "string") {
        tools.push({
          name: tool.name,
          description: typeof tool.description === "string" ? tool.description : "",
          parameters: isRecord(tool.input_schema) ? tool.input_schema : {},
        });
      }
    }
  }
  return {
    api,
    stream: body.stream === true,
    systemText,
    turns,
    tools,
    toolChoice: body.tool_choice,
    maxTokens: typeof body.max_tokens === "number" ? body.max_tokens : null,
    stops: Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === "string") : [],
    jsonMode: false,
    temperature: typeof body.temperature === "number" ? body.temperature : undefined,
    topP: typeof body.top_p === "number" ? body.top_p : undefined,
  };
}

// The last user turn's text and the last image in any user turn: those are
// what each wire's parser reports, and what the provider's usage counts.
function userTextOf(req: Req): string {
  for (let i = req.turns.length - 1; i >= 0; i--) {
    if (req.turns[i].role === "user") return req.turns[i].text;
  }
  return "";
}

function imageOf(req: Req): Image | null {
  let found: Image | null = null;
  for (const turn of req.turns) {
    if (turn.role === "user" && turn.image !== null) found = turn.image;
  }
  return found;
}

// The provider's usage formula, applied to the parsed upstream body:
// input = 1 + floor((system + user + tools.join(",")) / 4), output from the
// answer length. Used only as a cross-check: upstream answers carry usage.
function usageFormula(req: Req, answerText: string): { input: number; output: number } {
  const prompt = req.systemText.length + userTextOf(req).length + req.tools.map((t) => t.name).join(",").length;
  return { input: 1 + Math.floor(prompt / 4), output: 1 + Math.floor(answerText.length / 4) };
}

// ---------------------------------------------------------------------------
// Upstream bodies: the canonical request rendered in a provider's wire.
// Every rendering carries the same system text, user text, image, tool
// names in order, max_tokens, stops and JSON mode, so the provider's echo
// matches the client's request whichever wire serves it.
// ---------------------------------------------------------------------------

function openAIUpstream(req: Req, model: string): Record<string, unknown> {
  const messages: Array<Record<string, unknown>> = [];
  if (req.systemText !== "") messages.push({ role: "system", content: req.systemText });
  for (const turn of req.turns) {
    if (turn.image !== null) {
      messages.push({
        role: turn.role,
        content: [
          { type: "text", text: turn.text },
          { type: "image_url", image_url: { url: dataUrlOf(turn.image) } },
        ],
      });
    } else {
      messages.push({ role: turn.role, content: turn.text });
    }
  }
  const body: Record<string, unknown> = { model, messages, stream: false };
  if (req.tools.length > 0) {
    body.tools = req.tools.map((t) => ({ type: "function", function: { name: t.name, description: t.description, parameters: t.parameters } }));
    if (req.toolChoice !== undefined) body.tool_choice = req.toolChoice;
  }
  if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
  if (req.stops.length > 0) body.stop = req.stops;
  if (req.jsonMode) body.response_format = { type: "json_object" };
  if (req.temperature !== undefined) body.temperature = req.temperature;
  if (req.topP !== undefined) body.top_p = req.topP;
  return body;
}

function anthropicUpstream(req: Req, model: string): Record<string, unknown> {
  const messages: Array<Record<string, unknown>> = [];
  for (const turn of req.turns) {
    if (turn.image !== null) {
      const source =
        turn.image.kind === "base64"
          ? { type: "base64", media_type: turn.image.ref.mediaType, data: turn.image.ref.data }
          : { type: "url", url: turn.image.url };
      messages.push({ role: turn.role, content: [{ type: "text", text: turn.text }, { type: "image", source }] });
    } else {
      messages.push({ role: turn.role, content: turn.text });
    }
  }
  const body: Record<string, unknown> = { model, messages, stream: false };
  if (req.systemText !== "") body.system = req.systemText;
  if (req.tools.length > 0) {
    body.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
    if (req.toolChoice !== undefined) body.tool_choice = req.toolChoice;
  }
  if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
  if (req.stops.length > 0) body.stop_sequences = req.stops;
  if (req.temperature !== undefined) body.temperature = req.temperature;
  if (req.topP !== undefined) body.top_p = req.topP;
  return body;
}

function geminiUpstream(req: Req): Record<string, unknown> {
  const contents: Array<Record<string, unknown>> = [];
  for (const turn of req.turns) {
    const parts: Array<Record<string, unknown>> = [{ text: turn.text }];
    if (turn.image !== null) {
      if (turn.image.kind === "base64") {
        parts.push({ inlineData: { mimeType: turn.image.ref.mediaType, data: turn.image.ref.data } });
      } else {
        parts.push({ fileData: { fileUri: turn.image.url } });
      }
    }
    contents.push({ role: turn.role === "assistant" ? "model" : "user", parts });
  }
  const body: Record<string, unknown> = { contents };
  if (req.systemText !== "") body.systemInstruction = { parts: [{ text: req.systemText }] };
  if (req.tools.length > 0) {
    body.tools = [{ functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  const config: Record<string, unknown> = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.jsonMode) config.responseMimeType = "application/json";
  if (req.temperature !== undefined) config.temperature = req.temperature;
  if (req.topP !== undefined) config.topP = req.topP;
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

// ---------------------------------------------------------------------------
// Upstream calls. The provider is always called non-streaming; the answer
// is re-rendered in the client's wire afterwards. A provider answer is
// reduced to text, a stop word in its own wire's spelling, and usage.
// ---------------------------------------------------------------------------

interface ProviderAnswer {
  ok: boolean;
  status: number;
  text: string;
  stop: string | null;
  usage: { input: number; output: number } | null;
  // Why a refused call refused. All three refusal kinds share one status
  // and code per provider; only the error message tells them apart:
  // "budget is spent" and "is out" are terminal for the trial, while
  // "refused this call" is one scheduled failure and worth retrying.
  refusal: "budget" | "outage" | "transient" | null;
}

function classifyRefusal(raw: string): "budget" | "outage" | "transient" {
  let message = "";
  try {
    const value: unknown = JSON.parse(raw);
    if (isRecord(value) && isRecord(value.error) && typeof value.error.message === "string") message = value.error.message;
  } catch {
    message = "";
  }
  if (message.includes("budget is spent")) return "budget";
  if (message.includes(" is out")) return "outage";
  return "transient";
}

function authHeaders(provider: Provider): Record<string, string> {
  if (provider.auth === "bearer") return { authorization: `Bearer ${provider.key}` };
  if (provider.auth === "x-api-key") return { "x-api-key": provider.key };
  if (provider.auth === "x-goog-api-key") return { "x-goog-api-key": provider.key };
  return {};
}

function textFromBlocks(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") parts.push(block.text);
  }
  return parts.join("");
}

function parseUpstream(provider: Provider, status: number, raw: string): ProviderAnswer {
  if (status < 200 || status >= 300) return { ok: false, status, text: "", stop: null, usage: null, refusal: classifyRefusal(raw) };
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return { ok: false, status: 502, text: "", stop: null, usage: null, refusal: "transient" };
  }
  if (!isRecord(value)) return { ok: false, status: 502, text: "", stop: null, usage: null, refusal: "transient" };
  if (provider.wire === "openai") {
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
    const message = choice !== null && isRecord(choice.message) ? choice.message : null;
    const usage = isRecord(value.usage)
      ? { input: typeof value.usage.prompt_tokens === "number" ? value.usage.prompt_tokens : 0, output: typeof value.usage.completion_tokens === "number" ? value.usage.completion_tokens : 0 }
      : null;
    return {
      ok: true,
      status,
      text: message !== null ? textFromBlocks(message.content) : "",
      stop: choice !== null && typeof choice.finish_reason === "string" ? choice.finish_reason : null,
      usage,
      refusal: null,
    };
  }
  if (provider.wire === "anthropic") {
    const usage = isRecord(value.usage)
      ? { input: typeof value.usage.input_tokens === "number" ? value.usage.input_tokens : 0, output: typeof value.usage.output_tokens === "number" ? value.usage.output_tokens : 0 }
      : null;
    return {
      ok: true,
      status,
      text: textFromBlocks(value.content),
      stop: typeof value.stop_reason === "string" ? value.stop_reason : null,
      usage,
      refusal: null,
    };
  }
  const candidates = Array.isArray(value.candidates) ? value.candidates : [];
  const first = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : null;
  const content = first !== null && isRecord(first.content) ? first.content : null;
  const parts = content !== null && Array.isArray(content.parts) ? content.parts : [];
  const text = parts.filter(isRecord).map((p) => (typeof p.text === "string" ? p.text : "")).join("");
  const meta = isRecord(value.usageMetadata) ? value.usageMetadata : null;
  const usage =
    meta !== null
      ? { input: typeof meta.promptTokenCount === "number" ? meta.promptTokenCount : 0, output: typeof meta.candidatesTokenCount === "number" ? meta.candidatesTokenCount : 0 }
      : null;
  return {
    ok: true,
    status,
    text,
    stop: first !== null && typeof first.finishReason === "string" ? first.finishReason : null,
    usage,
    refusal: null,
  };
}

async function callProvider(envUrl: string, provider: Provider, req: Req, model: string): Promise<ProviderAnswer> {
  let path: string;
  let body: Record<string, unknown>;
  if (provider.wire === "openai") {
    path = "/v1/chat/completions";
    body = openAIUpstream(req, model);
  } else if (provider.wire === "anthropic") {
    path = "/v1/messages";
    body = anthropicUpstream(req, model);
  } else {
    path = `/v1beta/models/${encodeURIComponent(model)}:generateContent`;
    body = geminiUpstream(req);
  }
  const headers: Record<string, string> = { "content-type": "application/json", ...authHeaders(provider) };
  if (provider.wire === "anthropic") headers["anthropic-version"] = "2023-06-01";
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);
  try {
    const base = DIAL_BASE.get(provider.id) ?? `${envUrl}${provider.path}`;
    const res = await fetch(`${base}${path}`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    const raw = await res.text();
    return parseUpstream(provider, res.status, raw);
  } catch {
    return { ok: false, status: 0, text: "", stop: null, usage: null, refusal: "transient" };
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// Routing. Candidates are free models meeting the request's manifest
// requirements, with JSON-mode requests from an OpenAI client restricted
// to OpenAI-wire and Gemini-wire providers. Best-rated first; a refusal
// is retried twice on the same provider (usually one scheduled failure),
// then the next candidate is tried.
// ---------------------------------------------------------------------------

interface Requirements {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

function meetsRequirements(model: ModelRow, req: Requirements): boolean {
  if (req.tools === true && !model.tools) return false;
  if (req.vision === true && !model.vision) return false;
  if (req.json === true && !model.json) return false;
  if (typeof req.min_context === "number" && (model.context === null || model.context < req.min_context)) return false;
  return true;
}

function candidatesFor(req: Req, requires: Requirements): ModelRow[] {
  const list = MODELS.filter((m) => {
    if (!meetsRequirements(m, requires)) return false;
    if (req.api === "openai" && req.jsonMode) {
      const wire = PROVIDER_BY_ID.get(m.provider)?.wire;
      if (wire !== "openai" && wire !== "gemini") return false;
    }
    return true;
  });
  list.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.provider !== b.provider) return a.provider < b.provider ? -1 : 1;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
  return list;
}

interface ServeResult {
  answer: ProviderAnswer | null;
  model: ModelRow | null;
}

// Providers whose budget is spent or whose outage has begun: terminal
// for the trial, so no further call is wasted on them. A fresh process
// serves each trial, so this set never needs resetting.
const DEAD_PROVIDERS = new Set<string>();

async function serve(envUrl: string, req: Req, requires: Requirements): Promise<ServeResult> {
  const candidates = candidatesFor(req, requires);
  for (const model of candidates) {
    const provider = PROVIDER_BY_ID.get(model.provider);
    if (provider === undefined || DEAD_PROVIDERS.has(provider.id)) continue;
    for (let attempt = 0; attempt < MAX_ATTEMPTS_PER_PROVIDER; attempt++) {
      const answer = await callProvider(envUrl, provider, req, model.id);
      if (answer.ok) return { answer, model };
      if (answer.refusal === "budget" || answer.refusal === "outage") {
        DEAD_PROVIDERS.add(provider.id);
        break;
      }
    }
  }
  return { answer: null, model: null };
}

// ---------------------------------------------------------------------------
// Rendering back into the client's wire. Stop reasons are mapped by
// family: stop/end_turn/STOP, length/max_tokens/MAX_TOKENS,
// tool_calls/tool_use, content_filter/refusal/SAFETY.
// ---------------------------------------------------------------------------

const STOP_TO_OPENAI: Record<string, string> = {
  stop: "stop",
  end_turn: "stop",
  STOP: "stop",
  length: "length",
  max_tokens: "length",
  MAX_TOKENS: "length",
  tool_calls: "tool_calls",
  function_call: "tool_calls",
  tool_use: "tool_calls",
  content_filter: "content_filter",
  refusal: "content_filter",
  SAFETY: "content_filter",
};

const STOP_TO_ANTHROPIC: Record<string, string> = {
  stop: "end_turn",
  end_turn: "end_turn",
  STOP: "end_turn",
  length: "max_tokens",
  max_tokens: "max_tokens",
  MAX_TOKENS: "max_tokens",
  tool_calls: "tool_use",
  function_call: "tool_use",
  tool_use: "tool_use",
  content_filter: "refusal",
  refusal: "refusal",
  SAFETY: "refusal",
};

function mapStop(stop: string | null, api: "openai" | "anthropic"): string {
  if (stop === null) return api === "openai" ? "stop" : "end_turn";
  if (api === "openai") return STOP_TO_OPENAI[stop] ?? "stop";
  return STOP_TO_ANTHROPIC[stop] ?? "end_turn";
}

function splitPieces(text: string): string[] {
  const pieces: string[] = [];
  for (let i = 0; i < text.length; i += 13) pieces.push(text.slice(i, i + 13));
  return pieces;
}

function renderOpenAI(req: Req, answer: ProviderAnswer): { status: number; contentType: string; body: string } {
  const stop = mapStop(answer.stop, "openai");
  const usage = answer.usage ?? usageFormula(req, answer.text);
  if (!req.stream) {
    return {
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: "chatcmpl-agg",
        object: "chat.completion",
        created: FIXED_CREATED,
        model: "auto",
        choices: [{ index: 0, message: { role: "assistant", content: answer.text }, finish_reason: stop, logprobs: null }],
        usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
      }),
    };
  }
  const frames: string[] = [];
  const chunk = (delta: Record<string, unknown>, finish: string | null): string =>
    `data: ${JSON.stringify({ id: "chatcmpl-agg", object: "chat.completion.chunk", created: FIXED_CREATED, model: "auto", choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
  frames.push(chunk({ role: "assistant", content: "" }, null));
  for (const piece of splitPieces(answer.text)) frames.push(chunk({ content: piece }, null));
  frames.push(chunk({}, stop));
  frames.push(
    `data: ${JSON.stringify({ id: "chatcmpl-agg", object: "chat.completion.chunk", created: FIXED_CREATED, model: "auto", choices: [], usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output } })}\n\n`,
  );
  frames.push("data: [DONE]\n\n");
  return { status: 200, contentType: "text/event-stream; charset=utf-8", body: frames.join("") };
}

function renderAnthropic(req: Req, answer: ProviderAnswer): { status: number; contentType: string; body: string } {
  const stop = mapStop(answer.stop, "anthropic");
  const usage = answer.usage ?? usageFormula(req, answer.text);
  if (!req.stream) {
    return {
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: "msg_agg",
        type: "message",
        role: "assistant",
        model: "auto",
        content: [{ type: "text", text: answer.text }],
        stop_reason: stop,
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: usage.output },
      }),
    };
  }
  const frames: string[] = [];
  const event = (name: string, data: Record<string, unknown>): string => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
  frames.push(
    event("message_start", {
      type: "message_start",
      message: { id: "msg_agg", type: "message", role: "assistant", model: "auto", content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.input, output_tokens: 0 } },
    }),
  );
  frames.push(event("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }));
  for (const piece of splitPieces(answer.text)) {
    frames.push(event("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: piece } }));
  }
  frames.push(event("content_block_stop", { type: "content_block_stop", index: 0 }));
  frames.push(
    event("message_delta", { type: "message_delta", delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: usage.output } }),
  );
  frames.push(event("message_stop", { type: "message_stop" }));
  return { status: 200, contentType: "text/event-stream; charset=utf-8", body: frames.join("") };
}

// ---------------------------------------------------------------------------
// Server and trial loop.
// ---------------------------------------------------------------------------

const CATALOG = buildCatalog();

// Dial bases per provider: absolute base URL from GET /instance when it
// answers, else $MAKEFUN_ENV_URL joined to the provider's own path.
const DIAL_BASE = new Map<string, string>();
// The workload manifest: request id -> requirements. Filled from
// GET /instance; empty when that call fails, in which case requirements
// are derived from the request body alone.
const MANIFEST = new Map<string, Requirements>();
// The manifest's request order, for the arrival-count fallback.
const MANIFEST_ORDER: string[] = [];
// Requests served so far, in arrival order (the client sends one request
// at a time, in manifest order).
let servedCount = 0;
// The request id being served right now, for logging.
let currentRequestId: string | null = null;

async function fetchInstance(envUrl: string): Promise<void> {
  try {
    const res = await fetch(`${envUrl}/instance`);
    if (!res.ok) return;
    const value: unknown = await res.json();
    if (!isRecord(value)) return;
    if (Array.isArray(value.providers)) {
      for (const entry of value.providers) {
        if (!isRecord(entry) || typeof entry.id !== "string") continue;
        const known = PROVIDER_BY_ID.get(entry.id);
        if (known === undefined) continue;
        if (typeof entry.base_url === "string" && entry.base_url !== "") DIAL_BASE.set(entry.id, entry.base_url.replace(/\/$/, ""));
        if (typeof entry.key === "string" && entry.key !== "") known.key = entry.key;
        if (entry.wire === "openai" || entry.wire === "anthropic" || entry.wire === "gemini") known.wire = entry.wire;
        if (entry.auth === "bearer" || entry.auth === "x-api-key" || entry.auth === "x-goog-api-key" || entry.auth === "none") known.auth = entry.auth;
      }
    }
    if (isRecord(value.workload) && Array.isArray(value.workload.manifest)) {
      for (const entry of value.workload.manifest) {
        if (isRecord(entry) && typeof entry.id === "string" && isRecord(entry.requires)) {
          const requires: Requirements = {};
          if (entry.requires.tools === true) requires.tools = true;
          if (entry.requires.vision === true) requires.vision = true;
          if (entry.requires.json === true) requires.json = true;
          if (typeof entry.requires.min_context === "number") requires.min_context = entry.requires.min_context;
          MANIFEST.set(entry.id, requires);
          MANIFEST_ORDER.push(entry.id);
        }
      }
    }
  } catch {
    // Keep the hardcoded directory and body-derived requirements.
  }
}

function requiresFor(req: Req): Requirements {
  if (currentRequestId !== null) {
    const known = MANIFEST.get(currentRequestId);
    if (known !== undefined) return known;
  }
  const derived: Requirements = {};
  if (req.tools.length > 0) derived.tools = true;
  if (imageOf(req) !== null) derived.vision = true;
  if (req.jsonMode) derived.json = true;
  return derived;
}

function modelListResponse(req: IncomingMessage): { status: number; contentType: string; body: string } {
  const free = freeModelIds();
  if (typeof req.headers["x-api-key"] === "string") {
    return {
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        data: free.map((id) => {
          const row = MODELS.find((m) => m.id === id);
          return { type: "model", id, display_name: id, created_at: "2026-01-01T00:00:00Z", free: true, context_length: row?.context ?? null };
        }),
        has_more: false,
      }),
    };
  }
  return {
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({
      object: "list",
      data: free.map((id) => {
        const row = MODELS.find((m) => m.id === id);
        return { id, object: "model", created: FIXED_CREATED, owned_by: "aggregator", free: true, context_length: row?.context ?? null };
      }),
    }),
  };
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size <= 2 * 1024 * 1024) chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

async function handleCompletion(envUrl: string, api: "openai" | "anthropic", httpReq: IncomingMessage, res: ServerResponse): Promise<void> {
  const raw = await readBody(httpReq);
  let body: unknown;
  try {
    body = JSON.parse(raw);
  } catch {
    res.writeHead(400, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: { message: "the request body is not JSON" } }));
    return;
  }
  if (!isRecord(body)) {
    res.writeHead(400, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: { message: "the request body is not one JSON object" } }));
    return;
  }
  const req = normalize(body, api);
  // Resolve which request this is before choosing a model: ask the env
  // what it is sending right now (its index still points at this request
  // until the answer is delivered), and fall back to arrival order.
  currentRequestId = (await resolveRequestId(envUrl)) ?? MANIFEST_ORDER[servedCount] ?? null;
  servedCount++;
  const requires = requiresFor(req);
  const { answer, model } = await serve(envUrl, req, requires);
  if (answer === null || model === null) {
    process.stderr.write(`aggregator: ${currentRequestId ?? "?"} no provider answered\n`);
    res.writeHead(502, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: { message: "every provider refused" } }));
    return;
  }
  const rendered = api === "openai" ? renderOpenAI(req, answer) : renderAnthropic(req, answer);
  process.stderr.write(`aggregator: ${currentRequestId ?? "?"} served by ${model.provider}/${model.id} (${model.rating})\n`);
  res.writeHead(rendered.status, { "content-type": rendered.contentType });
  res.end(rendered.body);
}

async function register(envUrl: string, port: number): Promise<boolean> {
  for (let attempt = 0; attempt < 30; attempt++) {
    try {
      const res = await fetch(`${envUrl}/register`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: CATALOG }),
      });
      if (res.ok) {
        const value: unknown = await res.json().catch(() => null);
        process.stderr.write(`aggregator: registered on port ${port} (${JSON.stringify(value)})\n`);
        return true;
      }
    } catch {
      // The env may not be listening yet; retry below.
    }
    await new Promise((r) => setTimeout(r, 200));
  }
  return false;
}

async function pollWork(envUrl: string): Promise<boolean> {
  const res = await fetch(`${envUrl}/work`);
  if (!res.ok) return false;
  const value: unknown = await res.json().catch(() => null);
  if (!isRecord(value)) return false;
  return value.done === true;
}

// The request the env is sending right now. Read inside the handler,
// while the env's index still points at this request; null when the call
// itself fails, and the handler falls back to arrival order.
async function resolveRequestId(envUrl: string): Promise<string | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 5000);
  try {
    const res = await fetch(`${envUrl}/work`, { signal: controller.signal });
    if (!res.ok) return null;
    const value: unknown = await res.json().catch(() => null);
    if (!isRecord(value) || typeof value.next !== "string") return null;
    return value.next;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

async function main(): Promise<number> {
  if (process.argv.includes("--list")) {
    process.stdout.write(`${JSON.stringify(CATALOG)}\n`);
    return 0;
  }
  const envUrl = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/$/, "");
  if (envUrl === "") {
    process.stderr.write("aggregator: MAKEFUN_ENV_URL is not set\n");
    return 1;
  }
  const promptFile = process.env.MAKEFUN_PROMPT_FILE;
  if (promptFile !== undefined && promptFile !== "") {
    // The brief is the one file the agent reads; the sealed tables above
    // already carry everything it publishes, so this is only a presence
    // check and never changes a decision.
    await readFile(promptFile, "utf8").catch(() => "");
  }
  await fetchInstance(envUrl);
  for (const provider of PROVIDERS) {
    if (!DIAL_BASE.has(provider.id)) DIAL_BASE.set(provider.id, `${envUrl}${provider.path}`);
  }
  const server = createServer((httpReq, res) => {
    const url = new URL(httpReq.url ?? "/", "http://entry");
    if (httpReq.method === "GET" && url.pathname === "/v1/models") {
      const out = modelListResponse(httpReq);
      res.writeHead(out.status, { "content-type": out.contentType });
      res.end(out.body);
      return;
    }
    if (httpReq.method === "POST" && url.pathname === "/v1/chat/completions") {
      void handleCompletion(envUrl, "openai", httpReq, res);
      return;
    }
    if (httpReq.method === "POST" && url.pathname === "/v1/messages") {
      void handleCompletion(envUrl, "anthropic", httpReq, res);
      return;
    }
    res.writeHead(404, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: { message: "not found" } }));
  });
  const port = await new Promise<number>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => {
      const address = server.address();
      if (address !== null && typeof address === "object") resolve(address.port);
      else reject(new Error("the server did not bind a port"));
    });
  });
  process.stderr.write(`aggregator: listening on 0.0.0.0:${port}\n`);
  if (!(await register(envUrl, port))) {
    process.stderr.write("aggregator: register failed\n");
    server.close();
    return 1;
  }
  for (;;) {
    let done = false;
    try {
      done = await pollWork(envUrl);
    } catch {
      done = false;
    }
    if (done) break;
    await sleep(WORK_POLL_MS);
  }
  server.close();
  return 0;
}

process.exitCode = await main();
