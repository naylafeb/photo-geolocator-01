// photo-geolocator-01 entry: an agent that says where each photo was taken.
//
// For each of the instance's 25 photos it asks a vision model through the
// trial gateway (Anthropic claude-sonnet-5-5 first, OpenAI gpt-6.1-sol if the
// Anthropic route has no key), with a prompt that makes the model read the
// visible clues before it guesses. A photo whose model calls all fail gets a
// fixed prior guess instead of being left unanswered (unanswered counts
// 20,015 km; a prior in central Europe is far closer on average).
//
// Deterministic given the models' answers: every request is a function of the
// photo's bytes and fixed prompts, calls run four at a time, answers are
// posted in photo order, no wall clock and no unseeded randomness anywhere.

const ANTHROPIC_MODEL = 'claude-sonnet-5-5';
const OPENAI_MODEL = 'gpt-6.1-sol';
const MAX_TOKENS = 1024;
const CONCURRENCY = 4;
const RETRIES = 3;
const RETRY_STATUSES = [408, 429, 500, 502, 503, 504, 529];

// One fixed prior for photos no model could place: central Europe, the
// densest Mapillary region, so the expected error is far below 20,015 km.
const PRIOR = { lat: 48.0, lon: 10.0 };

const PROMPT =
  'Look carefully at this street-level photo and work out where it was taken. ' +
  'Reason from what you can see: the language and script on signs, road markings ' +
  'and which side traffic drives on, license plates, vegetation and climate, ' +
  'architecture and building materials, terrain, vehicles and street furniture. ' +
  'Narrow it down step by step: continent, then country, then region, then city ' +
  'or town, then the street if the clues allow. Then give your best estimate as ' +
  'one line of JSON and nothing else: {"lat": <degrees north>, "lon": <degrees east>}.';

interface Photo {
  id: string;
  url: string;
  answered: boolean;
}

interface Guess {
  id: string;
  lat: number;
  lon: number;
}

function opt(name: string): string | undefined {
  const v = process.env[name];
  return v === undefined || v === '' ? undefined : v;
}

function need(name: string): string {
  const v = opt(name);
  if (v === undefined) throw new Error(`${name} is not set`);
  return v;
}

const envUrl = need('MAKEFUN_ENV_URL').replace(/\/+$/, '');
const anthropicUrl = opt('ANTHROPIC_BASE_URL')?.replace(/\/+$/, '');
const openaiUrl = opt('OPENAI_BASE_URL')?.replace(/\/+$/, '');
const anthropicKey = opt('ANTHROPIC_API_KEY');
const openaiKey = opt('OPENAI_API_KEY');

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

type AskResult = 'ok' | 'no-key' | 'refused';
interface AskOutcome {
  status: AskResult;
  text: string | null;
}

/** One model call. 'ok' carries the model's text; 'no-key' means the gateway
 *  answered 401/403 (no key pushed for this vendor); 'refused' is any other
 *  final failure. Transient provider errors are retried with fixed backoff. */
async function askAnthropic(jpeg: Buffer): Promise<AskOutcome> {
  if (!anthropicUrl || !anthropicKey) return { status: 'no-key', text: null };
  const body = JSON.stringify({
    model: ANTHROPIC_MODEL,
    max_tokens: MAX_TOKENS,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: jpeg.toString('base64') } },
          { type: 'text', text: PROMPT },
        ],
      },
    ],
  });
  for (let attempt = 0; ; attempt += 1) {
    let res: Response;
    try {
      res = await fetch(`${anthropicUrl}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'x-api-key': anthropicKey, 'anthropic-version': '2023-06-01' },
        body,
      });
    } catch (err) {
      process.stderr.write(`agent: anthropic fetch failed: ${String(err).slice(0, 200)}\n`);
      return { status: 'refused', text: null };
    }
    if (res.ok) {
      const msg = (await res.json()) as { content?: { type: string; text?: string }[] };
      const text = (msg.content ?? [])
        .filter((b) => b.type === 'text')
        .map((b) => b.text ?? '')
        .join('\n');
      return { status: 'ok', text };
    }
    const text = await res.text();
    if ((res.status === 401 || res.status === 403)) return { status: 'no-key', text: null };
    if (RETRY_STATUSES.includes(res.status) && res.headers.get('x-should-retry') !== 'false' && attempt < RETRIES) {
      await sleep(2000 * (attempt + 1));
      continue;
    }
    process.stderr.write(`agent: anthropic refused (HTTP ${res.status}): ${text.slice(0, 200)}\n`);
    return { status: 'refused', text: null };
  }
}

async function askOpenai(jpeg: Buffer): Promise<AskOutcome> {
  if (!openaiUrl || !openaiKey) return { status: 'no-key', text: null };
  const body = JSON.stringify({
    model: OPENAI_MODEL,
    max_tokens: MAX_TOKENS,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: PROMPT },
          { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${jpeg.toString('base64')}` } },
        ],
      },
    ],
  });
  for (let attempt = 0; ; attempt += 1) {
    let res: Response;
    try {
      res = await fetch(`${openaiUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', authorization: `Bearer ${openaiKey}` },
        body,
      });
    } catch (err) {
      process.stderr.write(`agent: openai fetch failed: ${String(err).slice(0, 200)}\n`);
      return { status: 'refused', text: null };
    }
    if (res.ok) {
      const msg = (await res.json()) as { choices?: { message?: { content?: string | null } }[] };
      return { status: 'ok', text: msg.choices?.[0]?.message?.content ?? '' };
    }
    const text = await res.text();
    if ((res.status === 401 || res.status === 403)) return { status: 'no-key', text: null };
    if (RETRY_STATUSES.includes(res.status) && res.headers.get('x-should-retry') !== 'false' && attempt < RETRIES) {
      await sleep(2000 * (attempt + 1));
      continue;
    }
    process.stderr.write(`agent: openai refused (HTTP ${res.status}): ${text.slice(0, 200)}\n`);
    return { status: 'refused', text: null };
  }
}

const isLat = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -90 && v <= 90;
const isLon = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -180 && v <= 180;

/** The latitude and longitude an answer gives: its first JSON object with lat
 *  and lon, else its first two numbers in range. */
function coordinates(text: string): { lat: number; lon: number } | null {
  for (const m of text.matchAll(/\{[^{}]*\}/g)) {
    try {
      const o = JSON.parse(m[0]) as Record<string, unknown>;
      const lat = o.lat ?? o.latitude;
      const lon = o.lon ?? o.lng ?? o.longitude;
      if (isLat(lat) && isLon(lon)) return { lat, lon };
    } catch {
      // Not JSON: the next one, or the numbers below.
    }
  }
  const nums = [...text.matchAll(/-?\d+(?:\.\d+)?/g)].map((m) => Number(m[0]));
  for (let i = 0; i + 1 < nums.length; i += 1) {
    if (isLat(nums[i]) && isLon(nums[i + 1])) return { lat: nums[i]!, lon: nums[i + 1]! };
  }
  return null;
}

/** Place one photo: Anthropic, then OpenAI, then the fixed prior. */
async function place(jpeg: Buffer): Promise<{ lat: number; lon: number; how: string }> {
  const a = await askAnthropic(jpeg);
  if (a.status === 'ok' && a.text !== null) {
    const c = coordinates(a.text);
    if (c !== null) return { ...c, how: 'anthropic' };
    process.stderr.write(`agent: no coordinates in anthropic answer: ${a.text.slice(0, 200)}\n`);
  }
  const o = await askOpenai(jpeg);
  if (o.status === 'ok' && o.text !== null) {
    const c = coordinates(o.text);
    if (c !== null) return { ...c, how: 'openai' };
    process.stderr.write(`agent: no coordinates in openai answer: ${o.text.slice(0, 200)}\n`);
  }
  return { ...PRIOR, how: 'prior' };
}

async function main(): Promise<number> {
  const listed = (await (await fetch(`${envUrl}/photos`)).json()) as { photos: Photo[] };
  const photos = listed.photos.filter((p) => !p.answered);
  const guesses: Guess[] = new Array(photos.length);
  const how: string[] = new Array(photos.length).fill('prior');
  let next = 0;
  const worker = async (): Promise<void> => {
    while (next < photos.length) {
      const i = next;
      next += 1;
      const p = photos[i]!;
      const jpeg = Buffer.from(await (await fetch(`${envUrl}${p.url}`)).arrayBuffer());
      const g = await place(jpeg);
      guesses[i] = { id: p.id, lat: g.lat, lon: g.lon };
      how[i] = g.how;
    }
  };
  await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));
  const byHow: Record<string, number> = {};
  for (const h of how) byHow[h] = (byHow[h] ?? 0) + 1;
  process.stdout.write(`agent: ${guesses.length} of ${photos.length} photos placed (${JSON.stringify(byHow)})\n`);
  // Post one photo at a time, in photo order, so one refused or failed post
  // can never lose the answers already accepted. A post is retried only on
  // a network error or a 5xx; any other refusal is final for that photo.
  let posted = 0;
  for (const g of guesses) {
    let ok = false;
    for (let attempt = 0; attempt < 3 && !ok; attempt += 1) {
      if (attempt > 0) await sleep(2000 * attempt);
      try {
        const res = await fetch(`${envUrl}/answers`, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ answers: [g] }),
        });
        if (res.ok) {
          ok = true;
          posted += 1;
        } else {
          const text = await res.text();
          process.stderr.write(`agent: post ${g.id} refused (HTTP ${res.status}): ${text.slice(0, 200)}\n`);
          if (res.status < 500) break;
        }
      } catch (err) {
        process.stderr.write(`agent: post ${g.id} failed: ${String(err).slice(0, 200)}\n`);
      }
    }
    if (!ok) process.stderr.write(`agent: giving up on ${g.id} after 3 attempts\n`);
  }
  process.stdout.write(`agent: posted ${posted} of ${guesses.length} answers\n`);
  return posted === guesses.length ? 0 : 1;
}

try {
  process.exitCode = await main();
} catch (err) {
  process.stderr.write(`agent: ${(err as Error).stack ?? String(err)}\n`);
  process.exitCode = 1;
}
