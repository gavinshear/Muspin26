'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Head from "next/head";
import Image from 'next/image';
import type { Gain, Filter, FeedbackDelay, Reverb } from 'tone';
import type { Player, Loop, MembraneSynth, NoiseSynth, MetalSynth, FMSynth } from 'tone';

// Utilities
const clamp = (n: number, min: number, max: number) => Math.max(min, Math.min(max, n));
const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");

// Tone dynamic type
type ToneType = typeof import("tone");

// Complex math helpers
type Complex = { re: number; im: number };
const C = (re: number, im: number): Complex => ({ re, im });
const fromPolar = (r: number, theta: number): Complex => ({ re: r * Math.cos(theta), im: r * Math.sin(theta) });
const addC = (a: Complex, b: Complex): Complex => ({ re: a.re + b.re, im: a.im + b.im });
const subC = (a: Complex, b: Complex): Complex => ({ re: a.re - b.re, im: a.im - b.im });
const absC = (a: Complex): number => Math.hypot(a.re, a.im);

// Pattern cell type
type Cell = { on: boolean; vol: number }; // vol: 0..100 percent

type CycleClipboard = { beats: number; data: Cell[][] } | null;

// Config
const NUM_TRACKS = 6;
const INNER_SPACER_TRACKS = 2; // inner whitespace rings
const CANVAS_SIZE = 512; // canvas pixels
const DEFAULT_EVENT_VOLUME = 80; // default per-event volume %
const CLICK_THRESHOLD = 20; // px, for nearest-cell detection
const EVENT_VOLUME_STEP = 1; // slider step

const PALETTE = {
  bg: "#111827", // gray-900
  panel: "#1F2937", // gray-800
  grid: "#4B5563", // gray-600 (quieter grid so track colors read)
  playhead: "#EF4444", // red-500
  event: "#E5E7EB", // gray-200
  text: "#E5E7EB", // gray-200
  centerBtnIdle: "#3B82F6", // blue-500
  centerBtnActive: "#1D4ED8", // blue-700
};

// Per-track identity: color + name (matches the default kit)
const TRACK_COLORS = ["#F87171", "#FBBF24", "#34D399", "#60A5FA", "#A78BFA", "#F472B6"];
const TRACK_NAMES = ["Kick", "Snare", "Hat", "Clap", "Tom", "Perc"];

// Button styles (compact)
const btnBase = "px-2 py-1 text-xs rounded-md font-medium transition-colors text-center";
const btnPrimary = `${btnBase} bg-blue-600 hover:bg-blue-700 text-white`;
const btnSecondary = `${btnBase} bg-gray-700 hover:bg-gray-600 text-white`;
const btnMuted = `${btnBase} bg-gray-900 hover:bg-gray-700 text-gray-100 border border-gray-700`;
const btnFillBase =
  "flex items-center justify-center rounded-lg font-medium text-sm shadow transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500";
const btnFillSize = "w-full h-10"; // condensed height
const btnFillPrimary = `${btnFillBase} ${btnFillSize} bg-blue-600 hover:bg-blue-700 text-white`;
const btnFillSecondary = `${btnFillBase} ${btnFillSize} bg-gray-700 hover:bg-gray-600 text-white`;
const btnFillToggle = (active: boolean) =>
  `${btnFillBase} ${btnFillSize} ${active ? "bg-blue-600 text-white" : "bg-gray-700 text-white hover:bg-gray-600"}`;
const btnTiny = (active: boolean, activeClasses: string) =>
  `px-2 py-0.5 text-[10px] rounded font-bold transition-colors ${
    active ? activeClasses : "bg-gray-800 border border-gray-700 text-gray-400 hover:text-gray-200"
  }`;

// Map per-event volume percent (0..100) to dB.
const eventVolumeToDb = (vol: number) => {
  const v = clamp(vol, 0, 100);
  if (v <= 0) return -60; // effectively muted
  return -36 + 36 * (v / 100); // 1..100% => -36..0 dB
};

// Map filter cutoff percent (0..100) to Hz (exponential, 80 Hz .. 18 kHz)
const filterToHz = (v: number) => Math.round(80 * Math.pow(18000 / 80, clamp(v, 0, 100) / 100));

// ---------- Harmonic randomization helpers ----------
// Bresenham-style even pulse distribution across steps
function euclideanIndices(steps: number, pulses: number): number[] {
  if (pulses <= 0) return [];
  if (pulses >= steps) return Array.from({ length: steps }, (_, i) => i);
  const idx: number[] = [];
  let acc = 0;
  for (let i = 0; i < steps; i++) {
    acc += pulses;
    if (acc >= steps) {
      idx.push(i);
      acc -= steps;
    }
  }
  return idx;
}

function rotateIndices(indices: number[], offset: number, steps: number) {
  if (indices.length === 0) return indices.slice();
  return indices.map((i) => ((i + offset) % steps + steps) % steps);
}

// Pick a value using weighted probabilities
function weightedPick<T>(pairs: Array<{ value: T; weight: number }>, rng: () => number = Math.random): T {
  const total = pairs.reduce((s, p) => s + Math.max(0, p.weight), 0);
  if (total <= 0) return pairs[0].value;
  let r = rng() * total;
  for (const p of pairs) {
    const w = Math.max(0, p.weight);
    if (r < w) return p.value;
    r -= w;
  }
  return pairs[pairs.length - 1].value;
}

// Return harmonically pleasing anchor offsets over the step grid
function harmonicOffsets(steps: number): number[] {
  const divisors = [1, 2, 3, 4, 6, 8, 12];
  const offsets: number[] = [];
  for (const d of divisors) {
    if (steps % d !== 0) continue; // only include integer splits
    const unit = steps / d;
    for (let k = 0; k < d; k++) {
      const off = Math.round(k * unit);
      offsets.push(((off % steps) + steps) % steps);
    }
  }
  return Array.from(new Set(offsets)).sort((a, b) => a - b);
}

// Strength near harmonic divisions for accenting
function harmonicStrength(step: number, steps: number): number {
  const specs: Array<{ d: number; w: number }> = [
    { d: 2, w: 1.0 },
    { d: 3, w: 0.9 },
    { d: 4, w: 0.9 },
    { d: 6, w: 0.8 },
    { d: 8, w: 0.7 },
    { d: 12, w: 0.6 },
  ];
  let s = 0;
  for (const { d, w } of specs) {
    if (steps % d !== 0) continue;
    const unit = steps / d;
    const nearest = Math.round(step / unit) * unit;
    const dist = Math.min(Math.abs(step - nearest), Math.abs(step - (nearest + steps)));
    const closeness = clamp(1 - dist / (unit / 2), 0, 1);
    s += w * closeness;
  }
  const norm = specs.filter((x) => steps % x.d === 0).reduce((a, b) => a + b.w, 0) || 1;
  return clamp(s / norm, 0, 1);
}

// Favor counts that make musical sense and keep density reasonable
function pickPulsesForTrack(trackIndex: number, steps: number, rng: () => number) {
  const candidates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16].filter((k) => k > 0 && k < steps);
  const targetDensity = 0.16 + 0.06 * (trackIndex / Math.max(1, NUM_TRACKS - 1));
  const ideal = clamp(Math.round(steps * targetDensity), 1, Math.min(steps - 1, 12));
  const pairs = candidates.map((k) => {
    const isFactor = steps % k === 0 ? 1 : 0;
    const harmonicBonus = [2, 3, 4, 5, 6, 8, 9, 10, 12, 16].includes(k) ? 1 : 0;
    const densityPenalty = Math.abs(k - ideal) / steps;
    const w = 0.9 * (1 - densityPenalty) + 0.7 * isFactor + 0.6 * harmonicBonus + 0.2;
    return { value: k, weight: w };
  });
  return weightedPick(pairs, rng);
}

// Key harmonic drop probability (remove some events at strong grid points)
function keyHarmonicDropProb(trackIndex: number, b: number, steps: number): number {
  const isKick = trackIndex === 0;
  let p = 0;
  if (b === 0) {
    p = isKick ? 0.1 : 0.4; // downbeat
  } else if (steps % 2 === 0 && b % (steps / 2) === 0) {
    p = isKick ? 0.15 : 0.3; // halves
  } else if (steps % 4 === 0 && b % (steps / 4) === 0) {
    p = isKick ? 0.12 : 0.25; // quarters
  } else if (steps % 3 === 0 && b % (steps / 3) === 0) {
    p = 0.18; // triplet anchors
  }
  return clamp(p, 0, 0.9);
}

// Heavy‑tailed normal random (Box–Muller)
function normalRand(rng: () => number = Math.random): number {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); // mean 0, std 1
}

function SlimSlider({
  id,
  label,
  min,
  max,
  step = 1,
  value,
  onChange,
  display,
  className = "",
}: {
  id?: string;
  label: string;
  min: number;
  max: number;
  step?: number;
  value: number;
  onChange: (v: number) => void;
  display?: (v: number) => string;
  className?: string;
}) {
  const sliderId = id ?? `slider-${slug(label)}`;
  const shown = display ? display(value) : `${value}`;
  return (
    <div className={`flex flex-col items-stretch gap-1 bg-gray-900 rounded-lg p-2 border border-gray-700 w-full ${className}`}>
      <div className="flex items-center justify-between">
        <label htmlFor={sliderId} className="text-xs font-medium text-gray-200">
          {label}
        </label>
        <span id={`${sliderId}-value`} className="text-xs text-gray-300">
          {shown}
        </span>
      </div>
      <input
        id={sliderId}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={value}
        aria-valuetext={shown}
        aria-orientation="horizontal"
        aria-describedby={`${sliderId}-value`}
        className="w-full h-1"
      />
    </div>
  );
}

function CyclesBar({
  playing,
  currentDisplayCycle,
  totalCycles,
  onPrevCycle,
  onNextCycle,
  onSetTotal,
  onCopyCycle,
  onPasteCycle,
  canPaste,
  canvasId,
}: {
  playing: boolean;
  currentDisplayCycle: number;
  totalCycles: number;
  onPrevCycle: () => void;
  onNextCycle: () => void;
  onSetTotal: (n: number) => void;
  onCopyCycle: () => void;
  onPasteCycle: () => void;
  canPaste: boolean;
  canvasId?: string;
}) {
  return (
    <div className="flex items-center justify-center gap-2 bg-gray-800 rounded-lg p-2 border border-gray-700 w-full">
      <button
        className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-md disabled:opacity-60"
        onClick={onPrevCycle}
        disabled={playing}
        aria-label="Previous cycle"
        aria-controls={canvasId}
      >
        ‹
      </button>
      <span className="text-xs text-gray-300 w-16 text-center" aria-live="polite">
        {currentDisplayCycle}/{totalCycles}
      </span>
      <input
        type="number"
        min={1}
        max={16}
        value={totalCycles}
        onChange={(e) => onSetTotal(clamp(parseInt(e.target.value || "1"), 1, 16))}
        className="w-16 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-100 text-center focus:outline-none focus:ring-0"
        aria-label="Total cycles"
        aria-controls={canvasId}
      />
      <button
        className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-md disabled:opacity-60"
        onClick={onNextCycle}
        disabled={playing}
        aria-label="Next cycle"
        aria-controls={canvasId}
      >
        ›
      </button>
      <div className="w-px h-4 bg-gray-700 mx-1" aria-hidden="true" />
      <button
        className={btnSecondary}
        onClick={onCopyCycle}
        aria-label="Copy current cycle"
        aria-controls={canvasId}
      >
        Copy
      </button>
      <button
        className={canPaste ? btnPrimary : `${btnMuted} cursor-not-allowed`}
        onClick={onPasteCycle}
        disabled={!canPaste}
        aria-disabled={!canPaste}
        aria-label="Paste into current cycle"
        aria-controls={canvasId}
      >
        Paste
      </button>
    </div>
  );
}

export default function RadialSequencerPage() {
  // Canvas
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const canvasId = "radial-canvas";

  // Core state
  const [bpm, setBpm] = useState<number>(120);
  const [numBeats, setNumBeats] = useState<number>(16);
  const [numCycles, setNumCycles] = useState<number>(1);
  const [currentCycle, setCurrentCycle] = useState<number>(0);
  const [playing, setPlaying] = useState<boolean>(false);
  const [halfSpeed, setHalfSpeed] = useState<boolean>(false);
  const [globalVol, setGlobalVol] = useState<number>(100);
  const [trackVolumes, setTrackVolumes] = useState<number[]>(Array.from({ length: NUM_TRACKS }, () => 100));

  // Groove + mixing state
  const [swing, setSwing] = useState<number>(0); // 0..100, shifts every 2nd step toward triplet feel
  const swingRef = useRef<number>(0);
  useEffect(() => {
    swingRef.current = swing;
  }, [swing]);
  const [mutes, setMutes] = useState<boolean[]>(Array.from({ length: NUM_TRACKS }, () => false));
  const [solos, setSolos] = useState<boolean[]>(Array.from({ length: NUM_TRACKS }, () => false));

  // Master FX state
  const [filterCut, setFilterCut] = useState<number>(100); // lowpass cutoff, 100 = open
  const [reverbWet, setReverbWet] = useState<number>(12);
  const [delayWet, setDelayWet] = useState<number>(0);

  // Clipboard for cycle copy/paste
  const [clipboard, setClipboard] = useState<CycleClipboard>(null);

  // Store paused position
  const pausedStepRef = useRef<number>(0);

  // Pattern: per-cell on + volume
  const makeEmptyCell = (): Cell => ({ on: false, vol: DEFAULT_EVENT_VOLUME });
  const initTrack = (length: number) => Array.from({ length }, () => makeEmptyCell());
  const [pattern, setPattern] = useState<Cell[][]>(Array.from({ length: NUM_TRACKS }, () => initTrack(16 * 4)));
  const patternRef = useRef<Cell[][]>(pattern);
  useEffect(() => {
    patternRef.current = pattern;
  }, [pattern]);

  // Selection for per-event volume
  const [selected, setSelected] = useState<{ track: number; step: number } | null>(null);
  const [liveMsg, setLiveMsg] = useState<string>("");

  // Help modal
  const [helpOpen, setHelpOpen] = useState<boolean>(false);
  const helpCloseRef = useRef<HTMLButtonElement | null>(null);
  useEffect(() => {
    if (helpOpen) {
      setLiveMsg("Help opened");
      setTimeout(() => helpCloseRef.current?.focus(), 0);
    }
  }, [helpOpen]);
  const closeHelp = () => setHelpOpen(false);

  // UI tick
  const [, setUiTick] = useState(0);

  // Tone.js dynamic import
  const toneRef = useRef<ToneType | null>(null);
  const [toneReady, setToneReady] = useState<boolean>(false);
  useEffect(() => {
    let mounted = true;
    (async () => {
      const t = await import("tone");
      if (mounted) {
        toneRef.current = t;
        setToneReady(true);
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  // Audio graph refs
  const masterGainRef = useRef<Gain | null>(null); // Tone.Gain
  const filterRef = useRef<Filter | null>(null); // Tone.Filter (master lowpass)
  const delayRef = useRef<FeedbackDelay | null>(null); // Tone.FeedbackDelay
  const reverbRef = useRef<Reverb | null>(null); // Tone.Reverb
  const trackGainsRef = useRef<Gain[]>([]); // Tone.Gain[]
  const playersRef = useRef<(Player | null)[]>(Array.from({ length: NUM_TRACKS }, () => null)); // Tone.Player | null
  const playerNamesRef = useRef<(string | null)[]>(Array.from({ length: NUM_TRACKS }, () => null));
  const loadedUrlsRef = useRef<(string | null)[]>(Array.from({ length: NUM_TRACKS }, () => null));
  const loopRef = useRef<Loop | null>(null); // Tone.Loop
  // Built-in synth fallback per track
  const synthsRef = useRef<(MembraneSynth | NoiseSynth | MetalSynth | FMSynth | null)[]>(Array.from({ length: NUM_TRACKS }, () => null));
  const synthKindsRef = useRef<string[]>(Array.from({ length: NUM_TRACKS }, () => ""));

  // Playback progress refs
  const totalStepRef = useRef<number>(0); // absolute step counter (next step to play)
  const currentStepIdxRef = useRef<number>(0); // last fired step index
  const lastEventTimeRef = useRef<number>(0); // Transport.seconds of last event
  const nextEventTimeRef = useRef<number>(0); // Transport.seconds of next event

  // Geometry
  const geometry = useMemo(() => {
    const width = CANVAS_SIZE;
    const height = CANVAS_SIZE;
    const cx = width / 2;
    const cy = height / 2;
    const maxRadius = Math.min(width, height) / 2 - 32;
    const trackWidth = maxRadius / (NUM_TRACKS + INNER_SPACER_TRACKS);
    const innerRadius = INNER_SPACER_TRACKS * trackWidth;
    const center = C(cx, cy);
    return { width, height, cx, cy, center, maxRadius, trackWidth, innerRadius };
  }, []);

  // Angles and radii
  const angles = useMemo(() => Array.from({ length: numBeats }, (_, b) => (2 * Math.PI * b) / numBeats - Math.PI / 2), [numBeats]);
  const ringRadii = useMemo(
    () => Array.from({ length: NUM_TRACKS }, (_, t) => geometry.innerRadius + (t + 0.5) * geometry.trackWidth),
    [geometry.innerRadius, geometry.trackWidth]
  );

  // Precompute centers
  const centers = useMemo(() => {
    return ringRadii.map((r) => angles.map((theta) => addC(geometry.center, fromPolar(r, theta))));
  }, [ringRadii, angles, geometry.center]);

  // Build audio graph ONCE (avoid teardown on slider changes)
  useEffect(() => {
    if (!toneReady || !toneRef.current) return;
    const Tone = toneRef.current;

    // Clean before
    try {
      loopRef.current?.stop?.();
      loopRef.current?.dispose?.();
    } catch {}
    playersRef.current.forEach((p) => p?.dispose?.());
    trackGainsRef.current.forEach((g) => g?.dispose?.());
    masterGainRef.current?.dispose?.();
    filterRef.current?.dispose?.();
    delayRef.current?.dispose?.();
    reverbRef.current?.dispose?.();
    synthsRef.current.forEach((s) => s?.dispose?.());

    // Master chain: tracks -> master gain -> lowpass filter -> delay -> reverb -> speakers
    const master = new Tone.Gain(globalVol / 100);
    const filter = new Tone.Filter({ type: "lowpass", frequency: filterToHz(filterCut), rolloff: -24, Q: 0.7 });
    const delay = new Tone.FeedbackDelay({ delayTime: "8n.", feedback: 0.3, wet: delayWet / 100 });
    const reverb = new Tone.Reverb({ decay: 2.4, wet: reverbWet / 100 });
    master.chain(filter, delay, reverb, Tone.Destination);
    masterGainRef.current = master;
    filterRef.current = filter;
    delayRef.current = delay;
    reverbRef.current = reverb;

    trackGainsRef.current = Array.from({ length: NUM_TRACKS }, (_, i) => new Tone.Gain(trackVolumes[i] / 100).connect(master));

    playersRef.current = Array.from({ length: NUM_TRACKS }, () => null);
    playerNamesRef.current = Array.from({ length: NUM_TRACKS }, () => null);
    loadedUrlsRef.current = Array.from({ length: NUM_TRACKS }, () => null);

    // Build default kit (fallback when no sample is loaded)
    synthsRef.current = Array.from({ length: NUM_TRACKS }, () => null);
    synthKindsRef.current = Array.from({ length: NUM_TRACKS }, () => "");
    for (let t = 0; t < NUM_TRACKS; t++) {
      let inst = null;
      let kind = "";
      if (t === 0) {
        inst = new Tone.MembraneSynth({ pitchDecay: 0.02, octaves: 6, envelope: { attack: 0.001, decay: 0.4, sustain: 0, release: 0.1 } });
        kind = "kick";
      } else if (t === 1) {
        inst = new Tone.NoiseSynth({ noise: { type: "white" }, envelope: { attack: 0.001, decay: 0.2, sustain: 0 } });
        kind = "snare";
      } else if (t === 2) {
        // @ts-expect-error frequency type is valid at runtime
        inst = new Tone.MetalSynth({ frequency: 250, envelope: { attack: 0.001, decay: 0.12, release: 0.01 }, harmonicity: 5.1, modulationIndex: 32, resonance: 4000, octaves: 1.5 });
        kind = "hat";
      } else if (t === 3) {
        inst = new Tone.NoiseSynth({ noise: { type: "white" }, envelope: { attack: 0.001, decay: 0.3, sustain: 0 } });
        kind = "clap";
      } else if (t === 4) {
        inst = new Tone.MembraneSynth({ pitchDecay: 0.01, octaves: 5, envelope: { attack: 0.001, decay: 0.25, sustain: 0, release: 0.1 } });
        kind = "tom";
      } else {
        inst = new Tone.FMSynth({ harmonicity: 7, modulationIndex: 10, oscillator: { type: "sine" }, modulation: { type: "square" }, envelope: { attack: 0.001, decay: 0.15, sustain: 0, release: 0.05 }, modulationEnvelope: { attack: 0.001, decay: 0.06, sustain: 0 } });
        kind = "perc";
      }
      inst.volume.value = 0; // use track gain + per-event dB
      inst.connect(trackGainsRef.current[t]);
      synthsRef.current[t] = inst;
      synthKindsRef.current[t] = kind;
    }

    Tone.Transport.bpm.value = bpm;

    return () => {
      try {
        loopRef.current?.stop?.();
        loopRef.current?.dispose?.();
      } catch {}
      playersRef.current.forEach((p) => p?.dispose?.());
      trackGainsRef.current.forEach((g) => g?.dispose?.());
      masterGainRef.current?.dispose?.();
      filterRef.current?.dispose?.();
      delayRef.current?.dispose?.();
      reverbRef.current?.dispose?.();
      synthsRef.current.forEach((s) => s?.dispose?.());
      loadedUrlsRef.current.forEach((url) => {
        if (url) URL.revokeObjectURL(url);
      });
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [toneReady]);

  // Sync volumes / mute / solo / BPM without rebuilding the graph
  useEffect(() => {
    if (!toneRef.current || !masterGainRef.current) return;
    const Tone = toneRef.current;
    masterGainRef.current.gain.rampTo(globalVol / 100, 0.05);
    const anySolo = solos.some(Boolean);
    trackGainsRef.current.forEach((g, i) => {
      const audible = anySolo ? solos[i] : !mutes[i];
      g?.gain?.rampTo?.(audible ? trackVolumes[i] / 100 : 0, 0.05);
    });
    Tone.Transport.bpm.value = bpm;
  }, [globalVol, trackVolumes, bpm, mutes, solos]);

  // Sync master FX without rebuilding the graph
  useEffect(() => {
    filterRef.current?.frequency?.rampTo?.(filterToHz(filterCut), 0.05);
    delayRef.current?.wet?.rampTo?.(delayWet / 100, 0.05);
    reverbRef.current?.wet?.rampTo?.(reverbWet / 100, 0.05);
  }, [filterCut, delayWet, reverbWet]);

  // Resize pattern when beats/cycles change
  useEffect(() => {
    setPattern((prev) =>
      prev.map((track) => {
        const target = numBeats * numCycles;
        if (track.length === target) return track.slice();
        if (track.length > target) return track.slice(0, target);
        return [...track, ...Array.from({ length: target - track.length }, () => makeEmptyCell())];
      })
    );
    setCurrentCycle((c) => clamp(c, 0, Math.max(0, numCycles - 1)));
    setSelected((sel) => {
      if (!sel) return sel;
      const target = numBeats * numCycles;
      if (sel.step >= target) return { ...sel, step: clamp(sel.step, 0, Math.max(0, target - 1)) };
      return sel;
    });
  }, [numBeats, numCycles]);

  // Compute transport state
  const computeTransportState = useCallback(() => {
    const totalSteps = Math.max(1, numBeats * numCycles);

    if (!playing) {
      const idx = clamp(pausedStepRef.current, 0, totalSteps - 1);
      const beatInCycle = idx % numBeats;
      const cycleIdx = Math.floor(idx / numBeats);
      return {
        totalSteps,
        idx,
        frac: 0,
        beatInCycle,
        cycleIdx,
      } as const;
    }

    const Tone = toneRef.current;
    const seconds = Tone ? Tone.Transport.seconds : 0;
    const last = lastEventTimeRef.current;
    const next = nextEventTimeRef.current;
    const frac = next > last ? clamp((seconds - last) / (next - last), 0, 1) : 0;
    const idx = ((currentStepIdxRef.current % totalSteps) + totalSteps) % totalSteps;
    const beatInCycle = idx % numBeats;
    const cycleIdx = Math.floor(idx / numBeats);
    return { totalSteps, idx, frac, beatInCycle, cycleIdx } as const;
  }, [numBeats, numCycles, playing]);

  // Playback start (resumes from paused step)
  const startPlayback = useCallback(async () => {
    if (!toneRef.current || !masterGainRef.current) return;
    const Tone = toneRef.current;

    await Tone.start();

    try {
      loopRef.current?.stop?.();
      loopRef.current?.dispose?.();
    } catch {}

    const stepInterval = halfSpeed ? "8n" : "16n"; // 8n is twice the duration of 16n
    const stepSeconds = Tone.Time(stepInterval).toSeconds();

    const totalSteps = Math.max(1, numBeats * numCycles);
    totalStepRef.current = clamp(pausedStepRef.current, 0, totalSteps - 1);
    currentStepIdxRef.current = totalStepRef.current;

    const now = Tone.Transport.seconds;
    lastEventTimeRef.current = now;
    nextEventTimeRef.current = now + stepSeconds;

    const loop = new Tone.Loop((time: number) => {
      const total = Math.max(1, numBeats * numCycles);
      const step = totalStepRef.current % total;

      // Swing: push every second step late, up to a triplet feel at 100%
      const swingShift = step % 2 === 1 ? (swingRef.current / 100) * stepSeconds * (1 / 3) : 0;
      const tTime = time + swingShift;

      for (let t = 0; t < NUM_TRACKS; t++) {
        const cell = patternRef.current[t]?.[step];
        const player = playersRef.current[t];
        if (cell?.on) {
          const db = eventVolumeToDb(cell.vol);
          let played = false;
          if (player) {
            try {
              player.volume.value = db;
            } catch {}
            try {
              player.start(tTime);
              played = true;
            } catch {}
          }
          if (!played) {
            const synth = synthsRef.current[t];
            const kind = synthKindsRef.current[t];
            if (synth) {
              try { synth.volume.value = db; } catch {}
              try {
                switch (kind) {
                  case "kick":
                    (synth as MembraneSynth).triggerAttackRelease("C1", stepInterval, tTime);
                    break;
                  case "snare":
                    (synth as NoiseSynth).triggerAttackRelease(stepInterval, tTime);
                    break;
                  case "hat":
                    (synth as MetalSynth).triggerAttackRelease("C6", stepInterval, tTime);
                    break;
                  case "clap":
                    (synth as NoiseSynth).triggerAttackRelease(stepInterval, tTime);
                    break;
                  case "tom":
                    (synth as MembraneSynth).triggerAttackRelease("G2", stepInterval, tTime);
                    break;
                  default:
                    (synth as FMSynth).triggerAttackRelease("C5", stepInterval, tTime, 0.8);
                    break;
                }
              } catch {}
            }
          }
        }
      }

      lastEventTimeRef.current = time;
      nextEventTimeRef.current = time + stepSeconds;
      currentStepIdxRef.current = step;
      pausedStepRef.current = step;

      totalStepRef.current = (totalStepRef.current + 1) % total;
      setUiTick((v) => (v + 1) % 1000);
    }, stepInterval);

    loopRef.current = loop;
    loop.start(0);
    Tone.Transport.bpm.value = bpm;
    Tone.Transport.start();
  }, [bpm, numBeats, numCycles, halfSpeed]);

  // Pause playback without resetting counters
  const pausePlayback = useCallback(() => {
    if (!toneRef.current) return;
    const Tone = toneRef.current;
    try {
      loopRef.current?.stop?.();
    } finally {
      Tone.Transport.pause();
      const total = Math.max(1, numBeats * numCycles);
      const idx = ((currentStepIdxRef.current % total) + total) % total;
      pausedStepRef.current = idx;
      totalStepRef.current = idx;
      setCurrentCycle(Math.floor(idx / numBeats));
    }
  }, [numBeats, numCycles]);

  useEffect(() => {
    if (playing) startPlayback();
    else pausePlayback();
  }, [playing, startPlayback, pausePlayback]);

  // Drawing
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { width, height, cx, cy, innerRadius, maxRadius, trackWidth } = geometry;

    ctx.clearRect(0, 0, width, height);

    // Background
    ctx.fillStyle = PALETTE.bg;
    ctx.fillRect(0, 0, width, height);

    // Translate to center
    ctx.save();
    ctx.translate(cx, cy);

    // Solo state for dimming
    const anySolo = solos.some(Boolean);
    const trackAudible = (t: number) => (anySolo ? solos[t] : !mutes[t]);

    // Faint colored lane fill per track
    for (let t = 0; t < NUM_TRACKS; t++) {
      ctx.beginPath();
      ctx.arc(0, 0, ringRadii[t], 0, Math.PI * 2);
      ctx.strokeStyle = `${TRACK_COLORS[t]}${trackAudible(t) ? "24" : "10"}`; // low-alpha lane tint
      ctx.lineWidth = trackWidth * 0.72;
      ctx.stroke();
    }

    // Ring boundaries
    ctx.strokeStyle = PALETTE.grid;
    ctx.lineWidth = 1;
    for (let track = 0; track <= NUM_TRACKS; track++) {
      const r = innerRadius + track * trackWidth;
      ctx.beginPath();
      ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Beat boundaries
    for (let i = 0; i < numBeats; i++) {
      const angle = (2 * Math.PI * (i + 0.5)) / numBeats - Math.PI / 2;
      const z1 = fromPolar(innerRadius, angle);
      const z2 = fromPolar(maxRadius, angle);
      ctx.beginPath();
      ctx.moveTo(z1.re, z1.im);
      ctx.lineTo(z2.re, z2.im);
      ctx.strokeStyle = PALETTE.grid;
      ctx.stroke();
    }

    // Transport state
    const { frac, beatInCycle, cycleIdx } = computeTransportState();
    const playbackCycle = cycleIdx;
    const displayCycle = playbackCycle;

    // Events (colored per track, sized by volume, glow when the playhead hits them)
    for (let t = 0; t < NUM_TRACKS; t++) {
      const audible = trackAudible(t);
      for (let b = 0; b < numBeats; b++) {
        const stepIdx = displayCycle * numBeats + b;
        const cell = patternRef.current[t]?.[stepIdx];
        if (cell?.on) {
          const p = centers[t][b];
          const vol = cell.vol ?? DEFAULT_EVENT_VOLUME;
          const radius = 3.5 + 4.5 * (vol / 100);
          const isHot = playing && audible && b === beatInCycle;
          ctx.beginPath();
          ctx.arc(p.re - cx, p.im - cy, isHot ? radius + 2.5 : radius, 0, Math.PI * 2);
          ctx.fillStyle = TRACK_COLORS[t];
          ctx.globalAlpha = audible ? 1 : 0.3;
          if (isHot) {
            ctx.shadowColor = TRACK_COLORS[t];
            ctx.shadowBlur = 16;
          }
          ctx.fill();
          ctx.shadowBlur = 0;
          ctx.globalAlpha = 1;
        }
        if (selected && selected.track === t && selected.step === stepIdx) {
          const p = centers[t][b];
          ctx.beginPath();
          ctx.arc(p.re - cx, p.im - cy, 10, 0, Math.PI * 2);
          ctx.strokeStyle = "#3B82F6";
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }
    }

    // Smooth playhead
    const playAngle = (2 * Math.PI * (beatInCycle - 0.5 + frac)) / numBeats - Math.PI / 2;
    const p1 = fromPolar(innerRadius, playAngle);
    const p2 = fromPolar(maxRadius, playAngle);
    ctx.beginPath();
    ctx.moveTo(p1.re, p1.im);
    ctx.lineTo(p2.re, p2.im);
    ctx.strokeStyle = PALETTE.playhead;
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.restore();

    // Center play/pause button
    ctx.beginPath();
    ctx.arc(cx, cy, 28, 0, Math.PI * 2);
    ctx.fillStyle = playing ? PALETTE.centerBtnActive : PALETTE.centerBtnIdle;
    ctx.fill();
    ctx.fillStyle = PALETTE.text;
    ctx.font = "bold 18px ui-sans-serif, system-ui, -apple-system";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(playing ? "❚❚" : "▶", cx, cy);
  }, [geometry, numBeats, playing, centers, selected, computeTransportState, ringRadii, mutes, solos]);

  // Redraw strategy
  const lastTsRef = useRef<number>(0);
  useEffect(() => {
    if (!playing) {
      draw();
      return;
    }
    let raf = 0 as number;
    const targetFps = 30;
    const frameMs = 1000 / targetFps;
    const loop = (ts: number) => {
      if (ts - lastTsRef.current >= frameMs) {
        draw();
        lastTsRef.current = ts;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [playing, draw]);

  useEffect(() => {
    if (!playing) draw();
  }, [draw, pattern, selected, currentCycle, numBeats, numCycles, centers, playing, mutes, solos]);

  // Nearest cell helper
  const findNearestCell = (mx: number, my: number) => {
    const click = C(mx, my);
    const { cycleIdx } = computeTransportState();
    const displayCycle = cycleIdx;

    let minDist = Infinity;
    let best: { track: number; beat: number; idx: number; pos: Complex } | null = null;

    for (let t = 0; t < NUM_TRACKS; t++) {
      for (let b = 0; b < numBeats; b++) {
        const pos = centers[t][b];
        const dist = absC(subC(pos, click));
        if (dist < minDist) {
          minDist = dist;
          best = { track: t, beat: b, idx: displayCycle * numBeats + b, pos };
        }
      }
    }
    return { best, minDist };
  };

  // Canvas interaction
  const handleCanvasDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // Center play/pause
    if (Math.hypot(mx - geometry.cx, my - geometry.cy) < 28) {
      setPlaying((p) => !p);
      return;
    }

    const { best, minDist } = findNearestCell(mx, my);
    if (!best || minDist > CLICK_THRESHOLD) return;

    const { track, idx } = best;

    // Toggle
    setPattern((prev) => {
      const next = prev.map((arr) => arr.slice());
      const cell = { ...next[track][idx] } as Cell;
      cell.on = !cell.on;
      if (cell.on && (cell.vol == null || cell.vol < 0)) cell.vol = DEFAULT_EVENT_VOLUME;
      next[track][idx] = cell;
      return next;
    });

    setSelected({ track, step: idx });

    const cell = patternRef.current[track]?.[idx];
    setLiveMsg(`Selected ${TRACK_NAMES[track]}, Beat ${(idx % numBeats) + 1}. ${cell?.on ? "On" : "Off"}. Volume ${cell?.vol ?? DEFAULT_EVENT_VOLUME}%`);
  };

  const handleContextMenu = (e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
  };

  // Keyboard navigation
  const getDisplayCycle = () => {
    const st = computeTransportState();
    return st.cycleIdx;
  };

  const ensureSelectionForDisplay = () => {
    const displayCycle = getDisplayCycle();
    const track = selected?.track ?? 0;
    let beat = 0;
    if (selected) {
      const cyc = Math.floor(selected.step / numBeats);
      beat = cyc === displayCycle ? selected.step % numBeats : 0;
    }
    return { track, beat, displayCycle } as const;
  };

  const announceSelection = (track: number, step: number) => {
    const cell = patternRef.current[track]?.[step];
    setLiveMsg(`Selected ${TRACK_NAMES[track]}, Beat ${(step % numBeats) + 1}. ${cell?.on ? "On" : "Off"}. Volume ${cell?.vol ?? DEFAULT_EVENT_VOLUME}%`);
  };

  // Copy/Paste current cycle helpers
  const copyCurrentCycle = () => {
    const displayCycle = getDisplayCycle();
    const start = displayCycle * numBeats;
    const end = start + numBeats;
    const data: Cell[][] = patternRef.current.map((track) => track.slice(start, end).map((c) => ({ on: !!c?.on, vol: clamp(c?.vol ?? DEFAULT_EVENT_VOLUME, 0, 100) })));
    setClipboard({ beats: numBeats, data });
    setLiveMsg(`Copied cycle ${displayCycle + 1}`);
  };

  const pasteIntoCurrentCycle = () => {
    if (!clipboard) {
      setLiveMsg("Clipboard is empty");
      return;
    }
    const displayCycle = getDisplayCycle();
    const clipBeats = Math.max(1, clipboard.beats);
    setPattern((prev) => {
      const next = prev.map((track, tIdx) => {
        const arr = track.slice();
        for (let b = 0; b < numBeats; b++) {
          const dstIdx = displayCycle * numBeats + b;
          const srcB = Math.floor((b / Math.max(1, numBeats)) * clipBeats) % clipBeats; // simple scale
          const srcCell = clipboard.data[tIdx]?.[srcB] ?? { on: false, vol: DEFAULT_EVENT_VOLUME };
          arr[dstIdx] = { on: !!srcCell.on, vol: clamp(srcCell.vol ?? DEFAULT_EVENT_VOLUME, 0, 100) };
        }
        return arr;
      });
      return next;
    });
    setLiveMsg(`Pasted to cycle ${displayCycle + 1}`);
  };

  const handleCanvasKeyDown = (e: React.KeyboardEvent<HTMLCanvasElement>) => {
    // Copy/Paste shortcuts
    if ((e.ctrlKey || e.metaKey) && (e.key === "c" || e.key === "C")) {
      copyCurrentCycle();
      e.preventDefault();
      return;
    }
    if ((e.ctrlKey || e.metaKey) && (e.key === "v" || e.key === "V")) {
      pasteIntoCurrentCycle();
      e.preventDefault();
      return;
    }

    const { track: startTrack, beat: startBeat, displayCycle } = ensureSelectionForDisplay();
    let track = startTrack;
    let beat = startBeat;

    const commitSelection = () => {
      const step = displayCycle * numBeats + beat;
      setSelected({ track, step });
      announceSelection(track, step);
    };

    switch (e.key) {
      case "ArrowLeft":
        beat = (beat - 1 + numBeats) % numBeats;
        e.preventDefault();
        commitSelection();
        return;
      case "ArrowRight":
        beat = (beat + 1) % numBeats;
        e.preventDefault();
        commitSelection();
        return;
      case "ArrowUp":
        track = clamp(track - 1, 0, NUM_TRACKS - 1);
        e.preventDefault();
        commitSelection();
        return;
      case "ArrowDown":
        track = clamp(track + 1, 0, NUM_TRACKS - 1);
        e.preventDefault();
        commitSelection();
        return;
      case "Home":
        beat = 0;
        e.preventDefault();
        commitSelection();
        return;
      case "End":
        beat = numBeats - 1;
        e.preventDefault();
        commitSelection();
        return;
      case "m":
      case "M": {
        setMutes((prev) => {
          const next = prev.slice();
          next[track] = !next[track];
          setLiveMsg(`${TRACK_NAMES[track]} ${next[track] ? "muted" : "unmuted"}`);
          return next;
        });
        e.preventDefault();
        return;
      }
      case "s":
      case "S": {
        setSolos((prev) => {
          const next = prev.slice();
          next[track] = !next[track];
          setLiveMsg(`${TRACK_NAMES[track]} solo ${next[track] ? "on" : "off"}`);
          return next;
        });
        e.preventDefault();
        return;
      }
      case "[":
      case "BracketLeft": {
        if (!playing) {
          const newCycle = clamp(currentCycle - 1, 0, Math.max(0, numCycles - 1));
          setCurrentCycle(newCycle);
          const step = newCycle * numBeats + beat;
          setSelected({ track, step });
          announceSelection(track, step);
          pausedStepRef.current = step;
        }
        e.preventDefault();
        return;
      }
      case "]":
      case "BracketRight": {
        if (!playing) {
          const newCycle = clamp(currentCycle + 1, 0, Math.max(0, numCycles - 1));
          setCurrentCycle(newCycle);
          const step = newCycle * numBeats + beat;
          setSelected({ track, step });
          announceSelection(track, step);
          pausedStepRef.current = step;
        }
        e.preventDefault();
        return;
      }
      case " ":
      case "Spacebar":
      case "Enter": {
        const step = displayCycle * numBeats + beat;
        setPattern((prev) => {
          const next = prev.map((arr) => arr.slice());
          const cell = { ...next[track][step] } as Cell;
          cell.on = !cell.on;
          if (cell.on && (cell.vol == null || cell.vol < 0)) cell.vol = DEFAULT_EVENT_VOLUME;
          next[track][step] = cell;
          return next;
        });
        setSelected({ track, step });
        announceSelection(track, step);
        e.preventDefault();
        return;
      }
      case "Delete":
      case "Backspace": {
        const step = displayCycle * numBeats + beat;
        setPattern((prev) => {
          const next = prev.map((arr) => arr.slice());
          const cell = { ...next[track][step] } as Cell;
          cell.on = false;
          next[track][step] = cell;
          return next;
        });
        setSelected({ track, step });
        announceSelection(track, step);
        e.preventDefault();
        return;
      }
      case "+":
      case "=": {
        const step = displayCycle * numBeats + beat;
        setPattern((prev) => {
          const next = prev.map((arr) => arr.slice());
          const cell = { ...next[track][step] } as Cell;
          cell.vol = clamp((cell.vol ?? DEFAULT_EVENT_VOLUME) + 5, 0, 100);
          next[track][step] = cell;
          return next;
        });
        setSelected({ track, step });
        announceSelection(track, step);
        e.preventDefault();
        return;
      }
      case "-":
      case "_": {
        const step = displayCycle * numBeats + beat;
        setPattern((prev) => {
          const next = prev.map((arr) => arr.slice());
          const cell = { ...next[track][step] } as Cell;
          cell.vol = clamp((cell.vol ?? DEFAULT_EVENT_VOLUME) - 5, 0, 100);
          next[track][step] = cell;
          return next;
        });
        setSelected({ track, step });
        announceSelection(track, step);
        e.preventDefault();
        return;
      }
      case "Escape": {
        if (helpOpen) {
          closeHelp();
        } else {
          setSelected(null);
          setLiveMsg("Selection cleared");
        }
        e.preventDefault();
        return;
      }
      default:
        return;
    }
  };

  const handleCanvasFocus = () => {
    if (!selected) {
      const displayCycle = getDisplayCycle();
      setSelected({ track: 0, step: displayCycle * numBeats });
      setLiveMsg(`Selected ${TRACK_NAMES[0]}, Beat 1. Off. Volume ${DEFAULT_EVENT_VOLUME}%`);
    }
  };

  // Tone start helper
  const toneStart = async () => {
    if (!toneRef.current) return;
    const Tone = toneRef.current;
    await Tone.start();
  };

  // Tap tempo
  const tapTimesRef = useRef<number[]>([]);
  const tapTempo = () => {
    const now = performance.now();
    const recent = tapTimesRef.current.filter((t) => now - t < 3000);
    recent.push(now);
    tapTimesRef.current = recent.slice(-6);
    const taps = tapTimesRef.current;
    if (taps.length >= 2) {
      const intervals = taps.slice(1).map((t, i) => t - taps[i]);
      const avg = intervals.reduce((a, b) => a + b, 0) / intervals.length;
      const next = clamp(Math.round(60000 / avg), 30, 240);
      setBpm(next);
      setLiveMsg(`Tempo set to ${next} BPM`);
    } else {
      setLiveMsg("Keep tapping to set the tempo");
    }
  };

  // Mute/solo toggles
  const toggleMute = (i: number) => {
    setMutes((prev) => {
      const next = prev.slice();
      next[i] = !next[i];
      return next;
    });
  };
  const toggleSolo = (i: number) => {
    setSolos((prev) => {
      const next = prev.slice();
      next[i] = !next[i];
      return next;
    });
  };

  // Sample loading
  const handleFileChange = async (trackIndex: number, file: File | null) => {
    if (!toneRef.current || !file) return;
    const Tone = toneRef.current;

    const url = URL.createObjectURL(file);

    playersRef.current[trackIndex]?.dispose?.();
    if (loadedUrlsRef.current[trackIndex]) URL.revokeObjectURL(loadedUrlsRef.current[trackIndex]!);

    const gain = trackGainsRef.current[trackIndex];
    const player = new Tone.Player({ url, autostart: false, fadeIn: 0.003, fadeOut: 0.03 }).connect(gain);
    playersRef.current[trackIndex] = player;
    playerNamesRef.current[trackIndex] = file.name;
    loadedUrlsRef.current[trackIndex] = url;
    setUiTick((t) => t + 1);
  };

  const previewTrack = async (trackIndex: number) => {
    if (!toneRef.current) return;
    const Tone = toneRef.current;
    await Tone.start();
    const player = playersRef.current[trackIndex];
    if (player) {
      try { player.volume.value = 0; } catch {}
      try { player.start(); } catch {}
      return;
    }
    const synth = synthsRef.current[trackIndex];
    const kind = synthKindsRef.current[trackIndex];
    if (!synth) return;
    try { synth.volume.value = 0; } catch {}
    try {
      switch (kind) {
        case "kick":
          (synth as MembraneSynth).triggerAttackRelease("C1", "16n");
          break;
        case "snare":
          (synth as NoiseSynth).triggerAttackRelease("16n");
          break;
        case "hat":
          (synth as MetalSynth).triggerAttackRelease("C6", "16n");
          break;
        case "clap":
          (synth as NoiseSynth).triggerAttackRelease("16n");
          break;
        case "tom":
          (synth as MembraneSynth).triggerAttackRelease("G2", "16n");
          break;
        default:
          (synth as FMSynth).triggerAttackRelease("C5", "16n");
      }
    } catch {}
  };

  const changeTrackVolume = (i: number, v: number) => {
    setTrackVolumes((arr) => {
      const copy = arr.slice();
      copy[i] = v;
      return copy;
    });
  };

  // Harmonic, evenly spaced randomization with key-point drops and wider volume variance
  const randomizePattern = () => {
    void toneStart(); // ensure AudioContext is resumed on user gesture
    const total = numBeats * numCycles;
    const offsets = harmonicOffsets(numBeats);
    const rng = Math.random;

    const next: Cell[][] = Array.from({ length: NUM_TRACKS }, () => Array.from({ length: total }, () => ({ on: false, vol: DEFAULT_EVENT_VOLUME })));

    for (let t = 0; t < NUM_TRACKS; t++) {
      const pulses = pickPulsesForTrack(t, numBeats, rng);
      const baseIndices = euclideanIndices(numBeats, pulses);

      // Stagger tracks to reduce collisions and favor harmonic starts
      const harmonicOff = offsets.length ? offsets[Math.floor(rng() * offsets.length)] : 0;
      const trackSpread = Math.round((t * numBeats) / NUM_TRACKS) % numBeats;
      const rotation = (harmonicOff + trackSpread) % numBeats;
      const indices = rotateIndices(baseIndices, rotation, numBeats);

      for (let c = 0; c < numCycles; c++) {
        const cycleBase = c * numBeats;
        // Slight per-cycle rotation to distribute over long loops
        const cycleOff = offsets.length ? offsets[Math.floor(rng() * Math.min(offsets.length, 3))] : 0;
        const cycIndices = rotateIndices(indices, cycleOff, numBeats);

        // Drop some events at key harmonic points for syncopation
        let carved = cycIndices.filter((b) => rng() > keyHarmonicDropProb(t, b, numBeats));
        if (carved.length === 0) {
          // Ensure at least one pulse remains
          carved = [cycIndices[Math.floor(rng() * cycIndices.length)]];
        }

        for (const b of carved) {
          const step = cycleBase + b;
          const strength = harmonicStrength(b, numBeats); // 0..1
          // Heavy‑tailed random energy in [0,1]
          const n = normalRand(rng); // mean 0, std 1
          const wide = clamp(0.5 + 0.5 * n * 0.9, 0, 1);
          const energy = clamp(0.6 * strength + 0.4 * wide, 0, 1);
          const trackBoost = t === 0 ? 6 : t === 1 ? 2 : 0; // keep kick/snare a touch stronger
          const v = clamp(Math.round(40 + 60 * energy + trackBoost), 30, 100);
          next[t][step] = { on: true, vol: v };
        }
      }
    }

    setPattern(next);
    setSelected(null);
    setLiveMsg("Randomized with harmonic spacing, syncopation, and wider volume variance");
  };

  const clearPattern = () => {
    setPattern((prev) => prev.map((t) => t.map(() => ({ on: false, vol: DEFAULT_EVENT_VOLUME }))));
    setSelected(null);
    setLiveMsg("Pattern cleared");
  };

  // Download helper with showSaveFilePicker support
  const saveJsonFile = async (obj: unknown, filename: string) => {
    const json = JSON.stringify(obj, null, 2);
    try {
      if ('showSaveFilePicker' in window) {
        const handle = await (window as unknown as { showSaveFilePicker: (options: unknown) => Promise<unknown> }).showSaveFilePicker({
          suggestedName: filename,
          types: [
            {
              description: "JSON Files",
              accept: { "application/json": [".json"] },
            },
          ],
        }) as FileSystemFileHandle;
        const writable = await handle.createWritable();
        await writable.write(new Blob([json], { type: "application/json" }));
        await writable.close();
        return; // success with native save dialog
      }
    } catch (err) {
      console.error("showSaveFilePicker failed, falling back", err);
    }

    // Fallback: anchor-based download (opens browser's save dialog)
    try {
      const blob = new Blob([json], { type: "application/json" });
      const nav = typeof window !== "undefined" ? window.navigator : undefined;
      if (nav && 'msSaveOrOpenBlob' in nav) {
        (nav as unknown as { msSaveOrOpenBlob: (blob: Blob, filename: string) => void }).msSaveOrOpenBlob(blob, filename);
        return;
      }
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.rel = "noopener";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      // Important: delay revocation to avoid canceling the download in some browsers
      setTimeout(() => URL.revokeObjectURL(url), 1500);
    } catch (err) {
      console.error("Export fallback failed", err);
    }
  };

  // JSON export/import
  const exportJSON = () => {
    // Ensure we export the full contents sized to beats * cycles and sanitize cells
    const target = Math.max(1, numBeats * numCycles);
    const safePattern: Cell[][] = Array.from({ length: NUM_TRACKS }, (_, t) =>
      Array.from({ length: target }, (_, i) => {
        const src = patternRef.current[t]?.[i];
        const on = !!src?.on;
        const vol = clamp(typeof src?.vol === "number" ? Math.round(src.vol) : DEFAULT_EVENT_VOLUME, 0, 100);
        return { on, vol };
      })
    );

    const data = {
      version: 3,
      bpm,
      numBeats,
      numCycles,
      halfSpeed,
      swing,
      trackVolumes: trackVolumes.slice(0, NUM_TRACKS).map((v) => clamp(Math.round(v), 0, 100)),
      mutes: mutes.slice(0, NUM_TRACKS).map(Boolean),
      solos: solos.slice(0, NUM_TRACKS).map(Boolean),
      fx: {
        filterCut: clamp(Math.round(filterCut), 0, 100),
        reverbWet: clamp(Math.round(reverbWet), 0, 100),
        delayWet: clamp(Math.round(delayWet), 0, 100),
      },
      pattern: safePattern,
      samples: playerNamesRef.current.map((n) => (typeof n === "string" ? n : null)),
    } as const;

    void saveJsonFile(data, "radial-sequencer.json");
  };

  interface ParsedCell { on?: unknown; vol?: unknown; vel?: unknown; }

  const importJSON = (file: File | null) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(String(reader.result)) as { pattern?: unknown; fx?: unknown; [key: string]: unknown; };
        setBpm(parsed.bpm as number ?? bpm);
        setNumBeats(parsed.numBeats as number ?? numBeats);
        setNumCycles(parsed.numCycles as number ?? numCycles);
        setCurrentCycle(0);
        setTrackVolumes(Array.isArray(parsed.trackVolumes) ? parsed.trackVolumes.slice(0, NUM_TRACKS) as number[] : trackVolumes);
        if (typeof parsed.halfSpeed === "boolean") setHalfSpeed(parsed.halfSpeed);

        // v3 fields (older files simply skip these)
        if (typeof parsed.swing === "number") setSwing(clamp(Math.round(parsed.swing), 0, 100));
        if (Array.isArray(parsed.mutes)) setMutes(Array.from({ length: NUM_TRACKS }, (_, i) => !!(parsed.mutes as unknown[])[i]));
        if (Array.isArray(parsed.solos)) setSolos(Array.from({ length: NUM_TRACKS }, (_, i) => !!(parsed.solos as unknown[])[i]));
        const fx = parsed.fx as { filterCut?: unknown; reverbWet?: unknown; delayWet?: unknown } | undefined;
        if (fx && typeof fx === "object") {
          if (typeof fx.filterCut === "number") setFilterCut(clamp(Math.round(fx.filterCut), 0, 100));
          if (typeof fx.reverbWet === "number") setReverbWet(clamp(Math.round(fx.reverbWet), 0, 100));
          if (typeof fx.delayWet === "number") setDelayWet(clamp(Math.round(fx.delayWet), 0, 100));
        }

        if (Array.isArray(parsed.pattern) && parsed.pattern.length === NUM_TRACKS) {
          const converted: Cell[][] = parsed.pattern.map((track: unknown) => {
            if (!Array.isArray(track)) return [];
            return track.map((cell: unknown) => {
              const typedCell = cell as ParsedCell;
              const on = !!typedCell.on;
              let vol: number;
              if (typeof typedCell.vol === "number") vol = clamp(Math.round(typedCell.vol), 0, 100);
              else if (typeof typedCell.vel === "number") vol = clamp(Math.round(typedCell.vel / 127 * 100), 0, 100);
              else vol = DEFAULT_EVENT_VOLUME;
              return { on, vol };
            });
          });
          setPattern(converted);
        }
      } catch (err) {
        console.error("Import failed", err);
      }
    };
    reader.readAsText(file);
  };

  // Cycle controls
  const decViewCycle = () => {
    if (playing) return;
    const beat = selected ? selected.step % numBeats : 0;
    const newCycle = clamp(currentCycle - 1, 0, Math.max(0, numCycles - 1));
    const step = newCycle * numBeats + beat;
    setCurrentCycle(newCycle);
    setSelected((prev) => ({ track: prev?.track ?? 0, step }));
    pausedStepRef.current = step;
  };
  const incViewCycle = () => {
    if (playing) return;
    const beat = selected ? selected.step % numBeats : 0;
    const newCycle = clamp(currentCycle + 1, 0, Math.max(0, numCycles - 1));
    const step = newCycle * numBeats + beat;
    setCurrentCycle(newCycle);
    setSelected((prev) => ({ track: prev?.track ?? 0, step }));
    pausedStepRef.current = step;
  };

  // Transport state for UI
  const transportState = computeTransportState();
  const playbackCycle = transportState.cycleIdx;

  // Selected event info
  const selectedInfo = (() => {
    if (!selected) return null;
    const { track, step } = selected;
    const cyc = Math.floor(step / numBeats);
    const beat = step % numBeats;
    const cell = pattern[track]?.[step];
    return { track, cyc, beat, cell };
  })();

  // Debounced selection clearing when selected cycle is not visible
  const clearSelDebounceRef = useRef<number | null>(null);
  useEffect(() => {
    const displayCycle = playbackCycle;
    if (selectedInfo && selectedInfo.cyc !== displayCycle) {
      if (typeof window !== "undefined") {
        if (clearSelDebounceRef.current != null) {
          clearTimeout(clearSelDebounceRef.current);
        }
        clearSelDebounceRef.current = window.setTimeout(() => {
          setSelected(null);
          setLiveMsg("Selection cleared due to cycle change");
          clearSelDebounceRef.current = null;
        }, 120);
      }
    } else {
      if (clearSelDebounceRef.current != null) {
        clearTimeout(clearSelDebounceRef.current);
        clearSelDebounceRef.current = null;
      }
    }
  }, [selectedInfo, playbackCycle]);
  useEffect(() => {
    return () => {
      if (clearSelDebounceRef.current != null) {
        clearTimeout(clearSelDebounceRef.current);
      }
    };
  }, []);

  // Popup position
  const popupPos = useMemo(() => {
    if (!selectedInfo) return null;
    const displayCycle = playbackCycle;
    if (selectedInfo.cyc !== displayCycle) return null;
    const { track, beat } = selectedInfo;
    const p = centers[track]?.[beat];
    if (!p) return null;
    return { x: p.re, y: p.im };
  }, [selectedInfo, centers, playbackCycle]);

  // Step interval helper
  const stepInfo = useMemo(() => {
    const quarterMs = 60000 / bpm;
    const stepDiv = halfSpeed ? 2 : 4; // 8n vs 16n
    const stepMs = Math.round(quarterMs / stepDiv);
    const label = halfSpeed ? "8n" : "16n";
    return { label, stepMs };
  }, [bpm, halfSpeed]);

  // Inline SVG logo as data URI
  const logoSvg = encodeURIComponent(
    `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64' fill='none'>` +
      `<circle cx='32' cy='32' r='30' stroke='#3B82F6' stroke-width='4' fill='none'/>` +
      `<circle cx='32' cy='32' r='18' stroke='#60A5FA' stroke-width='4' fill='none'/>` +
      `<circle cx='32' cy='32' r='6' fill='#3B82F6'/>` +
      `<line x1='32' y1='2' x2='32' y2='12' stroke='#EF4444' stroke-width='4'/>` +
    `</svg>`
  );

  const anySolo = solos.some(Boolean);

  return (
    <>
      <Head>
        <title>MuSpin - Radial Music Sequencer</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="theme-color" content="#111827" />
        <meta name="description" content="MuSpin is a radial step sequencer for playful beat making." />
      </Head>

      <div className="min-h-screen bg-gray-900 text-gray-100 p-4 sm:p-6">
        <header className="max-w-7xl mx-auto mb-4 sm:mb-6 flex items-center gap-3">
          <Image src={`data:image/svg+xml,${logoSvg}`} alt="MuSpin logo" width={32} height={32} className="rounded" />
          <div className="flex items-baseline gap-2">
            <h1 className="text-lg font-semibold text-gray-100">MuSpin</h1>
            <span className="text-xs text-gray-400">Radial Music Sequencer</span>
          </div>
        </header>

        <main className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6">
            <div className="lg:col-span-2 bg-gray-800 rounded-xl p-3 sm:p-4">
              <div className="relative flex flex-col items-center justify-center gap-2">
                <div id="canvas-instructions" className="sr-only">
                  Radial step sequencer canvas. Use arrow keys to move between tracks and beats. Press Space or Enter to toggle events. Press M to mute or S to solo the selected track. Use left and right bracket keys to change cycle when paused. Use plus or minus to adjust selected event volume without turning it on. Press Escape to clear selection.
                </div>
                <div aria-live="polite" className="sr-only">{liveMsg}</div>

                <canvas
                  ref={canvasRef}
                  id={canvasId}
                  width={CANVAS_SIZE}
                  height={CANVAS_SIZE}
                  onMouseDown={handleCanvasDown}
                  onContextMenu={handleContextMenu}
                  onKeyDown={handleCanvasKeyDown}
                  onFocus={handleCanvasFocus}
                  tabIndex={0}
                  role="application"
                  aria-keyshortcuts="ArrowLeft,ArrowRight,ArrowUp,ArrowDown,Enter,Space,Delete,Backspace,BracketLeft,BracketRight,Plus,Minus,M,S,Escape,Control+C,Control+V"
                  aria-label="Radial step sequencer canvas"
                  aria-describedby="canvas-instructions"
                  className="rounded-xl border border-gray-700 shadow-inner focus:outline-none focus:ring-2 focus:ring-blue-500"
                />

                {popupPos && selectedInfo && (
                  <div
                    className="absolute bg-gray-900 border border-gray-700 rounded-lg p-3 shadow-lg text-xs transform -translate-x-1/2 -translate-y-full"
                    style={{ left: popupPos.x, top: popupPos.y }}
                    role="dialog"
                    aria-label="Event volume"
                  >
                    <div className="flex items-center justify-between mb-2 gap-2">
                      <span className="text-gray-300">
                        <span className="inline-block w-2 h-2 rounded-full mr-1" style={{ backgroundColor: TRACK_COLORS[selectedInfo.track] }} aria-hidden="true" />
                        {TRACK_NAMES[selectedInfo.track]} volume
                      </span>
                      <button className={btnSecondary} onClick={() => setSelected(null)}>
                        Close
                      </button>
                    </div>
                    <div className="flex items-center gap-2">
                      <input
                        type="range"
                        min={0}
                        max={100}
                        step={EVENT_VOLUME_STEP}
                        value={selectedInfo.cell?.vol ?? DEFAULT_EVENT_VOLUME}
                        onChange={(e) => {
                          const v = clamp(parseInt(e.target.value), 0, 100);
                          if (!selected) return;
                          setPattern((prev) => {
                            const next = prev.map((arr) => arr.slice());
                            const c = { ...next[selected.track][selected.step] } as Cell;
                            c.vol = v;
                            next[selected.track][selected.step] = c;
                            return next;
                          });
                          setLiveMsg(`Volume ${v} percent`);
                        }}
                        aria-label="Selected event volume"
                        aria-valuemin={0}
                        aria-valuemax={100}
                        aria-valuenow={selectedInfo.cell?.vol ?? DEFAULT_EVENT_VOLUME}
                        aria-orientation="horizontal"
                        className="flex-1"
                      />
                      <span className="w-12 text-right">{selectedInfo.cell?.vol ?? DEFAULT_EVENT_VOLUME}%</span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            <aside className="bg-gray-800 rounded-xl p-3 sm:p-4">
              <div className="flex justify-end mb-2">
                <button
                  className={btnSecondary}
                  onClick={() => setHelpOpen(true)}
                  aria-haspopup="dialog"
                  aria-controls="help-dialog"
                  aria-label="Open help"
                >
                  Help
                </button>
              </div>

              {/* Controls grid with equal-height columns */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 items-stretch">
                {/* Left column: cycles + action buttons */}
                <div className="flex flex-col gap-2 h-full min-h-0">
                  <CyclesBar
                    playing={playing}
                    currentDisplayCycle={playbackCycle + 1}
                    totalCycles={numCycles}
                    onPrevCycle={decViewCycle}
                    onNextCycle={incViewCycle}
                    onSetTotal={(n) => setNumCycles(n)}
                    onCopyCycle={copyCurrentCycle}
                    onPasteCycle={pasteIntoCurrentCycle}
                    canPaste={!!clipboard}
                    canvasId={canvasId}
                  />

                  <div className="grid grid-cols-2 gap-2 content-start">
                    <button
                      onClick={() => setPlaying((p) => !p)}
                      className={`${btnFillPrimary} col-span-2`}
                      onMouseDown={toneStart}
                      aria-pressed={playing}
                      aria-label={playing ? "Pause" : "Play"}
                      aria-controls={canvasId}
                    >
                      {playing ? "Pause" : "Play"}
                    </button>
                    <button
                      className={btnFillToggle(halfSpeed)}
                      onClick={() => setHalfSpeed((v) => !v)}
                      aria-pressed={halfSpeed}
                      aria-label="Toggle half speed"
                      aria-controls={canvasId}
                    >
                      Half
                    </button>
                    <button
                      onClick={tapTempo}
                      className={btnFillSecondary}
                      aria-label="Tap tempo"
                    >
                      Tap
                    </button>
                    <button onClick={randomizePattern} className={btnFillSecondary} aria-label="Randomize pattern" aria-controls={canvasId}>
                      Randomize
                    </button>
                    <button onClick={clearPattern} className={btnFillSecondary} aria-label="Clear pattern" aria-controls={canvasId}>
                      Clear
                    </button>
                    <button onClick={exportJSON} className={btnFillSecondary} aria-label="Export pattern">
                      Export
                    </button>
                    <label className={`${btnFillSecondary} cursor-pointer`} aria-label="Import pattern">
                      Import
                      <input type="file" accept="application/json" className="hidden" onChange={(e) => importJSON(e.target.files?.[0] ?? null)} />
                    </label>
                  </div>
                </div>

                {/* Right column: sliders on top, Donate pinned to bottom to match left column bottom */}
                <div className="flex flex-col gap-2 h-full min-h-0 justify-between">
                  <div className="flex flex-col gap-2">
                    <SlimSlider id="bpm" label="BPM" min={30} max={240} step={1} value={bpm} onChange={setBpm} />
                    <SlimSlider id="beats" label="Beats" min={6} max={32} step={1} value={numBeats} onChange={(v) => setNumBeats(Math.round(v))} />
                    <SlimSlider id="swing" label="Swing" min={0} max={100} step={1} value={swing} onChange={setSwing} display={(v) => `${v}%`} />
                    <SlimSlider id="master-vol" label="Master Vol" min={0} max={100} step={1} value={globalVol} onChange={setGlobalVol} />
                  </div>

                  <div>
                    <a
                      href="https://commerce.coinbase.com/checkout/fc87e8ae-90ef-481e-85bb-3f51d2ae742e"
                      target="_blank"
                      rel="noopener noreferrer"
                      className={btnFillSecondary}
                      role="button"
                    >
                      Donate
                    </a>
                  </div>
                </div>
              </div>

              {/* Master FX */}
              <div className="grid grid-cols-3 gap-2 mt-2">
                <SlimSlider id="fx-filter" label="Filter" min={0} max={100} step={1} value={filterCut} onChange={setFilterCut} display={(v) => (v >= 100 ? "Open" : `${(filterToHz(v) / 1000).toFixed(1)}k`)} />
                <SlimSlider id="fx-reverb" label="Reverb" min={0} max={100} step={1} value={reverbWet} onChange={setReverbWet} display={(v) => `${v}%`} />
                <SlimSlider id="fx-delay" label="Delay" min={0} max={100} step={1} value={delayWet} onChange={setDelayWet} display={(v) => `${v}%`} />
              </div>

              <div className="text-xs text-gray-400 mt-3">
                Step interval: {stepInfo.label}; {stepInfo.stepMs} ms at {bpm} BPM. At very high tempos, perceived halving may vary—trust your ear.
              </div>

              {/* Tracks: compact cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-3">
                {Array.from({ length: NUM_TRACKS }).map((_, i) => {
                  const audible = anySolo ? solos[i] : !mutes[i];
                  return (
                    <div key={i} className={`rounded-lg bg-gray-900 border border-gray-700 p-2 ${audible ? "" : "opacity-60"}`}>
                      <div className="mb-1">
                        <div className="flex items-center justify-between">
                          <span className="flex items-center gap-1.5 font-medium text-sm">
                            <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ backgroundColor: TRACK_COLORS[i] }} aria-hidden="true" />
                            {TRACK_NAMES[i]}
                          </span>
                          <span className="flex gap-1">
                            <button
                              className={btnTiny(mutes[i], "bg-red-600 text-white")}
                              onClick={() => toggleMute(i)}
                              aria-pressed={mutes[i]}
                              aria-label={`${mutes[i] ? "Unmute" : "Mute"} ${TRACK_NAMES[i]}`}
                            >
                              M
                            </button>
                            <button
                              className={btnTiny(solos[i], "bg-yellow-500 text-gray-900")}
                              onClick={() => toggleSolo(i)}
                              aria-pressed={solos[i]}
                              aria-label={`Solo ${TRACK_NAMES[i]} ${solos[i] ? "off" : "on"}`}
                            >
                              S
                            </button>
                          </span>
                        </div>
                        <div className="text-xs text-gray-400 truncate mt-1">{playerNamesRef.current[i] ?? "Default kit"}</div>
                      </div>

                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-xs w-12">Vol</span>
                        <input
                          type="range"
                          min={0}
                          max={100}
                          value={trackVolumes[i]}
                          onChange={(e) => changeTrackVolume(i, parseInt(e.target.value))}
                          aria-label={`${TRACK_NAMES[i]} volume`}
                          aria-valuemin={0}
                          aria-valuemax={100}
                          aria-valuenow={trackVolumes[i]}
                          aria-orientation="horizontal"
                          className="flex-1"
                        />
                        <span className="text-xs w-12 text-right">{trackVolumes[i]}%</span>
                      </div>

                      <div className="grid grid-cols-2 gap-2">
                        <label className={`${btnSecondary} cursor-pointer`}>
                          Load
                          <input
                            type="file"
                            accept="audio/*,.wav,.mp3,.ogg"
                            className="hidden"
                            onChange={(e) => handleFileChange(i, e.target.files?.[0] ?? null)}
                          />
                        </label>
                        <button
                          onClick={() => previewTrack(i)}
                          className={btnPrimary}
                          aria-label={`Preview ${TRACK_NAMES[i]}`}
                        >
                          Preview
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </aside>
          </div>
        </main>

        {/* Help Modal */}
        {helpOpen && (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center"
            role="dialog"
            id="help-dialog"
            aria-modal="true"
            aria-labelledby="help-title"
            aria-describedby="help-desc"
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                e.stopPropagation();
                closeHelp();
              }
            }}
          >
            <div className="absolute inset-0 bg-black bg-opacity-50" aria-hidden="true" onClick={closeHelp} />
            <div className="relative bg-gray-900 border border-gray-700 rounded-xl p-4 w-full max-w-lg mx-4 shadow-lg">
              <div className="flex items-center justify-between mb-2">
                <h2 id="help-title" className="text-sm font-semibold text-gray-100">Keyboard Shortcuts</h2>
                <button ref={helpCloseRef} className={btnSecondary} onClick={closeHelp} aria-label="Close help">Close</button>
              </div>
              <div id="help-desc" className="text-sm text-gray-300 space-y-2">
                <p>The canvas captures keyboard input (application role). Use these shortcuts:</p>
                <ul className="list-disc pl-4 space-y-1">
                  <li><span className="font-medium">Arrow keys:</span> Move selection between beats and tracks</li>
                  <li><span className="font-medium">Enter/Space:</span> Toggle event on/off</li>
                  <li><span className="font-medium">Delete/Backspace:</span> Clear event</li>
                  <li><span className="font-medium">M / S:</span> Mute or solo the selected track</li>
                  <li><span className="font-medium">Plus / Minus:</span> Adjust volume without turning the event on</li>
                  <li><span className="font-medium">[ / ]:</span> Change visible cycle when paused</li>
                  <li><span className="font-medium">Ctrl/Cmd + C:</span> Copy current cycle</li>
                  <li><span className="font-medium">Ctrl/Cmd + V:</span> Paste to current cycle</li>
                  <li><span className="font-medium">Escape:</span> Clear selection or close this dialog</li>
                </ul>
                <p>Swing pushes every second step late, up to a triplet feel at 100%. The Filter, Reverb, and Delay sliders shape the whole mix.</p>
                <p className="text-gray-400">If you haven&apos;t loaded samples, the default drum kit is used so Randomize is audible.</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
