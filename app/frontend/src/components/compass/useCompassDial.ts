import { useCallback, useEffect, useRef, type MutableRefObject } from "react";
import { shortestTurn } from "../../lib/compass";
import { prefersReducedMotion } from "./useAnimatedNumber";

export interface Dial {
  /** Bezel turn in degrees, unwrapped (it can pass 360) so a turn never jumps back the long way round. */
  turn: MutableRefObject<number>;
  /** Follow a drag directly. The caller decides when the drag has come to rest. */
  setTurn: (deg: number) => void;
  /** Ease to a turn along the shortest path; `onRest` fires when it arrives. */
  turnTo: (deg: number, ms?: number) => void;
}

/**
 * The bezel runs in a requestAnimationFrame loop outside React: each frame calls `apply(turn)`, which writes the
 * moving SVG attributes directly. Pointer moves only record the target, so a burst of them costs one DOM update per
 * frame and no React render. The loop stops once the bezel is still.
 */
export function useDial(apply: (turn: number) => void, onRest?: (turn: number) => void): Dial {
  const turn = useRef(0);
  const applyRef = useRef(apply);
  applyRef.current = apply;
  const restRef = useRef(onRest);
  restRef.current = onRest;
  const state = useRef({
    frame: 0,
    anim: null as null | { from: number; to: number; start: number; ms: number },
    restAfter: false, // a turnTo is in flight: report rest when the loop stops
  });

  const tick = useCallback((now: number) => {
    const s = state.current;
    if (s.anim) {
      const p = Math.min(1, (now - s.anim.start) / s.anim.ms);
      turn.current = s.anim.from + (s.anim.to - s.anim.from) * (1 - Math.pow(1 - p, 3));
      if (p >= 1) s.anim = null;
    }
    applyRef.current(turn.current);
    s.frame = s.anim ? requestAnimationFrame(tick) : 0;
    if (!s.frame && s.restAfter) {
      s.restAfter = false;
      restRef.current?.(turn.current);
    }
  }, []);

  const wake = useCallback(() => {
    const s = state.current;
    if (s.frame) return;
    s.frame = requestAnimationFrame(tick);
  }, [tick]);

  const setTurn = useCallback((deg: number) => {
    state.current.anim = null;
    state.current.restAfter = false;
    turn.current = deg;
    wake();
  }, [wake]);

  const turnTo = useCallback((deg: number, ms = 650) => {
    const s = state.current;
    const to = turn.current + shortestTurn(turn.current, deg);
    if (prefersReducedMotion()) {
      s.anim = null;
      turn.current = to;
    } else {
      s.anim = { from: turn.current, to, start: performance.now(), ms };
    }
    s.restAfter = true;
    wake();
  }, [wake]);

  useEffect(() => () => cancelAnimationFrame(state.current.frame), []);
  return { turn, setTurn, turnTo };
}
