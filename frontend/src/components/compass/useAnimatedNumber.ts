import { useEffect, useRef, useState } from "react";

const prefersReducedMotion = () =>
  typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;

/** Eases toward `target` over `ms`, so SVG attributes (like ring radii) can animate without CSS support for them. */
export function useAnimatedNumber(target: number, ms = 450): number {
  const [value, setValue] = useState(target);
  const current = useRef(target);

  useEffect(() => {
    if (prefersReducedMotion()) {
      current.current = target;
      setValue(target);
      return;
    }
    const from = current.current;
    const start = performance.now();
    let frame = 0;
    const step = (now: number) => {
      const p = Math.min(1, (now - start) / ms);
      const eased = 1 - Math.pow(1 - p, 3);
      current.current = from + (target - from) * eased;
      setValue(current.current);
      if (p < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [target, ms]);

  return value;
}

export { prefersReducedMotion };
