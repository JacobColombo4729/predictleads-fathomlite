import { useEffect, useRef } from "react";
import fishUrl from "../assets/anglerfish.svg";
import { routeHref } from "../lib/router";

const HEADLINE = "Job searching can feel like navigating deep water.";
const SUB = "Let us provide the light in the depths of that sea so you can discover what's next.";

const RADII = [230, 175, 128, 88];
const OPACITY = [0.045, 0.07, 0.11, 0.17];

/** Each word is its own element, so the pool of light can brighten the words it passes over. */
const words = (text: string) => text.split(" ").flatMap((w, i) => [i > 0 ? " " : null, <span key={`${w}-${i}`} className="landing__w">{w}</span>]);

/**
 * The first page a new visitor sees. A pool of light follows the pointer and brightens the words beneath it, like the
 * anglerfish's lure. It rests over the text when the pointer is away, and is switched off for people who prefer less
 * motion and on touch screens, where every word stays fully visible.
 */
export default function LandingPage() {
  const hero = useRef<HTMLElement>(null);
  const lights = useRef<HTMLDivElement>(null);
  const sub = useRef<HTMLParagraphElement>(null);

  useEffect(() => {
    const root = hero.current, layer = lights.current, anchor = sub.current;
    if (!root || !layer || !anchor) return;
    const spans = Array.from(root.querySelectorAll<HTMLElement>(".landing__w"));
    const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches || window.matchMedia("(hover: none)").matches;
    const rings = RADII.map((r, i) => {
      const ring = document.createElement("div");
      ring.className = "landing__ring";
      ring.style.cssText = `width:${r * 2}px;height:${r * 2}px;opacity:${OPACITY[i]}`;
      layer.appendChild(ring);
      return ring;
    });
    if (calm) {
      spans.forEach((s) => { s.style.opacity = "1"; });
      layer.style.display = "none";
      return () => rings.forEach((r) => r.remove());
    }
    const lightAt = (x: number, y: number) => {
      rings.forEach((ring, i) => { ring.style.left = `${x - RADII[i]}px`; ring.style.top = `${y - RADII[i]}px`; });
      const box = root.getBoundingClientRect();
      spans.forEach((s) => {
        const r = s.getBoundingClientRect();
        const dx = r.left + r.width / 2 - box.left - x, dy = r.top + r.height / 2 - box.top - y;
        s.style.opacity = Math.max(0.28, Math.min(1, 1.25 - Math.sqrt(dx * dx + dy * dy) / 300)).toFixed(2);
      });
    };
    const rest = () => {
      const box = root.getBoundingClientRect(), r = anchor.getBoundingClientRect();
      lightAt(r.left - box.left + r.width / 2, r.top - box.top + r.height * 0.1);
    };
    const move = (e: PointerEvent) => { const box = root.getBoundingClientRect(); lightAt(e.clientX - box.left, e.clientY - box.top); };
    rest();
    root.addEventListener("pointermove", move);
    root.addEventListener("pointerleave", rest);
    window.addEventListener("resize", rest);
    return () => {
      root.removeEventListener("pointermove", move);
      root.removeEventListener("pointerleave", rest);
      window.removeEventListener("resize", rest);
      rings.forEach((r) => r.remove());
    };
  }, []);

  return (
    <main className="landing" ref={hero} aria-label="Bright Depth">
      <div className="landing__lights" ref={lights} aria-hidden="true" />
      <div className="landing__content">
        <p className="landing__brand">Bright Depth</p>
        <img className="landing__fish" src={fishUrl} alt="An anglerfish with a glowing lure" />
        <h1>{words(HEADLINE)}</h1>
        <p className="landing__sub" ref={sub}>{words(SUB)}</p>
        <a className="landing__cta" href={routeHref("profile")}>Create your profile</a>
        <p className="landing__skip">Already set up? <a href={routeHref("discover")}>Go to Discover</a></p>
        <p className="landing__by">— Built for PredictLeads</p>
      </div>
    </main>
  );
}
