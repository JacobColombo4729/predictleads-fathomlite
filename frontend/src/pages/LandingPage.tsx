import { useEffect, useRef } from "react";
import fishUrl from "../assets/fathomlite-mark.png";
import wallLeftUrl from "../assets/ocean-wall.svg";
import wallRightUrl from "../assets/ocean-wall-right.svg";
import { routeHref } from "../lib/router";

const RADII = [230, 175, 128, 88];
const OPACITY = [0.018, 0.028, 0.042, 0.065];

/**
 * The first page a new visitor sees: the anglerfish on the left, the FathomLite name and motto on the right.
 * A soft pool of light follows the pointer, like the anglerfish's lure. It rests beside the motto when the pointer
 * is away, and is switched off for people who prefer less motion and on touch screens.
 */
export default function LandingPage() {
  const hero = useRef<HTMLElement>(null);
  const lights = useRef<HTMLDivElement>(null);
  const anchor = useRef<HTMLParagraphElement>(null);

  useEffect(() => {
    const root = hero.current, layer = lights.current, rest_ = anchor.current;
    if (!root || !layer || !rest_) return;
    const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches || window.matchMedia("(hover: none)").matches;
    const rings = RADII.map((r, i) => {
      const ring = document.createElement("div");
      ring.className = "landing__ring";
      ring.style.cssText = `width:${r * 2}px;height:${r * 2}px;opacity:${OPACITY[i]}`;
      layer.appendChild(ring);
      return ring;
    });
    if (calm) {
      layer.style.display = "none";
      return () => rings.forEach((r) => r.remove());
    }
    const lightAt = (x: number, y: number) => {
      rings.forEach((ring, i) => { ring.style.left = `${x - RADII[i]}px`; ring.style.top = `${y - RADII[i]}px`; });
    };
    const rest = () => {
      const box = root.getBoundingClientRect(), r = rest_.getBoundingClientRect();
      lightAt(r.left - box.left + r.width / 2, r.top - box.top + r.height / 2);
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
    <main className="landing" ref={hero} aria-label="FathomLite">
      <div className="landing__rays" aria-hidden="true" />
      <img className="landing__wall landing__wall--left" src={wallLeftUrl} alt="" aria-hidden="true" />
      <img className="landing__wall landing__wall--right" src={wallRightUrl} alt="" aria-hidden="true" />
      <div className="landing__lights" aria-hidden="true" ref={lights} />
      <div className="landing__content">
        <img className="landing__fish" src={fishUrl} alt="An anglerfish with a glowing lure" />
        <div className="landing__text">
          <h1 className="landing__brand">Fathom<span className="landing__lite">Lite</span></h1>
          <p className="landing__tag" ref={anchor}>Deep dive. Clear light.</p>
          <a className="landing__cta" href={routeHref("profile")}>Get Started <span aria-hidden="true">→</span></a>
          <p className="landing__skip">Already set up? <a href={routeHref("discover")}>Go to Discover</a></p>
        </div>
      </div>
      <p className="landing__by">Built for PredictLeads</p>
    </main>
  );
}
