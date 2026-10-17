import { useEffect, useState } from "react";

export type Route = "welcome" | "discover" | "saved" | "compass" | "signals" | "profile" | "settings";

const routes: Route[] = ["welcome", "discover", "saved", "compass", "signals", "profile", "settings"];

function parse(hash: string): Route {
  const name = hash.replace(/^#\/?/, "").split(/[/?]/)[0];
  return (routes as string[]).includes(name) ? (name as Route) : "discover";
}

export function routeHref(route: Route): string {
  return `#/${route}`;
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parse(window.location.hash));

  useEffect(() => {
    if (!window.location.hash) window.history.replaceState(null, "", routeHref(route));
    const onChange = () => setRoute(parse(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return route;
}
