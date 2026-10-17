import { BookmarkIcon, CompassIcon, FindIcon, GearIcon, SignalsIcon, UserIcon } from "./Icons";
import { routeHref, type Route } from "../lib/router";
import ThemeToggle from "./ThemeToggle";
import markUrl from "../assets/fathomlite-icon.png";

interface Props {
  route: Route;
  savedCount: number;
}

const items: { route: Route; label: string; Icon: typeof CompassIcon }[] = [
  { route: "discover", label: "Discover", Icon: FindIcon },
  { route: "saved", label: "Saved", Icon: BookmarkIcon },
  { route: "compass", label: "Compass", Icon: CompassIcon },
  { route: "signals", label: "Signals", Icon: SignalsIcon },
  { route: "profile", label: "Profile", Icon: UserIcon },
  { route: "settings", label: "Settings", Icon: GearIcon },
];

export default function SideNav({ route, savedCount }: Props) {
  return (
    <nav className="sidenav" aria-label="Primary">
      <a className="sidenav__brand" href={routeHref("welcome")} aria-label="FathomLite home" title="FathomLite: Deep dive. Clear light."><img src={markUrl} alt="" /></a>
      <ul className="sidenav__list">
        {items.map(({ route: r, label, Icon }) => (
          <li key={r}>
            <a
              href={routeHref(r)}
              className={`navitem${r === route ? " navitem--active" : ""}`}
              aria-current={r === route ? "page" : undefined}
              aria-label={r === "saved" ? `Saved jobs, ${savedCount} saved` : label}
            >
              <Icon width={20} height={20} />
              <span>{label}</span>
              {r === "saved" && savedCount > 0 && <span className="navitem__count" aria-hidden="true">{savedCount}</span>}
            </a>
          </li>
        ))}
      </ul>
      <div className="sidenav__foot"><ThemeToggle /></div>
    </nav>
  );
}
