import { BookmarkIcon, CompassIcon, FindIcon, UserIcon } from "./Icons";
import { routeHref, type Route } from "../lib/router";
import ThemeToggle from "./ThemeToggle";

interface Props {
  route: Route;
  savedCount: number;
}

const items: { route: Route; label: string; Icon: typeof CompassIcon }[] = [
  { route: "discover", label: "Discover", Icon: FindIcon },
  { route: "saved", label: "Saved", Icon: BookmarkIcon },
  { route: "compass", label: "Compass", Icon: CompassIcon },
  { route: "profile", label: "Profile", Icon: UserIcon },
];

export default function SideNav({ route, savedCount }: Props) {
  return (
    <nav className="sidenav" aria-label="Primary">
      <div className="sidenav__brand" aria-hidden="true">PL</div>
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
