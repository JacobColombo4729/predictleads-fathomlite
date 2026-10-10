import type { SVGProps } from "react";

function Svg(props: SVGProps<SVGSVGElement>) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    />
  );
}

export const SearchIcon = () => (
  <Svg><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></Svg>
);
export const CompassIcon = (p: SVGProps<SVGSVGElement>) => (
  <Svg {...p}><circle cx="12" cy="12" r="9" /><path d="m15.5 8.5-2 5-5 2 2-5z" /></Svg>
);
export const BookmarkIcon = ({ filled, ...p }: SVGProps<SVGSVGElement> & { filled?: boolean }) => (
  <Svg {...p} fill={filled ? "currentColor" : "none"}><path d="M6 4h12v17l-6-4-6 4z" /></Svg>
);
export const UserIcon = (p: SVGProps<SVGSVGElement>) => (
  <Svg {...p}><circle cx="12" cy="8" r="4" /><path d="M4 21c1-4 4-6 8-6s7 2 8 6" /></Svg>
);
export const PinIcon = () => (
  <Svg><path d="M12 21s7-6.2 7-11.5A7 7 0 0 0 5 9.5C5 14.800 12 21 12 21z" /><circle cx="12" cy="9.500" r="2.500" /></Svg>
);
export const BuildingIcon = () => (
  <Svg><path d="M4 21V5l8-2v18M12 9h8v12M4 21h16M8 9h.01M8 13h.01M8 17h.01M16 13h.01M16 17h.01" /></Svg>
);
export const CashIcon = () => (
  <Svg><rect x="3" y="6" width="18" height="12" rx="2" /><circle cx="12" cy="12" r="2.500" /></Svg>
);
export const CalendarIcon = () => (
  <Svg><rect x="4" y="5" width="16" height="16" rx="2" /><path d="M4 10h16M9 3v4M15 3v4" /></Svg>
);
export const ExternalIcon = () => (
  <Svg width={14} height={14}><path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5" /></Svg>
);
export const BackIcon = () => (
  <Svg><path d="M15 6l-6 6 6 6" /></Svg>
);

export const ThumbUpIcon = () => (
  <Svg><path d="M7 10v12" /><path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z" /></Svg>
);
export const ThumbDownIcon = () => (
  <Svg><path d="M17 14V2" /><path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z" /></Svg>
);
