import { STAGES, type Stage } from "../types";

interface Props {
  value: Stage;
  label: string;
  onChange: (s: Stage) => void;
}

export default function StageSelect({ value, label, onChange }: Props) {
  return (
    <select
      className={`stage stage--${value.toLowerCase()}`}
      aria-label={label}
      value={value}
      onClick={(e) => e.stopPropagation()}
      onChange={(e) => onChange(e.target.value as Stage)}
    >
      {STAGES.map((s) => <option key={s} value={s}>{s}</option>)}
    </select>
  );
}
