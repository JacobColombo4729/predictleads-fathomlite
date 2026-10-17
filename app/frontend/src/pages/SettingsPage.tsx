import { useAdmin } from "../lib/admin";

export default function SettingsPage() {
  const { admin, setAdmin } = useAdmin();
  return (
    <main className="cp" aria-label="Settings">
      <header className="cp-head">
        <div>
          <div className="cp-eyebrow">Settings</div>
          <h1>Settings</h1>
          <p className="cp-sub">How the app looks and what it shows. These settings are saved in this browser only.</p>
        </div>
      </header>

      <section className="cp-card settings-card" aria-label="Admin mode">
        <label className="settings-switch">
          <input type="checkbox" role="switch" checked={admin} onChange={(e) => setAdmin(e.target.checked)} />
          <span className="settings-switch__text">
            <strong>Admin mode</strong>
            <span className="cp-sub">
              Adds a collapsed “How this works” box beside each machine-learning part of the app (Discover, Compass, Signals and Magic fill).
              Each one says what the approach is, why we chose it, what data it uses, how it was checked and where it is weak.
              It only adds explanations: nothing about your results or your data changes.
            </span>
          </span>
        </label>
        {admin && (
          <p className="notice notice--compact" role="status">
            Admin mode is on. Look for the <span className="mlnote__tag">Admin</span> boxes on the other pages.
          </p>
        )}
      </section>
    </main>
  );
}
