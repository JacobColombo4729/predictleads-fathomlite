import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";

/** Admin mode shows the "How this works" notes beside each machine-learning part of the app. It only changes what is shown. */
const KEY = "pl.admin";

/** Off unless switched on. Browser storage may be unavailable, so every access is guarded. */
function readAdmin(): boolean {
  try {
    return window.localStorage.getItem(KEY) === "1";
  } catch {
    return false;
  }
}

function writeAdmin(on: boolean): void {
  try {
    window.localStorage.setItem(KEY, on ? "1" : "0");
  } catch {
    // not remembered; the choice still applies for this visit
  }
}

interface AdminValue {
  admin: boolean;
  setAdmin: (on: boolean) => void;
}

const AdminContext = createContext<AdminValue>({ admin: false, setAdmin: () => {} });

export function AdminProvider({ children }: { children: ReactNode }) {
  const [admin, setAdminState] = useState<boolean>(readAdmin);
  const setAdmin = useCallback((on: boolean) => {
    writeAdmin(on);
    setAdminState(on);
  }, []);
  const value = useMemo(() => ({ admin, setAdmin }), [admin, setAdmin]);
  return <AdminContext.Provider value={value}>{children}</AdminContext.Provider>;
}

export function useAdmin(): AdminValue {
  return useContext(AdminContext);
}
