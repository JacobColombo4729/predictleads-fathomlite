export interface ToastState {
  message: string;
  onUndo?: () => void;
}

export default function Toast({ toast, onDismiss }: { toast: ToastState; onDismiss: () => void }) {
  return (
    <div className="toast" role="status">
      <span>{toast.message}</span>
      {toast.onUndo && (
        <button type="button" className="toast__btn" onClick={() => { toast.onUndo?.(); onDismiss(); }}>
          Undo
        </button>
      )}
      <button type="button" className="toast__btn toast__btn--quiet" onClick={onDismiss} aria-label="Dismiss">
        Dismiss
      </button>
    </div>
  );
}
