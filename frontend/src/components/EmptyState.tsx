export default function EmptyState({ onReset }: { onReset: () => void }) {
  return (
    <div className="empty" role="status">
      <h3>No jobs match these filters</h3>
      <p>Try a different search term or clear a filter.</p>
      <button type="button" className="btn btn--primary" onClick={onReset}>Clear search and filters</button>
    </div>
  );
}
