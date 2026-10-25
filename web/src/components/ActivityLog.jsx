import { timeOf } from "../format.jsx";

export default function ActivityLog({ log }) {
  if (log.length === 0) return null;
  return (
    <section className="panel log">
      <h2>Activity</h2>
      <ul>
        {[...log].reverse().map((entry, i) => (
          <li key={i}>
            <span className="muted small">{timeOf(entry.at)}</span> {entry.text}
          </li>
        ))}
      </ul>
    </section>
  );
}
