import { stepLabel, timeOf } from "../format.jsx";

// The question running right now (with each pipeline step as it happens),
// and the questions waiting behind it.
export default function LiveQuestion({ status, current, queue, onRemove }) {
  if (status === "loading") {
    return <div className="panel live muted">Loading the search models - questions asked now will start shortly...</div>;
  }
  if (!current && queue.length === 0) return null;

  return (
    <div className="panel live">
      {current && (
        <>
          <div className="live-head">
            <span className="pulse" /> <strong>Running:</strong> {current.text}
            <span className="muted small"> - asked by {current.asked_by}</span>
          </div>
          <ol className="steps">
            {current.steps.length === 0 && <li className="muted">Starting...</li>}
            {current.steps.map((s, i) => (
              <li key={i}>
                <span className="muted small">{timeOf(s.at)}</span> {stepLabel(s)}
              </li>
            ))}
          </ol>
        </>
      )}
      {queue.length > 0 && (
        <>
          <div className="queue-head">Waiting ({queue.length})</div>
          <ul className="queue">
            {queue.map((q) => (
              <li key={q.id}>
                <span>{q.text}</span>
                <span className="muted small"> - {q.asked_by}</span>
                <button className="link" onClick={() => onRemove(q.id)} title="Take it out of the queue">
                  remove
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
