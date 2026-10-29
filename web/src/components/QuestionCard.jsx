import { useState } from "react";
import { ACTION_LABELS, WithMarkers, stepLabel, timeOf } from "../format.jsx";

function Confidence({ confidence, citations }) {
  if (!confidence) return null;
  const c = confidence;
  // An answer citing a database figure ([S1]) gets its score from that figure,
  // not from the three search signals, so showing them would only confuse.
  const fromFigures = citations.some((cit) => cit.marker.startsWith("[S"));
  return (
    <div className="confidence">
      <span className={`band band-${c.band}`}>{c.band.toUpperCase()} confidence ({c.score})</span>
      {!fromFigures && (
        <span className="muted small">
          {" "}both searches agreed {c.hybrid_agreement} - reranker {c.rerank_strength} - first-try {c.loop_cleanliness}
        </span>
      )}
      {c.notes?.length > 0 && (
        <ul className="notes">
          {c.notes.map((n, i) => <li key={i}>{n}</li>)}
        </ul>
      )}
    </div>
  );
}

// The start of a passage, with "more" to read all of it in place. The full
// text keeps one table row per line; the short version is one running line,
// minus the long "─" rules tables are drawn with.
function PassageText({ passage }) {
  const [open, setOpen] = useState(false);
  const full = passage.text;
  const short = passage.snippet.replace(/─{3,}/g, " ");
  // Old answers (asked before the full text was sent) have no `text`.
  if (!full) return <div className="small">{short}...</div>;
  if (short.length >= full.replace(/\s+/g, " ").length) {
    return <div className="small passage-text">{full}</div>;
  }
  return (
    <div className="small passage-text">
      {open ? full : `${short}...`}
      <button type="button" className="link" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? "less" : "more"}
      </button>
    </div>
  );
}

function ActionButtons({ item, actions, actionPending, onPropose }) {
  const a = item.answer;
  if (!a) return null;
  return (
    <div className="action-buttons">
      {actions.map((kind) => {
        const notAllowed = !a.answered && kind !== "flag";
        return (
          <button
            key={kind}
            className="secondary"
            disabled={actionPending || notAllowed}
            title={
              actionPending
                ? "The room is already deciding on an action"
                : notAllowed
                  ? "Only a flag can act on an unanswered question"
                  : "Propose this - everyone in the room must approve it"
            }
            onClick={() => onPropose(kind, item.id)}
          >
            {ACTION_LABELS[kind] || kind}
          </button>
        );
      })}
    </div>
  );
}

// One finished question: the answer, its sources and confidence, and the
// passages and steps behind it.
export default function QuestionCard({ item, actions, actionPending, onPropose }) {
  const a = item.answer;
  return (
    <article className="panel card">
      <header>
        <strong>{item.text}</strong>
        <span className="muted small"> - {item.asked_by}, {timeOf(item.asked_at)}</span>
      </header>

      {item.error && <p className="error-text">The pipeline failed on this question: {item.error}</p>}

      {a && (
        <>
          {a.answered ? (
            <p className="answer"><WithMarkers text={a.text} /></p>
          ) : (
            <p className="refused">Not answered: {a.text}</p>
          )}

          {a.citations.length > 0 && (
            <ul className="sources">
              {a.citations.map((c) => (
                <li key={c.marker}><span className="marker">{c.marker}</span> {c.source}</li>
              ))}
            </ul>
          )}

          <Confidence confidence={a.confidence} citations={a.citations} />
          <div className="muted small">{a.elapsed_s}s, {a.attempts.length} search attempt(s)</div>

          {a.passages.length > 0 && (
            <details>
              <summary>Search results ({a.passages.length} passages graded)</summary>
              <ul className="passages">
                {a.passages.map((p, i) => (
                  <li key={i}>
                    <span
                      className={p.relevance >= 0.5 ? "rel rel-yes" : "rel"}
                      title="How well the grader judged this passage answers the question (0-1). Not the answer's confidence."
                    >
                      relevance {p.relevance.toFixed(2)}
                    </span>{" "}
                    <strong>{p.where}</strong>
                    <PassageText passage={p} />
                    <div className="muted small">grader: {p.reason}</div>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </>
      )}

      {item.steps?.length > 0 && (
        <details>
          <summary>How it was answered ({item.steps.length} steps)</summary>
          <ol className="steps">
            {item.steps.map((s, i) => <li key={i}>{stepLabel(s)}</li>)}
          </ol>
        </details>
      )}

      <ActionButtons item={item} actions={actions} actionPending={actionPending} onPropose={onPropose} />
    </article>
  );
}
