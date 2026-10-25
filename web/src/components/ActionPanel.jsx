import { ACTION_LABELS } from "../format.jsx";

const STATUS_TEXT = {
  drafting: "Drafting it and checking it against the rules...",
  voting: "Waiting for everyone to approve",
  running: "Everyone approved - running it now...",
};

function DraftFields({ draft }) {
  if (!draft) return null;
  return (
    <dl className="draft">
      {Object.entries(draft).map(([field, value]) => (
        <div key={field}>
          <dt>{field}</dt>
          <dd className={field === "body" ? "pre" : undefined}>{Array.isArray(value) ? value.join(", ") : String(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function Pending({ action, you, onVote }) {
  const me = action.voters.find((v) => v.id === you);
  const canVote = action.status === "voting" && me && me.vote === null;
  return (
    <div className="pending">
      <div>
        <strong>{ACTION_LABELS[action.kind] || action.kind}</strong>
        <span className="muted small"> - proposed by {action.proposed_by}</span>
      </div>
      <div className="muted small">for: {action.question}</div>
      <p className="status">{STATUS_TEXT[action.status] || action.status}</p>
      <DraftFields draft={action.draft} />

      {action.voters.length > 0 && (
        <ul className="voters">
          {action.voters.map((v) => (
            <li key={v.id}>
              <span className={v.vote === true ? "vote-yes" : v.vote === false ? "vote-no" : "vote-wait"}>
                {v.vote === true ? "approved" : v.vote === false ? "rejected" : "waiting"}
              </span>{" "}
              {v.name}{v.id === you ? " (you)" : ""}
            </li>
          ))}
        </ul>
      )}

      {canVote && (
        <div className="vote-buttons">
          <button onClick={() => onVote(action.id, true)}>Approve</button>
          <button className="danger" onClick={() => onVote(action.id, false)}>Reject</button>
        </div>
      )}
      {action.status === "voting" && !me && (
        <p className="muted small">You joined after this was proposed, so you don't vote on it.</p>
      )}
    </div>
  );
}

// Layer 6 actions, decided by the whole room: the one being decided now,
// and what happened to earlier ones.
export default function ActionPanel({ pending, history, you, onVote }) {
  return (
    <section className="panel actions">
      <h2>Actions</h2>
      {pending ? (
        <Pending action={pending} you={you} onVote={onVote} />
      ) : (
        <p className="muted small">
          Propose an action from any answer below. It runs only after everyone in the room approves it.
        </p>
      )}

      {history.length > 0 && (
        <ul className="action-history">
          {[...history].reverse().map((a) => (
            <li key={a.id}>
              <strong>{ACTION_LABELS[a.kind] || a.kind}</strong>: {a.status}
              {a.result && <div className="small">{a.result}</div>}
              {a.problems.length > 0 && (
                <ul className="small problems">
                  {a.problems.map((p, i) => <li key={i}>{p}</li>)}
                </ul>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
