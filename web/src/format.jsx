// Small display helpers shared by the components.

// One pipeline step (sent by Layers 4-5 through the room) as plain English.
export function stepLabel({ stage, details }) {
  const d = details || {};
  switch (stage) {
    case "lookup":
      return d.facts?.length
        ? `Found ${d.facts.length} figure(s) in the financial statements: ${[...new Set(d.facts)].join(", ")}`
        : "No matching figures in the financial statements";
    case "search":
      return `Searching the filing (attempt ${d.attempt}): "${d.query}"`;
    case "grade":
      return d.facts
        ? `Grading ${d.passages} passages and ${d.facts} figure(s) for relevance`
        : `Grading ${d.passages} passages for relevance`;
    case "graded": {
      const passages = `Attempt ${d.attempt}: ${d.relevant} of ${d.total} passages relevant`;
      // Figures are graded once, on attempt 1 (facts_total is null after that).
      return d.facts_total
        ? `${passages}; ${d.facts_relevant} of ${d.facts_total} figure(s) judged relevant`
        : passages;
    }
    case "rewrite":
      return `Weak results - rewording the question (attempt ${d.attempt}): "${d.query}"`;
    case "stagnated":
      return `Rewording found the same passages again (attempt ${d.attempt}) - stopping`;
    case "give_up":
      return `Nothing scored above ${d.below} (best ${d.best}) - the filing likely doesn't cover this; stopping early`;
    case "write":
      return `Writing the answer from ${d.passages} passage(s) and ${d.facts} figure(s)`;
    default:
      return stage;
  }
}

// "Net income was $112,010 million [S1]" with the marker highlighted.
export function WithMarkers({ text }) {
  const parts = (text || "").split(/(\[S?\d+\])/g);
  return (
    <>
      {parts.map((part, i) =>
        /^\[S?\d+\]$/.test(part) ? (
          <span key={i} className="marker">{part}</span>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  );
}

export function timeOf(iso) {
  if (!iso) return "";
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export const ACTION_LABELS = {
  checklist: "Add to checklist",
  flag: "Flag for review",
  memo: "Draft risk memo",
};
