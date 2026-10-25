import { useState } from "react";

const MAX = 500;

export default function AskBox({ onAsk, busy }) {
  const [text, setText] = useState("");

  function submit(event) {
    event.preventDefault();
    const q = text.trim();
    if (q.length < 3) return;
    onAsk(q);
    setText("");
  }

  return (
    <form className="panel ask" onSubmit={submit}>
      <textarea
        rows={2}
        maxLength={MAX}
        placeholder="Ask a question about the 10-K, e.g. What risks does Apple disclose about tariffs?"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) submit(e);
        }}
      />
      <div className="ask-row">
        <span className="muted small">
          {busy ? "A question is running - yours will wait in the queue." : "Enter to ask, Shift+Enter for a new line."}
        </span>
        <button type="submit" disabled={text.trim().length < 3}>Ask</button>
      </div>
    </form>
  );
}
