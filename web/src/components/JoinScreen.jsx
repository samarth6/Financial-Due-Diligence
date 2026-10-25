import { useState } from "react";

export default function JoinScreen({ connected, onJoin, initialName }) {
  const [name, setName] = useState(initialName || "");

  function submit(event) {
    event.preventDefault();
    if (name.trim()) onJoin(name);
  }

  return (
    <div className="join">
      <h1>Due-Diligence Room</h1>
      <p className="muted">
        Ask questions about Apple's FY2025 10-K together. Everyone here sees each
        question, how it's being answered, and votes on any action before it runs.
      </p>
      <form onSubmit={submit}>
        <input
          autoFocus
          maxLength={40}
          placeholder="Your name"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <button type="submit" disabled={!connected || !name.trim()}>
          {connected ? "Join" : "Connecting..."}
        </button>
      </form>
    </div>
  );
}
