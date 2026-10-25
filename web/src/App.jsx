import { useEffect } from "react";
import ActionPanel from "./components/ActionPanel.jsx";
import ActivityLog from "./components/ActivityLog.jsx";
import AskBox from "./components/AskBox.jsx";
import JoinScreen from "./components/JoinScreen.jsx";
import LiveQuestion from "./components/LiveQuestion.jsx";
import QuestionCard from "./components/QuestionCard.jsx";
import { useRoom } from "./useRoom.js";

const ERROR_SHOWN_MS = 6000;

export default function App() {
  const room = useRoom();
  const { connected, state, you, error, clearError } = room;

  useEffect(() => {
    if (!error) return undefined;
    const timer = setTimeout(clearError, ERROR_SHOWN_MS);
    return () => clearTimeout(timer);
  }, [error, clearError]);

  const joined = Boolean(you && state && state.members.some((m) => m.id === you));
  if (!joined) {
    return (
      <>
        <JoinScreen connected={connected} onJoin={room.join} initialName={room.name} />
        {error && <div className="toast">{error.message}</div>}
      </>
    );
  }

  const history = [...state.history].reverse(); // newest first
  const busy = Boolean(state.current) || state.queue.length > 0;

  return (
    <div className="app">
      <header className="topbar">
        <h1>Due-Diligence Room</h1>
        <div className="members">
          {!connected && <span className="offline">reconnecting...</span>}
          {state.members.map((m) => (
            <span key={m.id} className={m.id === you ? "member me" : "member"}>
              {m.name}{m.id === you ? " (you)" : ""}
            </span>
          ))}
        </div>
      </header>

      <main className="layout">
        <section className="column">
          <AskBox onAsk={room.ask} busy={busy} />
          <LiveQuestion status={state.status} current={state.current} queue={state.queue} onRemove={room.remove} />
          {history.length === 0 && !state.current && (
            <p className="muted empty">No questions yet. Anything asked here appears for everyone in the room.</p>
          )}
          {history.map((item) => (
            <QuestionCard
              key={item.id}
              item={item}
              actions={state.actions}
              actionPending={Boolean(state.pending_action)}
              onPropose={room.propose}
            />
          ))}
        </section>

        <aside className="column side">
          <ActionPanel pending={state.pending_action} history={state.action_history} you={you} onVote={room.vote} />
          <ActivityLog log={state.log} />
        </aside>
      </main>

      {error && (
        <div className="toast" onClick={clearError}>{error.message}</div>
      )}
    </div>
  );
}
