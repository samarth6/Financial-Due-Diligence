import { useCallback, useEffect, useRef, useState } from "react";

// One WebSocket to the room (/ws on the same host that served this page).
// The server sends the WHOLE room state after every change, so this hook only
// keeps the latest copy - there is nothing to merge and nothing to drift.
//
// If the connection drops (server restart, Wi-Fi blip) it reconnects every
// two seconds and re-joins under the same name automatically.

const NAME_KEY = "dd-room-name";
const RECONNECT_MS = 2000;

function socketUrl() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${window.location.host}/ws`;
}

function savedName() {
  try {
    return window.localStorage.getItem(NAME_KEY) || "";
  } catch {
    return "";
  }
}

function saveName(name) {
  try {
    window.localStorage.setItem(NAME_KEY, name);
  } catch {
    // Private mode etc. - you'll just be asked for your name again.
  }
}

export function useRoom() {
  const [connected, setConnected] = useState(false);
  const [state, setState] = useState(null);
  const [you, setYou] = useState(null);
  const [error, setError] = useState(null);
  const [name, setName] = useState(savedName);

  const socketRef = useRef(null);
  const nameRef = useRef(name);
  nameRef.current = name;

  useEffect(() => {
    let closedOnPurpose = false;
    let retryTimer = null;

    function connect() {
      const ws = new WebSocket(socketUrl());
      socketRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        if (nameRef.current) {
          ws.send(JSON.stringify({ type: "join", name: nameRef.current }));
        }
      };
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === "state") {
          setState(msg.state);
          setYou(msg.you);
        } else if (msg.type === "error") {
          setError({ message: msg.message, at: Date.now() });
        }
      };
      ws.onclose = () => {
        setConnected(false);
        setYou(null);
        if (!closedOnPurpose) retryTimer = setTimeout(connect, RECONNECT_MS);
      };
    }

    connect();
    return () => {
      closedOnPurpose = true;
      clearTimeout(retryTimer);
      socketRef.current?.close();
    };
  }, []);

  const send = useCallback((message) => {
    const ws = socketRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    } else {
      setError({ message: "not connected to the room - reconnecting...", at: Date.now() });
    }
  }, []);

  const join = useCallback(
    (newName) => {
      const trimmed = newName.trim();
      if (!trimmed) return;
      saveName(trimmed);
      setName(trimmed);
      send({ type: "join", name: trimmed });
    },
    [send],
  );

  return {
    connected,
    state,
    you,
    name,
    error,
    clearError: () => setError(null),
    join,
    ask: (text) => send({ type: "ask", text }),
    remove: (questionId) => send({ type: "remove", question_id: questionId }),
    propose: (kind, questionId) => send({ type: "propose", kind, question_id: questionId }),
    vote: (actionId, approve) => send({ type: "vote", action_id: actionId, approve }),
  };
}
