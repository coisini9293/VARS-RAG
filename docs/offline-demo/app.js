(function () {
  "use strict";

  const DISCIPLINARY_Q =
    "According to the Laws of the Game, is a card warranted here? What disciplinary action should the referee take?";
  const DISCIPLINARY_A =
    "Yes, a yellow card is warranted here. The defender has no intention of playing the ball, but rather hits the attacker with medium intensity on the thigh.";
  const DISCIPLINARY_CAPTION =
    "Combined with football rule knowledge · Processing time: 1.15s";

  const QUICK_QUESTIONS = {
    scene: {
      q: "Please describe what happens in this clip and whether any infringement is visible.",
      a: "A defender challenges an attacker near the ball. Contact is made on the attacker's thigh without a clear attempt to play the ball. The challenge appears careless to reckless and should be stopped immediately.",
      caption: "Combined with football rule knowledge · Processing time: 1.02s",
      delay: 1000,
      rag: false,
    },
    disciplinary: {
      q: DISCIPLINARY_Q,
      a: DISCIPLINARY_A,
      caption: DISCIPLINARY_CAPTION,
      delay: 1150,
      rag: true,
    },
    conduct: {
      q: "Analyse the players' conduct in this clip. Does it comply with the Laws of the Game?",
      a: "The defender's conduct does not comply with fair play expectations. There is no genuine attempt to play the ball; the contact targets the opponent's leg. The attacker does not exaggerate. A caution is appropriate.",
      caption: "Combined with football rule knowledge · Processing time: 1.05s",
      delay: 1000,
      rag: false,
    },
    rule: {
      q: "Please explain this incident using the FIFA Laws of the Game. Would the referee's decision be correct?",
      a: "Under Law 12 (Fouls and Misconduct), a challenge that is careless or reckless is a foul. Medium-intensity contact on the thigh without playing the ball is typically reckless, meriting a yellow card. A caution would be correct.",
      caption: "Combined with football rule knowledge · Processing time: 1.08s",
      delay: 1000,
      rag: false,
    },
    basis: {
      q: "What is the specific rule basis for this decision? Please cite the relevant Laws of the Game.",
      a: "Law 12 — Fouls and Misconduct: a tackle or challenge that endangers the safety of an opponent or uses excessive force must be sanctioned. Careless → free kick; reckless → yellow card; excessive force → red card. This clip fits reckless challenge → yellow card.",
      caption: "Combined with football rule knowledge · Processing time: 1.04s",
      delay: 1000,
      rag: true,
    },
    detail: {
      q: "From a Laws of the Game perspective, what are the key details and decision points in this incident?",
      a: "Key points: (1) timing of contact vs ball; (2) point of contact (thigh); (3) intensity (medium); (4) intent to play the ball (absent). Together these indicate a reckless challenge. Decision: foul + yellow card; restart with direct free kick.",
      caption: "Combined with football rule knowledge · Processing time: 1.06s",
      delay: 1000,
      rag: false,
    },
  };

  const RAG_QUERY =
    "football foul rules direct free kick careless reckless excessive force tackle holding pushing";

  const RAG_SNIPPETS = [
    {
      rank: 1,
      score: "1.5",
      snippet:
        "Page 109 — 109 Laws of the Game 2025/26 | Law 12 | Fouls and Misconduct Direct and indirect free kicks and penalty kicks can or...",
    },
    {
      rank: 2,
      score: "1.3538",
      snippet:
        "Page 126 — 126 free kicks for offences involving a player entering, re-entering or leaving the field of play without permission are...",
    },
    {
      rank: 3,
      score: "1.2923",
      snippet:
        "Page 118 — 118 A player, substitute or substituted player who has been sent off must leave the vicinity of the field of play and the...",
    },
  ];

  const GENERIC_REPLY =
    "This is a scripted offline demo of X-VARS. For a full answer matching the recorded demo, try the Disciplinary quick ask, or ask: “According to the Laws of the Game, is a card warranted here? What disciplinary action should the referee take?”";

  const state = {
    processed: false,
    busy: false,
    messages: [],
    showedRag: false,
  };

  const el = {
    sidebar: document.getElementById("sidebar"),
    sidebarToggle: document.getElementById("sidebar-toggle"),
    btnProcess: document.getElementById("btn-process"),
    processSpinner: document.getElementById("process-spinner"),
    processSuccess: document.getElementById("process-success"),
    processStatus: document.getElementById("process-status"),
    metricsCard: document.getElementById("metrics-card"),
    chatLog: document.getElementById("chat-log"),
    quickAsks: document.getElementById("quick-asks"),
    thinking: document.getElementById("thinking"),
    chatForm: document.getElementById("chat-form"),
    chatInput: document.getElementById("chat-input"),
    btnSend: document.getElementById("btn-send"),
    btnClear: document.getElementById("btn-clear"),
    btnExport: document.getElementById("btn-export"),
    ragEmpty: document.getElementById("rag-empty"),
    ragResults: document.getElementById("rag-results"),
    ragQuery: document.getElementById("rag-query"),
    ragTbody: document.getElementById("rag-tbody"),
    statFrames: document.getElementById("stat-frames"),
    statTime: document.getElementById("stat-time"),
    statFeat: document.getElementById("stat-feat"),
    statDevice: document.getElementById("stat-device"),
    statusHint: document.getElementById("status-hint"),
    tabs: document.querySelectorAll(".tab"),
    panels: {
      dialogue: document.getElementById("panel-dialogue"),
      rag: document.getElementById("panel-rag"),
      status: document.getElementById("panel-status"),
    },
  };

  function sleep(ms) {
    return new Promise(function (resolve) {
      setTimeout(resolve, ms);
    });
  }

  function normalize(text) {
    return String(text || "")
      .trim()
      .replace(/\s+/g, " ")
      .toLowerCase();
  }

  function isDisciplinaryQuestion(text) {
    return normalize(text) === normalize(DISCIPLINARY_Q);
  }

  function setChatEnabled(on) {
    el.chatInput.disabled = !on || state.busy;
    el.btnSend.disabled = !on || state.busy;
    if (on) {
      el.chatInput.placeholder = "Type your question…";
    } else {
      el.chatInput.placeholder = "Type your question… (process video first)";
    }
  }

  function updateActionButtons() {
    const hasMsgs = state.messages.length > 0;
    el.btnClear.disabled = !hasMsgs || state.busy;
    el.btnExport.disabled = !hasMsgs;
  }

  function renderChat() {
    el.chatLog.innerHTML = "";
    if (!state.messages.length) {
      const empty = document.createElement("div");
      empty.className = "chat-empty";
      empty.textContent = state.processed
        ? "Video ready. Use Quick Ask or type a question."
        : "Process the video to start the dialogue.";
      el.chatLog.appendChild(empty);
    } else {
      state.messages.forEach(function (m) {
        const wrap = document.createElement("div");
        wrap.className = "msg " + m.role;
        const bubble = document.createElement("div");
        bubble.className = "msg-bubble";
        bubble.textContent = m.content;
        wrap.appendChild(bubble);
        if (m.caption) {
          const meta = document.createElement("span");
          meta.className = "msg-meta";
          meta.textContent = m.caption;
          wrap.appendChild(meta);
        }
        el.chatLog.appendChild(wrap);
      });
      el.chatLog.scrollTop = el.chatLog.scrollHeight;
    }

    const showQuick =
      state.processed && state.messages.length === 0 && !state.busy;
    el.quickAsks.classList.toggle("hidden", !showQuick);
    updateActionButtons();
  }

  function showRagResults(switchToTab) {
    if (el.ragQuery) {
      el.ragQuery.textContent = "Query: " + RAG_QUERY;
    }
    el.ragTbody.innerHTML = "";
    RAG_SNIPPETS.forEach(function (row) {
      const tr = document.createElement("tr");
      tr.innerHTML =
        "<td>" +
        row.rank +
        "</td><td>" +
        escapeHtml(row.score) +
        "</td><td>" +
        escapeHtml(row.snippet) +
        "</td>";
      el.ragTbody.appendChild(tr);
    });
    el.ragEmpty.classList.add("hidden");
    el.ragResults.classList.remove("hidden");
    state.showedRag = true;
    if (switchToTab) {
      switchTab("rag");
    }
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function fillSystemStatus() {
    el.statFrames.textContent = "16";
    el.statTime.textContent = "0.60";
    el.statFeat.textContent = "200";
    el.statDevice.textContent = "cuda";
    el.statusHint.textContent =
      "Status reflects the last successful offline video processing run.";
  }

  function clearSystemStatusSoft() {
    // Keep metrics after reprocess reset until process finishes again
    el.statFrames.textContent = "—";
    el.statTime.textContent = "—";
    el.statFeat.textContent = "—";
    el.statDevice.textContent = "—";
    el.statusHint.textContent = "Process the video to populate system status.";
  }

  async function processVideo() {
    if (state.busy) return;
    state.busy = true;
    setChatEnabled(false);
    el.btnProcess.disabled = true;
    el.processSuccess.classList.add("hidden");
    el.processSpinner.classList.remove("hidden");
    el.processStatus.textContent = "Processing example video…";

    await sleep(600);

    state.processed = true;
    state.busy = false;
    el.processSpinner.classList.add("hidden");
    el.processSuccess.textContent =
      "Video processing complete! Extracted 16 frames, took 0.6s.";
    el.processSuccess.classList.remove("hidden");
    el.metricsCard.classList.remove("hidden");
    el.processStatus.textContent = "✅ Video processed successfully.";
    el.btnProcess.textContent = "🔄 Reprocess Video";
    el.btnProcess.disabled = false;
    fillSystemStatus();
    setChatEnabled(true);
    renderChat();
  }

  function resetForReprocess() {
    state.processed = false;
    state.messages = [];
    state.showedRag = false;
    el.ragEmpty.classList.remove("hidden");
    el.ragResults.classList.add("hidden");
    el.ragTbody.innerHTML = "";
    el.metricsCard.classList.add("hidden");
    el.processSuccess.classList.add("hidden");
    clearSystemStatusSoft();
    setChatEnabled(false);
    renderChat();
  }

  async function askQuestion(question, preset) {
    if (!state.processed || state.busy) return;

    const q = question;
    let answer;
    let caption;
    let delay;
    let useRag;

    if (preset) {
      answer = preset.a;
      caption = preset.caption;
      delay = preset.delay;
      useRag = !!preset.rag;
    } else if (isDisciplinaryQuestion(q)) {
      answer = DISCIPLINARY_A;
      caption = DISCIPLINARY_CAPTION;
      delay = 1150;
      useRag = true;
    } else {
      answer = GENERIC_REPLY;
      caption = "Offline scripted demo · Processing time: 1.00s";
      delay = 1000;
      useRag = false;
    }

    state.busy = true;
    setChatEnabled(false);
    el.quickAsks.classList.add("hidden");
    el.btnProcess.disabled = true;
    updateActionButtons();

    state.messages.push({ role: "user", content: q });
    renderChat();
    el.thinking.classList.remove("hidden");

    await sleep(delay);

    el.thinking.classList.add("hidden");
    state.messages.push({
      role: "assistant",
      content: answer,
      caption: caption,
    });
    state.busy = false;
    el.btnProcess.disabled = false;
    setChatEnabled(true);
    renderChat();

    if (useRag) {
      showRagResults(true);
    }
  }

  function clearConversation() {
    if (state.busy) return;
    state.messages = [];
    state.showedRag = false;
    el.ragEmpty.classList.remove("hidden");
    el.ragResults.classList.add("hidden");
    el.ragTbody.innerHTML = "";
    renderChat();
  }

  function exportConversation() {
    if (!state.messages.length) return;
    const lines = [
      "X-VARS Intelligent Football Referee System — Conversation Export",
      "Generated: " + new Date().toISOString(),
      "Video: clip_1.mp4",
      "",
    ];
    state.messages.forEach(function (m, i) {
      const who = m.role === "user" ? "User" : "Assistant";
      lines.push("[" + (i + 1) + "] " + who + ":");
      lines.push(m.content);
      if (m.caption) lines.push("(" + m.caption + ")");
      lines.push("");
    });
    const blob = new Blob([lines.join("\n")], {
      type: "text/plain;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download =
      "xvars_conversation_" +
      new Date().toISOString().replace(/[:.]/g, "-") +
      ".txt";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function switchTab(name) {
    el.tabs.forEach(function (tab) {
      const active = tab.getAttribute("data-tab") === name;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", active ? "true" : "false");
    });
    Object.keys(el.panels).forEach(function (key) {
      const panel = el.panels[key];
      if (key === name) {
        panel.hidden = false;
        panel.classList.add("active");
      } else {
        panel.hidden = true;
        panel.classList.remove("active");
      }
    });
  }

  // Events
  el.sidebarToggle.addEventListener("click", function () {
    el.sidebar.classList.toggle("open");
  });

  el.btnProcess.addEventListener("click", function () {
    if (state.busy) return;
    if (state.processed) {
      resetForReprocess();
      el.btnProcess.textContent = "🚀 Process Video";
      el.processStatus.textContent =
        "Use the example video, then click Process Video.";
      return;
    }
    processVideo();
  });

  document.querySelectorAll(".btn-quick").forEach(function (btn) {
    btn.addEventListener("click", function () {
      const key = btn.getAttribute("data-q");
      const preset = QUICK_QUESTIONS[key];
      if (!preset) return;
      askQuestion(preset.q, preset);
    });
  });

  el.chatForm.addEventListener("submit", function (e) {
    e.preventDefault();
    const text = el.chatInput.value.trim();
    if (!text) return;
    el.chatInput.value = "";
    askQuestion(text, null);
  });

  el.btnClear.addEventListener("click", clearConversation);
  el.btnExport.addEventListener("click", exportConversation);

  el.tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      switchTab(tab.getAttribute("data-tab"));
    });
  });

  // Init
  renderChat();
  setChatEnabled(false);
})();
