import { useState } from "react";
import {
  Bot,
  User,
  Mail,
  Paperclip,
  Send,
  AlertTriangle,
  CheckCircle,
  MessageCircle,
} from "lucide-react";

import "./App.css";


// ============================================================
// BACKEND URL
// ============================================================

const API_URL = "http://127.0.0.1:5000";


// ============================================================
// REASON GENERATOR
// ============================================================

function generateReasons(message, prediction) {

  const text = message.toLowerCase();

  const reasons = [];

  if (prediction === "SPAM") {

    if (
      text.includes("win") ||
      text.includes("won") ||
      text.includes("winner") ||
      text.includes("prize") ||
      text.includes("free") ||
      text.includes("reward")
    ) {
      reasons.push(
        "Contains suspicious words such as win, prize or free"
      );
    }

    if (
      text.includes("http") ||
      text.includes("www.") ||
      text.includes("click")
    ) {
      reasons.push(
        "Contains a suspicious link or call-to-action"
      );
    }

    if (
      text.includes("urgent") ||
      text.includes("hurry") ||
      text.includes("limited") ||
      text.includes("now") ||
      text.includes("today")
    ) {
      reasons.push(
        "Contains urgency-related language"
      );
    }

    if (
      text.includes("claim") ||
      text.includes("offer") ||
      text.includes("cash") ||
      text.includes("money")
    ) {
      reasons.push(
        "Contains promotional or reward-related content"
      );
    }

    if (reasons.length === 0) {

      reasons.push(
        "The trained spam detection model classified this message as spam"
      );

      reasons.push(
        "The message contains patterns similar to spam messages in the dataset"
      );
    }

  } else {

    reasons.push(
      "The message was classified as normal email content"
    );

    if (
      text.includes("meeting") ||
      text.includes("project") ||
      text.includes("schedule") ||
      text.includes("team")
    ) {
      reasons.push(
        "Contains normal business or communication content"
      );
    }

    if (
      !text.includes("http") &&
      !text.includes("www.")
    ) {
      reasons.push(
        "No suspicious link detected"
      );
    }

    if (
      !text.includes("urgent") &&
      !text.includes("hurry") &&
      !text.includes("limited")
    ) {
      reasons.push(
        "No obvious urgency-related language detected"
      );
    }
  }

  return reasons.slice(0, 4);
}


// ============================================================
// APP
// ============================================================

function App() {

  const [message, setMessage] = useState("");

  const [messages, setMessages] = useState([]);

  const [loading, setLoading] = useState(false);

  // Chat is initially closed
  const [chatOpen, setChatOpen] = useState(false);


  // ==========================================================
  // CURRENT TIME
  // ==========================================================

  function getTime() {

    return new Date().toLocaleTimeString(
      [],
      {
        hour: "2-digit",
        minute: "2-digit",
      }
    );
  }


  // ==========================================================
  // OPEN CHAT
  // ==========================================================

  const openChat = () => {

    setChatOpen(true);

  };


  // ==========================================================
  // PREDICT EMAIL
  // ==========================================================

  const predictEmail = async () => {

    if (!message.trim()) {
      return;
    }

    const userMessage = message;


    // Add user message
    setMessages((previous) => [
      ...previous,
      {
        type: "user",
        text: userMessage,
        time: getTime(),
      },
    ]);


    setMessage("");

    setLoading(true);


    try {

      // ======================================================
      // CALL FLASK BACKEND
      // ======================================================

      const response = await fetch(
        `${API_URL}/api/predict?message=${encodeURIComponent(
          userMessage
        )}`
      );


      const data = await response.json();


      if (!response.ok || data.status !== "success") {

        throw new Error(
          data.message || "Prediction failed"
        );
      }


      // ======================================================
      // GENERATE REASONS
      // ======================================================

      const reasons = generateReasons(
        userMessage,
        data.prediction
      );


      // ======================================================
      // ADD RESULT
      // ======================================================

      setMessages((previous) => [
        ...previous,
        {
          type: "result",

          prediction: data.prediction,

          label: data.label,

          decisionScore: data.decision_score,

          reasons: reasons,

          time: getTime(),
        },
      ]);

    } catch (error) {

      console.error(error);

      setMessages((previous) => [
        ...previous,
        {
          type: "error",

          text:
            "Unable to connect to the spam detection backend.",

          time: getTime(),
        },
      ]);

    } finally {

      setLoading(false);

    }
  };


  // ==========================================================
  // ENTER KEY
  // ==========================================================

  const handleKeyDown = (event) => {

    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {

      event.preventDefault();

      predictEmail();

    }

  };


  // ==========================================================
  // UI
  // ==========================================================

  return (

    <div className="app-container">


      {/* ======================================================
          HEADER
      ====================================================== */}

      <header className="top-header">

        <div className="header-left">

          <div className="mail-logo">

            <Mail
              size={48}
              strokeWidth={1.8}
            />

          </div>


          <div>

            <h1>
              Spam Email Detection Chatbot
            </h1>

            <p>
              Paste your email content and let AI detect
              whether it is Spam or Not Spam!
            </p>

          </div>

        </div>


        <div className="ai-badge">

          <Bot size={43} />

          <div>

            <strong>
              AI Powered
            </strong>

            <span>
              Safe. Smart. Secure.
            </span>

          </div>

        </div>

      </header>


      {/* ======================================================
          MAIN LAYOUT
      ====================================================== */}

      <div className="main-layout">


        {/* ====================================================
            LEFT SIDEBAR
            ONLY CHAT BUTTON
        ==================================================== */}

        <aside className="sidebar">

          <button
            className={`chat-button ${
              chatOpen ? "active" : ""
            }`}
            onClick={openChat}
          >

            <MessageCircle size={34} />

            <span>
              Chat
            </span>

          </button>

        </aside>


        {/* ====================================================
            CHAT AREA
        ==================================================== */}

        <main className="chat-area">


          {/* ==================================================
              BEFORE CHAT IS OPEN
          ================================================== */}

          {!chatOpen && (

            <div className="welcome-area">

              <Bot
                size={75}
                strokeWidth={1.5}
              />

              <h2>
                Click Chat to Start
              </h2>

              <p>
                Click the Chat button on the left
                to analyze your email.
              </p>

            </div>

          )}


          {/* ==================================================
              CHAT CONTENT
          ================================================== */}

          {chatOpen && (

            <>


              {/* ==============================================
                  MESSAGES
              ============================================== */}

              <div className="messages-container">


                {messages.length === 0 && (

                  <div className="chat-start-message">

                    <Bot
                      size={65}
                      strokeWidth={1.5}
                    />

                    <h2>
                      Spam Email Detection
                    </h2>

                    <p>
                      Paste your email content below
                      and click Send to check whether
                      it is Spam or Not Spam.
                    </p>

                  </div>

                )}


                {/* ============================================
                    MESSAGE LIST
                ============================================ */}

                {messages.map(
                  (item, index) => {


                    // ========================================
                    // USER MESSAGE
                    // ========================================

                    if (item.type === "user") {

                      return (

                        <div
                          className="message-row user-row"
                          key={index}
                        >

                          <div className="user-message">

                            <pre>
                              {item.text}
                            </pre>

                            <span className="message-time">
                              {item.time}
                            </span>

                          </div>


                          <div className="user-icon">

                            <User size={28} />

                          </div>

                        </div>

                      );

                    }


                    // ========================================
                    // RESULT
                    // ========================================

                    if (item.type === "result") {

                      const isSpam =
                        item.prediction === "SPAM";


                      return (

                        <div
                          className="message-row result-row"
                          key={index}
                        >

                          <div className="bot-icon">

                            <Bot size={29} />

                          </div>


                          <div
                            className={`result-card ${
                              isSpam
                                ? "spam-card"
                                : "ham-card"
                            }`}
                          >


                            {/* RESULT HEADER */}

                            <div className="result-header">

                              <div className="result-title">

                                {isSpam ? (

                                  <AlertTriangle
                                    size={48}
                                    strokeWidth={2}
                                  />

                                ) : (

                                  <CheckCircle
                                    size={48}
                                    strokeWidth={2}
                                  />

                                )}


                                <span>

                                  Result:

                                  {" "}

                                  <strong>

                                    {isSpam
                                      ? "SPAM"
                                      : "NOT SPAM (Ham)"}

                                  </strong>

                                </span>

                              </div>


                              <div className="score">

                                Model Score:

                                {" "}

                                <strong>
                                  {item.decisionScore}
                                </strong>

                              </div>

                            </div>


                            <div className="result-divider" />


                            {/* REASONS */}

                            <div className="reasons">

                              <h4>
                                Reasons:
                              </h4>

                              <ul>

                                {item.reasons.map(
                                  (reason, reasonIndex) => (

                                    <li
                                      key={reasonIndex}
                                    >
                                      {reason}
                                    </li>

                                  )
                                )}

                              </ul>

                            </div>


                            <span className="result-time">
                              {item.time}
                            </span>

                          </div>

                        </div>

                      );

                    }


                    // ========================================
                    // ERROR
                    // ========================================

                    if (item.type === "error") {

                      return (

                        <div
                          className="message-row result-row"
                          key={index}
                        >

                          <div className="bot-icon">

                            <Bot size={29} />

                          </div>


                          <div className="error-card">

                            <AlertTriangle size={30} />

                            <span>
                              {item.text}
                            </span>

                          </div>

                        </div>

                      );

                    }


                    return null;

                  }
                )}


                {/* ==========================================
                    LOADING
                ========================================== */}

                {loading && (

                  <div className="message-row result-row">

                    <div className="bot-icon">

                      <Bot size={29} />

                    </div>


                    <div className="loading-card">

                      <div className="loading-dots">

                        <span></span>
                        <span></span>
                        <span></span>

                      </div>

                      <span>
                        Analyzing your email...
                      </span>

                    </div>

                  </div>

                )}

              </div>


              {/* ==============================================
                  INPUT
              ============================================== */}

              <div className="input-section">

                <div className="input-box">

                  <textarea
                    value={message}
                    onChange={(event) =>
                      setMessage(event.target.value)
                    }
                    onKeyDown={handleKeyDown}
                    placeholder="Paste your email content here..."
                    rows={2}
                  />


                  <button
                    className="attachment-button"
                    type="button"
                  >

                    <Paperclip size={25} />

                  </button>

                </div>


                <button
                  className="send-button"
                  onClick={predictEmail}
                  disabled={
                    loading ||
                    !message.trim()
                  }
                >

                  <Send size={25} />

                  <span>
                    {loading
                      ? "Checking..."
                      : "Send"}
                  </span>

                </button>

              </div>

            </>

          )}

        </main>

      </div>

    </div>

  );
}


export default App;