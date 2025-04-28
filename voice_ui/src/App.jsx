import { useState } from "react";
import axios from "axios";
import "./App.css";

export default function App() {
  const [transcript, setTranscript] = useState("");
  const [results, setResults] = useState([]);

  const handleVoiceInput = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Speech recognition not supported in this browser.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      console.log("Voice recognition started. Speak now.");
    };

    recognition.onresult = (event) => {
      const transcriptText = event.results[0][0].transcript;
      console.log("Recognized:", transcriptText);
      setTranscript(transcriptText);
      runSearch(transcriptText);
    };

    recognition.onerror = (event) => {
      console.error("Recognition error:", event.error);
    };

    recognition.start();
  };

  const [searchTime, setSearchTime] = useState(null);

  const runSearch = async (inputText) => {
    if (!inputText.trim()) return;
    try {
      const start = Date.now();  // Start timer here — after speech finishes
      const res = await axios.post("http://127.0.0.1:8000/api/search", {
        text: inputText,
      });
      const elapsed = Date.now() - start;
      setResults(res.data);
      setSearchTime(elapsed);  // Set the latency time
    } catch (err) {
      console.error("Search failed", err);
      setSearchTime(null);  // Clear latency if failed
    }
  };
  
  

  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-100 via-purple-100 to-blue-100 py-10 px-4">
      <div className="max-w-4xl mx-auto bg-white shadow-2xl rounded-3xl p-10">
      {searchTime !== null && (
        <div className="text-center text-sm text-gray-600 mt-2">
          Latency: {(searchTime / 1000).toFixed(2)} seconds
        </div>
      )}
        <h1 className="text-4xl font-extrabold text-center text-indigo-700 mb-6">
          Voice AI Workout Assistant
        </h1>

        <textarea
          rows={3}
          value={transcript}
          onChange={(e) => setTranscript(e.target.value)}
          placeholder="e.g. Find me a 20 minute yoga with Alex"
          className="w-full p-4 border-2 border-indigo-200 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
        />

        <div className="mt-4 flex justify-center gap-4">
          <button
            onClick={() => runSearch(transcript)}
            className="px-6 py-2 bg-indigo-600 text-white rounded-full font-semibold hover:bg-indigo-700 shadow-md"
          >
            🔍 Search
          </button>
          <button
            onClick={handleVoiceInput}
            className="px-6 py-2 bg-white border border-indigo-300 text-indigo-700 rounded-full font-semibold hover:bg-indigo-50 shadow-md"
          >
            🎙 Speak
          </button>
        </div>

        <div className="mt-10">
          <h2 className="text-2xl font-bold text-indigo-800 mb-4">Top Recommendations</h2>
          {results.length === 0 ? (
            <p className="text-gray-500">No results yet.</p>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              {results.map((r, i) => (
                <div
                  key={i}
                  className="p-5 bg-white border border-gray-200 rounded-xl shadow-lg text-left hover:shadow-xl transition-shadow"
                >
                  <div className="text-lg font-semibold text-gray-900 mb-1">
                    {r.title}
                  </div>
                  <div className="text-sm text-gray-700">
                    Duration: {r.duration} min<br />
                    Instructor: {r.instructor}<br />
                    Intensity: {r.intensity}<br />
                    Type: {r.type}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
