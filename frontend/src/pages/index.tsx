import { useState, useEffect } from 'react';
import Head from 'next/head';
import PreprocessingPanel from '../components/PreprocessingPanel';
import ChatbotPanel from '../components/ChatbotPanel';
import MetricsDashboard from '../components/MetricsDashboard';
import LearningProgressPanel from '../components/LearningProgressPanel';
import Header from '../components/Header';
import ScriptIDE from '../components/ScriptIDE';
import { Toaster } from 'react-hot-toast';

export default function Home() {
  const [showMetrics, setShowMetrics] = useState(false);
  const [showChatbot, setShowChatbot] = useState(false);
  const [showIDE, setShowIDE] = useState(false);

  return (
    <>
      <Head>
        <title>AURORA - Intelligent Data Preprocessing</title>
        <meta name="description" content="AI-powered data preprocessing with privacy-preserving learning" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <Toaster position="top-right" />

      {/* Custom Script IDE Modal */}
      <ScriptIDE isOpen={showIDE} onClose={() => setShowIDE(false)} />

      <div className="min-h-screen">
        {/* Header */}
        <Header
          onToggleMetrics={() => setShowMetrics(!showMetrics)}
          onToggleIDE={() => setShowIDE(true)}
        />

        {/* Metrics Dashboard (Collapsible) */}
        {showMetrics && (
          <div className="animate-in slide-in-from-top duration-300">
            <MetricsDashboard />
          </div>
        )}

        {/* Main Content */}
        <main className="w-full px-6 py-8">
          {/* Full Width Layout */}
          <div className="w-full">
            <PreprocessingPanel />
          </div>
        </main>

        {/* Floating Chatbot Button */}
        {!showChatbot && (
          <button
            onClick={() => setShowChatbot(true)}
            className="fixed bottom-6 right-6 w-14 h-14 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-full shadow-lg hover:shadow-xl transition-all duration-300 flex items-center justify-center z-50 hover:scale-110"
            aria-label="Open AI Assistant"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
            </svg>
          </button>
        )}

        {/* Floating Chatbot Panel */}
        {showChatbot && (
          <div className="fixed bottom-6 right-6 w-96 h-[600px] bg-white rounded-lg shadow-2xl z-50 flex flex-col overflow-hidden animate-in slide-in-from-bottom duration-300">
            {/* Chatbot Header */}
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                </svg>
                <h3 className="font-semibold">AI Assistant</h3>
              </div>
              <button
                onClick={() => setShowChatbot(false)}
                className="hover:bg-white/20 rounded-lg p-1 transition-colors"
                aria-label="Close chatbot"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Chatbot Content */}
            <div className="flex-1 overflow-hidden">
              <ChatbotPanel />
            </div>
          </div>
        )}

        {/* Footer */}
        <footer className="mt-12 py-6 text-center text-slate-400 text-sm">
          <p>
            AURORA V4 - Intelligent Data Preprocessing System
            <br />
            <span className="text-xs text-slate-500">
              Symbolic-first • Validated Learning • A/B Testing • Zero Overgeneralization
            </span>
          </p>
        </footer>
      </div>
    </>
  );
}
