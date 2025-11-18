import { useState, useEffect } from 'react';
import Head from 'next/head';
import PreprocessingPanel from '../components/PreprocessingPanel';
import ChatbotPanel from '../components/ChatbotPanel';
import MetricsDashboard from '../components/MetricsDashboard';
import Header from '../components/Header';
import { Toaster } from 'react-hot-toast';

export default function Home() {
  const [showMetrics, setShowMetrics] = useState(false);

  return (
    <>
      <Head>
        <title>AURORA - Intelligent Data Preprocessing</title>
        <meta name="description" content="AI-powered data preprocessing with privacy-preserving learning" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <Toaster position="top-right" />

      <div className="min-h-screen">
        {/* Header */}
        <Header onToggleMetrics={() => setShowMetrics(!showMetrics)} />

        {/* Metrics Dashboard (Collapsible) */}
        {showMetrics && (
          <div className="animate-in slide-in-from-top duration-300">
            <MetricsDashboard />
          </div>
        )}

        {/* Main Content */}
        <main className="container mx-auto px-4 py-8">
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
            {/* Left Panel - Preprocessing Interface (60%) */}
            <div className="lg:col-span-3">
              <PreprocessingPanel />
            </div>

            {/* Right Panel - Chatbot (40%) */}
            <div className="lg:col-span-2">
              <ChatbotPanel />
            </div>
          </div>
        </main>

        {/* Footer */}
        <footer className="mt-12 py-6 text-center text-slate-600 text-sm">
          <p>
            AURORA v1.0 - Intelligent Data Preprocessing System
            <br />
            <span className="text-xs">
              Privacy-preserving • Lightning-fast • Self-learning
            </span>
          </p>
        </footer>
      </div>
    </>
  );
}
