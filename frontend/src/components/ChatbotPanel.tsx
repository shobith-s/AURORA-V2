import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Sparkles } from 'lucide-react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export default function ChatbotPanel() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: "👋 Hi! I'm AURORA's AI assistant. I can help you understand preprocessing recommendations, explain technical concepts, or guide you through the system. What would you like to know?",
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const quickQuestions = [
    "What is symbolic preprocessing?",
    "How does the neural oracle work?",
    "Explain privacy-preserving learning",
    "When should I use log transform?"
  ];

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    // Simulate AI response (in production, connect to OpenAI or custom LLM)
    setTimeout(() => {
      const response = generateResponse(input);
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, assistantMessage]);
      setIsTyping(false);
    }, 1000);
  };

  const generateResponse = (query: string): string => {
    const q = query.toLowerCase();

    if (q.includes('symbolic') || q.includes('rule')) {
      return "🎯 **Symbolic Preprocessing** uses 100+ hand-crafted rules to handle 80% of common cases. It's lightning-fast (<100μs) and fully explainable. For example, if it detects all unique values, it knows it's likely an ID column and recommends dropping it. No ML needed for obvious cases!";
    }

    if (q.includes('neural') || q.includes('oracle')) {
      return "🧠 **Neural Oracle** is our lightweight XGBoost model that handles ambiguous cases (20%). It only activates when symbolic confidence < 0.9. Using just 10 features, it makes decisions in <5ms. It's trained specifically on edge cases where multiple preprocessing options seem valid.";
    }

    if (q.includes('privacy') || q.includes('learn')) {
      return "🔒 **Privacy-Preserving Learning** is AURORA's secret sauce! When you correct a recommendation, we extract a statistical pattern (like 'numeric + high skew + outliers') WITHOUT storing your actual data. After 3-4 similar corrections, we automatically create a new rule. Your data never leaves your machine!";
    }

    if (q.includes('log') || q.includes('transform')) {
      return "📊 **Log Transform** is recommended for highly skewed positive data (skewness > 2.0). It compresses large values and spreads small values, making the distribution more normal. Perfect for revenue, prices, or any exponentially-distributed data. Note: requires all positive values!";
    }

    if (q.includes('confidence') || q.includes('score')) {
      return "🎯 **Confidence Score** indicates how certain AURORA is about its recommendation. 90%+ = highly confident (symbolic rule matched perfectly), 70-90% = moderate (neural oracle had to decide), <70% = uncertain (might want human review). You can always provide corrections to help the system learn!";
    }

    return `I understand you're asking about: "${query}"\n\n🤔 While I don't have a specific answer for that, I can help with:\n\n• Explaining preprocessing techniques\n• Understanding AURORA's recommendations\n• Troubleshooting data issues\n• Best practices for data preparation\n\nTry asking something more specific, or select a quick question below!`;
  };

  return (
    <div className="glass-card h-[calc(100vh-12rem)] flex flex-col">
      {/* Header */}
      <div className="border-b border-slate-200 p-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
            <Bot className="w-6 h-6 text-white" />
          </div>
          <div>
            <h3 className="font-bold text-slate-800">AI Assistant</h3>
            <p className="text-xs text-slate-600">Powered by AURORA Intelligence</p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div className="flex items-start gap-2 max-w-[85%]">
              {message.role === 'assistant' && (
                <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center flex-shrink-0">
                  <Bot className="w-5 h-5 text-white" />
                </div>
              )}
              <div
                className={`${
                  message.role === 'user' ? 'chat-bubble-user' : 'chat-bubble-assistant'
                }`}
              >
                <p className="text-sm whitespace-pre-line">{message.content}</p>
                <p className="text-xs opacity-60 mt-1">
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </p>
              </div>
              {message.role === 'user' && (
                <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center flex-shrink-0">
                  <User className="w-5 h-5 text-white" />
                </div>
              )}
            </div>
          </div>
        ))}

        {isTyping && (
          <div className="flex items-start gap-2">
            <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div className="chat-bubble-assistant">
              <div className="loading-dots flex gap-1">
                <span className="w-2 h-2 bg-slate-400 rounded-full"></span>
                <span className="w-2 h-2 bg-slate-400 rounded-full"></span>
                <span className="w-2 h-2 bg-slate-400 rounded-full"></span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Quick Questions */}
      {messages.length <= 2 && (
        <div className="px-4 pb-2">
          <p className="text-xs font-medium text-slate-600 mb-2">Quick questions:</p>
          <div className="flex flex-wrap gap-2">
            {quickQuestions.map((q, idx) => (
              <button
                key={idx}
                onClick={() => setInput(q)}
                className="text-xs px-3 py-1.5 bg-purple-50 text-purple-700 rounded-lg hover:bg-purple-100 transition"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <div className="border-t border-slate-200 p-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask me anything about data preprocessing..."
            className="flex-1 px-4 py-2 rounded-lg border border-slate-300 focus:border-purple-500 focus:ring-2 focus:ring-purple-200 outline-none transition text-sm"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isTyping}
            className="px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:shadow-lg transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
}
