import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Sparkles, Brain, TrendingUp } from 'lucide-react';
import axios from 'axios';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  confidence?: number;
  suggestions?: string[];
}

interface ChatbotPanelEnhancedProps {
  dataContext?: {
    columns?: string[];
    rowCount?: number;
    currentResults?: any;
  };
}

export default function ChatbotPanelEnhanced({ dataContext }: ChatbotPanelEnhancedProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: "👋 Hi! I'm AURORA's Intelligent Assistant. I can analyze your data, explain decisions with SHAP, and answer statistical questions. What would you like to know?",
      timestamp: new Date(),
      confidence: 1.0,
      suggestions: [
        "What can you do?",
        "Give me a dataset summary",
        "Explain SHAP values"
      ]
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

  // Update context when dataContext changes
  useEffect(() => {
    if (dataContext?.columns && dataContext.columns.length > 0) {
      // Optionally notify assistant of new data
      axios.post('/api/chat/set_context', {
        columns: dataContext.columns,
        row_count: dataContext.rowCount
      }).catch(() => {
        // Silent fail - context update is non-critical
      });
    }
  }, [dataContext]);

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

    try {
      // Call real backend API
      const response = await axios.post('/api/chat/query', {
        question: input,
        context: dataContext
      });

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.data.answer,
        timestamp: new Date(),
        confidence: response.data.confidence,
        suggestions: response.data.suggestions || []
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      // Fallback to basic response on error
      const fallbackMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `I apologize, but I encountered an error. Please try again or rephrase your question.\n\n**Error:** ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date(),
        confidence: 0.0
      };
      setMessages(prev => [...prev, fallbackMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleQuickQuestion = (question: string) => {
    setInput(question);
    // Auto-send after short delay
    setTimeout(() => {
      if (question === input) {
        handleSend();
      }
    }, 100);
  };

  const getConfidenceColor = (confidence?: number) => {
    if (!confidence) return 'text-slate-600';
    if (confidence >= 0.9) return 'text-green-600';
    if (confidence >= 0.7) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="glass-card h-[calc(100vh-12rem)] flex flex-col">
      {/* Header */}
      <div className="border-b border-slate-200 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center relative">
              <Bot className="w-6 h-6 text-white" />
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-pulse" />
            </div>
            <div>
              <h3 className="font-bold text-slate-800 flex items-center gap-2">
                Intelligent Assistant
                <span className="text-xs px-2 py-0.5 bg-purple-100 text-purple-700 rounded-full">
                  SHAP Enabled
                </span>
              </h3>
              <p className="text-xs text-slate-600">Real-time analysis • Statistical queries • SHAP explanations</p>
            </div>
          </div>
        </div>
      </div>

      {/* Data Context Indicator */}
      {dataContext?.columns && dataContext.columns.length > 0 && (
        <div className="px-4 py-2 bg-gradient-to-r from-blue-50 to-purple-50 border-b border-slate-200">
          <div className="flex items-center gap-2 text-xs">
            <TrendingUp className="w-4 h-4 text-blue-600" />
            <span className="text-slate-700">
              📊 Analyzing: <strong>{dataContext.rowCount || 0}</strong> rows × <strong>{dataContext.columns.length}</strong> columns
            </span>
          </div>
        </div>
      )}

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
              <div className="flex-1">
                <div
                  className={`${
                    message.role === 'user' ? 'chat-bubble-user' : 'chat-bubble-assistant'
                  }`}
                >
                  <p className="text-sm whitespace-pre-line">{message.content}</p>
                  <div className="flex items-center justify-between mt-2">
                    <p className="text-xs opacity-60">
                      {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </p>
                    {message.confidence !== undefined && message.confidence < 1.0 && (
                      <span className={`text-xs font-medium ${getConfidenceColor(message.confidence)}`}>
                        {(message.confidence * 100).toFixed(0)}% confident
                      </span>
                    )}
                  </div>
                </div>

                {/* Suggestions */}
                {message.suggestions && message.suggestions.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-2">
                    {message.suggestions.map((suggestion, idx) => (
                      <button
                        key={idx}
                        onClick={() => handleQuickQuestion(suggestion)}
                        className="text-xs px-3 py-1.5 bg-purple-50 text-purple-700 rounded-lg hover:bg-purple-100 transition border border-purple-200"
                      >
                        {suggestion}
                      </button>
                    ))}
                  </div>
                )}
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
              <div className="flex items-center gap-2">
                <Brain className="w-4 h-4 animate-pulse" />
                <span className="text-sm">Analyzing...</span>
              </div>
              <div className="loading-dots flex gap-1 mt-2">
                <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-slate-200 p-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
            placeholder="Ask about data, statistics, SHAP explanations..."
            className="flex-1 px-4 py-3 rounded-lg border border-slate-300 focus:border-purple-500 focus:ring-2 focus:ring-purple-200 outline-none transition text-sm"
            disabled={isTyping}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isTyping}
            className="px-5 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:shadow-lg transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            <Send className="w-5 h-5" />
            <span className="text-sm font-medium hidden sm:inline">Send</span>
          </button>
        </div>

        {/* Example queries */}
        <div className="mt-3 flex flex-wrap gap-2">
          <span className="text-xs text-slate-500">Try:</span>
          {[
            "Dataset summary",
            "Statistics for revenue",
            "Explain SHAP"
          ].map((example, idx) => (
            <button
              key={idx}
              onClick={() => handleQuickQuestion(example)}
              className="text-xs px-2 py-1 text-purple-600 hover:bg-purple-50 rounded transition"
            >
              "{example}"
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
