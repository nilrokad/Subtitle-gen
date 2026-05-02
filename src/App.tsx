import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Upload, FileAudio, Download, Loader2, CheckCircle, AlertCircle, FileText, Sparkles, List, AlignLeft, Type, Files, X, Clock, Play, Pause } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { GoogleGenAI } from "@google/genai";

interface Word {
  text: string;
  start: number;
  end: number;
  confidence: number;
  speaker?: string;
}

interface Sentence {
  text: string;
  start: number;
  end: number;
  words: Word[];
}

interface Paragraph {
  text: string;
  start: number;
  end: number;
  words: Word[];
}

interface TranscriptionResult {
  fileName: string;
  id: string;
  text: string;
  words: Word[];
  sentences: Sentence[];
  paragraphs: Paragraph[];
  srt?: string;
}

interface QueueItem {
  id: string;
  file: File;
  status: 'pending' | 'transcribing' | 'gemini-crafting' | 'completed' | 'error';
  result?: TranscriptionResult;
  error?: string;
  startTime?: number;
  elapsedTime?: number;
}

type TabType = 'transcript' | 'sentences' | 'paragraphs' | 'ai-subtitles';

const CONCURRENCY_LIMIT = 1;

export default function App() {
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [activeTab, setActiveTab] = useState<TabType>('ai-subtitles');
  const [selectedQueueId, setSelectedQueueId] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isQueueRunning, setIsQueueRunning] = useState(false);
  const [elapsedTimers, setElapsedTimers] = useState<Record<string, number>>({});
  const fileInputRef = useRef<HTMLInputElement>(null);
  const processingIdRef = useRef<string | null>(null);

  const handleFiles = (newFiles: File[]) => {
    const newItems: QueueItem[] = newFiles.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      file,
      status: 'pending'
    }));
    setQueue(prev => [...prev, ...newItems]);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleFiles(Array.from(e.target.files) as File[]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files) {
      handleFiles(Array.from(e.dataTransfer.files) as File[]);
    }
  };

  const removeFile = (id: string) => {
    setQueue(prev => prev.filter(item => item.id !== id));
    if (selectedQueueId === id) setSelectedQueueId(null);
  };

  const generateSRTWithGemini = async (words: Word[], fileName: string, file: File): Promise<{ srt: string, correctedText: string }> => {
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || '' });

      // Convert file to base64 for Gemini
      const fileBase64 = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const base64 = (reader.result as string).split(',')[1];
          resolve(base64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });

      const prompt = `
        You are an expert subtitle editor and transcriber.
        
        INPUTS:
        1. AUDIO: (Provided as binary/inline data).
        2. ASSEMBLYAI DATA: A JSON array of words with start/end timestamps from a different engine.
        
        TASK (The "Gemini Correction" Workflow):
        1. GEMINI TRANSCRIPTION: First, you must listen to the provided audio yourself and transcribe it with perfect accuracy, including proper punctuation (commas, full stops, question marks) and natural Hinglish flow.
        2. COMPARISON & CORRECTION: Compare your inner transcription with the provided AssemblyAI word list. 
        3. MASTER TRANSCRIPT: Using your transcription as the cleaner reference, fix the AssemblyAI transcript. Correct wrong words, missing words, and spelling errors.
        4. TIMESTAMP MAPPING (CRITICAL): You MUST keep the precise word-level start/end timestamps from AssemblyAI. If you add/correct a word, map it to the closest original timestamp.
        5. GENERATE SRT: Create an SRT file based on this corrected Master Transcript.
        
        STRICT CONSTRAINTS:
        1. CONVERT ALL TEXT TO HINGLISH (Hindi written in Latin script).
           Example: "You need the right guide" -> "Tumhein sahi guide chahiye"
           Example: "All government schemes" -> "Sabhi sarkari yojnaon"
        2. NUMERICAL NUMBERS & RANGES: Always write numbers in digits, NOT words. Convert Hindi number multipliers (sau, hazar, lakh) accurately.
           - "do hazar" -> "2000", "pachas" -> "50", "aath sau" -> "800", "barah sau" -> "1200".
           - Ranges: "aath sau se barah sau" -> "800 se 1200" or "800-1200".
        3. NO SKIPPING: Include every single word and letter. Do not summarize.
        4. NO WORD BREAKING: Ensure every word is complete. 
        5. Each segment MUST have a minimum of 2 words and a maximum of 4 words.
        6. PUNCTUATION BREAK (CRITICAL): If a word ends with punctuation (like a comma ",", full stop ".", question mark "?", or exclamation "!"), that segment MUST end there. NO words should follow punctuation within the same segment.
        7. GAP REDUCTION & TIMESTAMP ADJUSTMENT (CRITICAL): Adjust the END timestamp of every segment to reduce awkward gaps, while strictly ensuring NO OVERLAPS.
           - If the gap between the current segment's end and the next segment's start is LESS than 50ms, extend the current segment's end time to EXACTLY match the next segment's start time (fill the gap).
           - If the gap is 50ms or MORE, extend the current segment's end time by exactly 100ms (but NEVER let it exceed or overlap with the next segment's start time).
           - NEVER change the START timestamp of any segment.
        8. CURRENCY FORMATTING: Always change "rupee", "rupees", "rupaye", or "rupiya" to the symbol "₹" placed BEFORE the number.
           - "1 rupee" -> "₹1", "500 rupaye" -> "₹500".
        
        LOGIC FLOW:
        - Listen to audio -> Correct words/punctuation in the word list -> Maintain timestamps -> Generate JSON.
        
        JSON DATA (AssemblyAI Reference):
        ${JSON.stringify(words.map(w => ({ text: w.text, start: w.start, end: w.end })))}
        
        Return ONLY a JSON object with the following structure:
        {
          "correctedText": "The full reconstructed transcript with perfect Hinglish punctuation",
          "srt": "The final SRT content"
        }
      `;

      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: {
          parts: [
            {
              inlineData: {
                data: fileBase64,
                mimeType: file.type
              }
            },
            { text: prompt }
          ]
        },
        config: {
          responseMimeType: "application/json"
        }
      });

      const text = response.text;
      const result = JSON.parse(text || '{}');
      
      return {
        srt: result.srt || '',
        correctedText: result.correctedText || ''
      };
    } catch (err: any) {
      console.error(`Gemini error for ${fileName}:`, err);
      return {
        srt: `Error generating SRT for ${fileName}: ${err.message}`,
        correctedText: ''
      };
    }
  };

  const processItem = useCallback(async (item: QueueItem) => {
    if (processingIdRef.current === item.id) return;
    processingIdRef.current = item.id;

    const startTime = Date.now();
    // Update status to transcribing and set start time
    setQueue(prev => prev.map(q => q.id === item.id ? { ...q, status: 'transcribing', startTime } : q));
    setElapsedTimers(prev => ({ ...prev, [item.id]: 0 }));

    try {
      const formData = new FormData();
      formData.append('audio', item.file);

      const response = await fetch('/api/transcribe', {
        method: 'POST',
        body: formData,
      });

      const contentType = response.headers.get("content-type");
      if (!response.ok) {
        let errorMessage = 'Transcription failed';
        if (contentType && contentType.includes("application/json")) {
          try {
            const errData = await response.json();
            errorMessage = errData.error || errorMessage;
          } catch (e) {
            errorMessage = `Server error (${response.status}): ${response.statusText}`;
          }
        } else {
          errorMessage = `Server error (${response.status}): ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const transcribeData = await response.json();
      const transcriptId = transcribeData.id;

      // Polling loop for transcription status
      let data = null;
      let pollRetries = 0;
      while (true) {
        try {
          const statusResponse = await fetch(`/api/status/${transcriptId}`);
          const statusContentType = statusResponse.headers.get("content-type");
          
          if (!statusContentType || !statusContentType.includes("application/json")) {
            throw new Error("Invalid content type received during polling");
          }

          if (!statusResponse.ok) {
            throw new Error(`Status check failed: ${statusResponse.statusText}`);
          }
          
          const statusData = await statusResponse.json();
          if (statusData.status === 'completed') {
            data = statusData.result;
            break;
          } else if (statusData.status === 'error') {
            throw new Error(statusData.error || "Transcription failed");
          }
          
          pollRetries = 0; 
        } catch (pollErr) {
          console.warn("Polling error, retrying...", pollErr);
          pollRetries++;
          if (pollRetries > 10) {
            throw new Error("Lost connection to server while polling status. Please try again.");
          }
        }
        await new Promise(resolve => setTimeout(resolve, 3000));
      }

      if (!data) throw new Error("No transcription data received");
      
      // Update status to gemini-crafting
      setQueue(prev => prev.map(q => q.id === item.id ? { ...q, status: 'gemini-crafting', result: data } : q));

      // Generate SRT using enhanced multi-modal step
      const srtData = await generateSRTWithGemini(data.words, item.file.name, item.file);

      const finalElapsed = Math.floor((Date.now() - startTime) / 1000);
      setQueue(prev => prev.map(q => q.id === item.id ? { 
        ...q, 
        status: 'completed', 
        result: { 
          ...data, 
          fileName: item.file.name,
          srt: srtData.srt, 
          text: srtData.correctedText || data.text 
        },
        elapsedTime: finalElapsed
      } : q));
      
      setElapsedTimers(prev => {
        const next = { ...prev };
        delete next[item.id];
        return next;
      });

      setSelectedQueueId(prev => prev || item.id);
    } catch (err: any) {
      setQueue(prev => prev.map(q => q.id === item.id ? { ...q, status: 'error', error: err.message } : q));
    } finally {
      processingIdRef.current = null;
    }
  }, []);


  // Timer Effect: Update elapsed time for active items separately to avoid queue re-renders
  useEffect(() => {
    if (!isQueueRunning) return;
    
    const timer = setInterval(() => {
      const now = Date.now();
      setElapsedTimers(prev => {
        const next: Record<string, number> = {};
        let changed = false;
        
        queue.forEach(q => {
          if (q.status === 'transcribing' || q.status === 'gemini-crafting') {
            const elapsed = Math.floor((now - (q.startTime || now)) / 1000);
            if (prev[q.id] !== elapsed) {
              next[q.id] = elapsed;
              changed = true;
            } else {
              next[q.id] = prev[q.id];
            }
          }
        });
        
        return changed ? next : prev;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [isQueueRunning, queue]);

  // Queue Manager Effect: Single-File Logic
  useEffect(() => {
    if (!isQueueRunning) return;

    const activeCount = queue.filter(q => q.status === 'transcribing' || q.status === 'gemini-crafting').length;
    const pendingItems = queue.filter(q => q.status === 'pending');
    
    // Only start a new item if no items are currently active
    if (activeCount === 0 && pendingItems.length > 0 && !processingIdRef.current) {
      // Take the next item
      const nextItem = pendingItems[0];
      processItem(nextItem);
    }
  }, [queue, processItem, isQueueRunning]);

  const downloadSRT = (result: TranscriptionResult) => {
    try {
      if (!result.srt) {
        console.error("No SRT content to download");
        return;
      }
      const blob = new Blob([result.srt], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      
      const safeFileName = (result.fileName || 'transcript').split('.')[0] || 'transcript';
      a.download = `${safeFileName}.srt`;
      
      document.body.appendChild(a);
      a.click();
      setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }, 100);
    } catch (err) {
      console.error("Download failed:", err);
      alert("Failed to download SRT file. Please check console for details.");
    }
  };

  const formatTime = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const selectedItem = queue.find(q => q.id === selectedQueueId);
  const currentResult = selectedItem?.result;

  const stats = {
    pending: queue.filter(q => q.status === 'pending').length,
    active: queue.filter(q => q.status === 'transcribing' || q.status === 'gemini-crafting').length,
    completed: queue.filter(q => q.status === 'completed').length,
    error: queue.filter(q => q.status === 'error').length,
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans selection:bg-indigo-100 pb-20">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 py-6 sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-indigo-600 p-2 rounded-lg">
              <FileAudio className="text-white w-6 h-6" />
            </div>
            <h1 className="text-xl font-bold tracking-tight">SubGen AI Pro</h1>
          </div>
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-4 text-xs font-bold uppercase tracking-widest text-slate-400">
              <span className="flex items-center gap-1.5 text-indigo-500">
                <Loader2 className="w-3.5 h-3.5 animate-spin" /> {stats.active} Active
              </span>
              <span className="flex items-center gap-1.5 text-emerald-500">
                <CheckCircle className="w-3.5 h-3.5" /> {stats.completed} Done
              </span>
            </div>
            <div className="h-6 w-px bg-slate-200" />
            <div className="text-sm text-slate-500 font-medium flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-indigo-500" />
              Hinglish Single Mode (1 at a time)
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-12">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-extrabold text-slate-900 mb-4 tracking-tight">
            High-Concurrency Subtitle Engine
          </h2>
          <p className="text-lg text-slate-600 max-w-2xl mx-auto">
            Processing files one by one to ensure maximum accuracy and stability.
          </p>
        </div>

        {/* Upload Section */}
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8 mb-12">
          <div 
            className={`border-2 border-dashed rounded-xl p-12 text-center transition-all cursor-pointer ${
              isDragging ? 'border-indigo-500 bg-indigo-50 scale-[1.01]' : 'border-slate-200 hover:border-indigo-300'
            }`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input 
              type="file" 
              ref={fileInputRef}
              onChange={handleFileChange}
              className="hidden"
              accept="audio/*,video/*"
              multiple
            />
            
            <div className="flex flex-col items-center gap-4">
              <div className={`p-4 rounded-full transition-colors ${isDragging ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-400'}`}>
                <Files className="w-8 h-8" />
              </div>
              <div>
                <p className="text-lg font-semibold text-slate-700">
                  {isDragging ? 'Drop files here' : 'Click or drag and drop audio files here'}
                </p>
                <p className="text-sm text-slate-500 mt-1">
                  Files will be added to the queue below. Click "Start Processing" to begin.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Queue & Results Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Sidebar: Queue Manager */}
          <div className="lg:col-span-4 space-y-4">
            <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm mb-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider flex items-center gap-2">
                  <List className="w-4 h-4 text-indigo-500" />
                  Queue Manager
                </h3>
                <span className="text-xs font-bold text-slate-400">{queue.length} Total</span>
              </div>
              
              {queue.length > 0 && (
                <button
                  onClick={() => setIsQueueRunning(!isQueueRunning)}
                  className={`w-full py-3 rounded-xl text-sm font-bold flex items-center justify-center gap-2 transition-all ${
                    isQueueRunning 
                      ? 'bg-red-50 text-red-600 hover:bg-red-100 border border-red-100' 
                      : 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-lg shadow-indigo-100'
                  }`}
                >
                  {isQueueRunning ? (
                    <><Pause className="w-4 h-4" /> Stop Queue</>
                  ) : (
                    <><Play className="w-4 h-4" /> Start Processing</>
                  )}
                </button>
              )}
            </div>
            
            <div className="space-y-2 max-h-[600px] overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-slate-200">
              <AnimatePresence initial={false}>
                {queue.map((item) => (
                  <motion.div
                    key={item.id}
                    layout
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    className={`group relative p-3 rounded-xl border transition-all cursor-pointer ${
                      selectedQueueId === item.id 
                        ? 'bg-indigo-600 border-indigo-600 text-white shadow-lg shadow-indigo-100' 
                        : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                    }`}
                    onClick={() => setSelectedQueueId(item.id)}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg ${
                        selectedQueueId === item.id ? 'bg-white/20' : 'bg-slate-100'
                      }`}>
                        {item.status === 'completed' ? (
                          <CheckCircle className={`w-4 h-4 ${selectedQueueId === item.id ? 'text-white' : 'text-emerald-500'}`} />
                        ) : item.status === 'error' ? (
                          <AlertCircle className="w-4 h-4 text-red-500" />
                        ) : item.status === 'pending' ? (
                          <Clock className="w-4 h-4 text-slate-400" />
                        ) : (
                          <Loader2 className={`w-4 h-4 animate-spin ${selectedQueueId === item.id ? 'text-white' : 'text-indigo-500'}`} />
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-bold truncate">{item.file.name}</p>
                        <div className="flex items-center gap-2">
                          <p className={`text-[10px] uppercase tracking-wider font-bold ${
                            selectedQueueId === item.id ? 'text-indigo-100' : 'text-slate-400'
                          }`}>
                            {item.status}
                          </p>
                          {(item.elapsedTime !== undefined || elapsedTimers[item.id] !== undefined) && (
                            <span className={`text-[10px] font-mono ${selectedQueueId === item.id ? 'text-indigo-200' : 'text-slate-400'}`}>
                              • {item.elapsedTime ?? elapsedTimers[item.id]}s
                            </span>
                          )}
                        </div>
                      </div>
                      <button 
                        onClick={(e) => { e.stopPropagation(); removeFile(item.id); }}
                        className={`opacity-0 group-hover:opacity-100 p-1 rounded-md transition-opacity ${
                          selectedQueueId === item.id ? 'hover:bg-white/20 text-white' : 'hover:bg-slate-200 text-slate-400'
                        }`}
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
              
              {queue.length === 0 && (
                <div className="text-center py-12 bg-white rounded-2xl border border-slate-200 border-dashed">
                  <p className="text-slate-400 text-sm">No files in queue</p>
                </div>
              )}
            </div>
          </div>

          {/* Main Content: Tabs & Preview */}
          <div className="lg:col-span-8">
            <AnimatePresence mode="wait">
              {selectedItem ? (
                <motion.div
                  key={selectedItem.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="space-y-6"
                >
                  <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-sm">
                    <div className="flex border-b border-slate-200 bg-slate-50/50 p-1">
                      {[
                        { id: 'ai-subtitles', label: 'AI Subtitles (Hinglish)', icon: Sparkles },
                        { id: 'transcript', label: 'Original Transcript', icon: Type },
                        { id: 'sentences', label: 'Sentences', icon: List },
                        { id: 'paragraphs', label: 'Paragraphs', icon: AlignLeft },
                      ].map((tab) => (
                        <button
                          key={tab.id}
                          onClick={() => setActiveTab(tab.id as TabType)}
                          className={`flex-1 flex items-center justify-center gap-2 py-3 px-4 rounded-xl font-bold text-sm transition-all ${
                            activeTab === tab.id 
                              ? 'bg-white text-indigo-600 shadow-sm' 
                              : 'text-slate-500 hover:text-slate-700 hover:bg-slate-100/50'
                          }`}
                        >
                          <tab.icon className={`w-4 h-4 ${activeTab === tab.id ? 'text-indigo-600' : 'text-slate-400'}`} />
                          <span className="hidden sm:inline">{tab.label}</span>
                        </button>
                      ))}
                    </div>

                    <div className="p-8 min-h-[500px]">
                      <div className="flex items-center justify-between mb-6">
                        <div>
                          <h4 className="text-xl font-bold text-slate-800 truncate max-w-[400px]">
                            {selectedItem.file.name}
                          </h4>
                          <div className="flex items-center gap-2 mt-1">
                            <span className={`text-xs font-bold uppercase tracking-wider px-2 py-0.5 rounded-full ${
                              selectedItem.status === 'completed' ? 'bg-emerald-100 text-emerald-700' :
                              selectedItem.status === 'error' ? 'bg-red-100 text-red-700' :
                              'bg-indigo-100 text-indigo-700'
                            }`}>
                              {selectedItem.status}
                            </span>
                          </div>
                        </div>
                        {currentResult?.srt && (
                          <button
                            onClick={() => downloadSRT(currentResult)}
                            className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg text-sm font-bold flex items-center gap-2 transition-colors shadow-lg shadow-indigo-100"
                          >
                            <Download className="w-4 h-4" />
                            Download SRT
                          </button>
                        )}
                      </div>

                      {selectedItem.status === 'error' ? (
                        <div className="flex flex-col items-center justify-center py-20 text-red-500 gap-4">
                          <AlertCircle className="w-12 h-12" />
                          <div className="text-center">
                            <p className="font-bold text-lg">Processing Failed</p>
                            <p className="text-sm opacity-80">{selectedItem.error}</p>
                          </div>
                        </div>
                      ) : selectedItem.status === 'transcribing' ? (
                        <div className="flex flex-col items-center justify-center py-20 text-slate-400 gap-6">
                          <div className="relative">
                            <Loader2 className="w-16 h-16 animate-spin text-indigo-500" />
                            <div className="absolute inset-0 flex items-center justify-center">
                              <FileAudio className="w-6 h-6 text-indigo-300" />
                            </div>
                          </div>
                          <div className="text-center">
                            <p className="font-bold text-lg text-slate-600">Transcribing Audio...</p>
                            <p className="text-sm mt-1">Whisper is converting your audio to text. ({elapsedTimers[selectedItem.id] || 0}s)</p>
                          </div>
                        </div>
                      ) : currentResult ? (
                        <>
                          {selectedItem.status === 'gemini-crafting' && (
                            <motion.div 
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: 'auto' }}
                              className="mb-6 bg-indigo-50 border border-indigo-100 rounded-xl p-4 flex items-center justify-between"
                            >
                              <div className="flex items-center gap-3">
                                <Loader2 className="w-5 h-5 animate-spin text-indigo-600" />
                                <p className="text-indigo-700 font-bold">Gemini is crafting Hinglish Subtitles... ({elapsedTimers[selectedItem.id] || 0}s)</p>
                              </div>
                              <p className="text-xs text-indigo-500 font-medium uppercase tracking-wider">Background Process</p>
                            </motion.div>
                          )}

                          <AnimatePresence mode="wait">
                            {activeTab === 'transcript' && (
                            <motion.div
                              key="transcript"
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              exit={{ opacity: 0, y: -10 }}
                              className="leading-relaxed text-slate-600"
                            >
                              <p className="text-lg">{currentResult.text}</p>
                            </motion.div>
                          )}

                          {activeTab === 'sentences' && (
                            <motion.div
                              key="sentences"
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              exit={{ opacity: 0, y: -10 }}
                              className="space-y-4"
                            >
                              {currentResult.sentences.map((sentence, i) => (
                                <div key={i} className="flex gap-4 p-4 rounded-xl hover:bg-slate-50 transition-colors group">
                                  <span className="text-xs font-mono text-slate-400 mt-1 flex-shrink-0 w-12">
                                    {formatTime(sentence.start)}
                                  </span>
                                  <p className="text-slate-700">{sentence.text}</p>
                                </div>
                              ))}
                            </motion.div>
                          )}

                          {activeTab === 'paragraphs' && (
                            <motion.div
                              key="paragraphs"
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              exit={{ opacity: 0, y: -10 }}
                              className="space-y-8"
                            >
                              {currentResult.paragraphs.map((para, i) => (
                                <div key={i} className="space-y-2">
                                  <span className="text-xs font-bold text-indigo-500 uppercase tracking-wider">
                                    Paragraph {i + 1} • {formatTime(para.start)}
                                  </span>
                                  <p className="text-slate-700 leading-relaxed text-lg">{para.text}</p>
                                </div>
                              ))}
                            </motion.div>
                          )}

                          {activeTab === 'ai-subtitles' && (
                            <motion.div
                              key="ai-subtitles"
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              exit={{ opacity: 0, y: -10 }}
                              className="space-y-6"
                            >
                              {currentResult.srt ? (
                                <div className="bg-slate-900 rounded-2xl p-8 shadow-inner overflow-hidden relative group">
                                  <pre className="font-mono text-sm text-slate-300 leading-relaxed overflow-y-auto max-h-[500px] scrollbar-thin scrollbar-thumb-white/10">
                                    {currentResult.srt}
                                  </pre>
                                </div>
                              ) : (
                                <div className="flex flex-col items-center justify-center py-20 text-slate-400 gap-4">
                                  <div className="relative">
                                    <Loader2 className="w-12 h-12 animate-spin text-indigo-400" />
                                    <div className="absolute inset-0 flex items-center justify-center">
                                      <Sparkles className="w-5 h-5 text-indigo-300" />
                                    </div>
                                  </div>
                                  <div className="text-center">
                                    <p className="font-bold text-slate-600">Gemini is crafting Hinglish Subtitles... ({elapsedTimers[selectedItem.id] || 0}s)</p>
                                    <p className="text-sm mt-1">Applying Hinglish conversion and numerical rules.</p>
                                  </div>
                                </div>
                              )}
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </>
                    ) : null}
                    </div>
                  </div>
                </motion.div>
              ) : (
                <div className="h-full flex flex-col items-center justify-center py-40 bg-white border border-slate-200 border-dashed rounded-2xl text-slate-400">
                  <Files className="w-16 h-16 mb-4 opacity-20" />
                  <p className="font-bold text-lg">Select a file from the queue</p>
                  <p className="text-sm">Upload files to start the automatic processing</p>
                </div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </main>

      <footer className="max-w-7xl mx-auto px-4 py-12 text-center text-slate-400 text-sm border-t border-slate-200 mt-12">
        <p>© 2026 SubGen AI Pro. All rights reserved.</p>
        <p className="mt-2">Optimized for Single File Processing • Hinglish Support Enabled</p>
      </footer>
    </div>
  );
}
