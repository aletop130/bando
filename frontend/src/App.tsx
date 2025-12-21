import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import './App.css'
import type { ChatMessage, UploadResponse, StatusResponse, ChatRequest, ChatResponse } from './types'

const API_BASE = 'http://localhost:8000'

type ProcessingStatus = 'queued' | 'processing' | 'completed' | 'error' | null

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState<boolean>(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [status, setStatus] = useState<ProcessingStatus>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputMessage, setInputMessage] = useState<string>('')
  const [loadingAnswer, setLoadingAnswer] = useState<boolean>(false)
  const [chatStarted, setChatStarted] = useState<boolean>(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = (): void => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    const selectedFile = e.target.files?.[0] || null
    setFile(selectedFile)
    setJobId(null)
    setStatus(null)
    setMessages([])
    setChatStarted(false)
  }

  const handleUpload = async (): Promise<void> => {
    if (!file) {
      alert('Seleziona un file PDF')
      return
    }

    setUploading(true)
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post<UploadResponse>(`${API_BASE}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      setJobId(response.data.job_id)
      setStatus(response.data.status as ProcessingStatus)
      
      pollStatus(response.data.job_id)
    } catch (error) {
      console.error('Errore upload:', error)
      const errorMessage = axios.isAxiosError(error) 
        ? error.message 
        : 'Errore sconosciuto durante il caricamento'
      alert('Errore durante il caricamento: ' + errorMessage)
      setUploading(false)
    }
  }

  const pollStatus = async (id: string): Promise<void> => {
    const interval = setInterval(async () => {
      try {
        const response = await axios.get<StatusResponse>(`${API_BASE}/status/${id}`)
        const newStatus = response.data
        
        setStatus(newStatus.status)
        
        if (newStatus.status === 'completed') {
          clearInterval(interval)
          setUploading(false)
          setMessages([{
            role: 'assistant',
            content: `✅ Documento processato con successo! Ho analizzato ${newStatus.chunks_count || 'N/A'} sezioni. Ora puoi farmi domande sul bando.`
          }])
          setChatStarted(true)
        } else if (newStatus.status === 'error') {
          clearInterval(interval)
          setUploading(false)
          alert('Errore durante il processing: ' + (newStatus.error || 'Errore sconosciuto'))
        }
      } catch (error) {
        console.error('Errore polling status:', error)
        clearInterval(interval)
        setUploading(false)
      }
    }, 2000)
  }

  const handleSendMessage = async (): Promise<void> => {
    if (!inputMessage.trim() || loadingAnswer) return

    const userMessage: ChatMessage = {
      role: 'user',
      content: inputMessage.trim()
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setLoadingAnswer(true)

    try {
      const allMessages: ChatMessage[] = [...messages, userMessage]

      const request: ChatRequest = {
        messages: allMessages,
        collection_name: 'Bandi'
      }

      const response = await axios.post<ChatResponse>(`${API_BASE}/chat`, request)

      setMessages(prev => [...prev, response.data.message])
    } catch (error) {
      console.error('Errore chat:', error)
      const errorMessage = axios.isAxiosError(error)
        ? error.response?.data?.detail || error.message
        : 'Errore sconosciuto'
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '❌ Errore durante l\'elaborazione: ' + errorMessage
      }])
    } finally {
      setLoadingAnswer(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const startNewChat = (): void => {
    setMessages([])
    setChatStarted(false)
    setInputMessage('')
    setFile(null)
    setJobId(null)
    setStatus(null)
  }

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <div className="header-content">
            <h1>📄 GraphRAG Bandi Chat</h1>
            <p className="subtitle">Carica un bando PDF e chatta con l'AI per ottenere informazioni specifiche</p>
          </div>
        </header>

        <main className="main-content">
          {/* Upload Section */}
          <section className="upload-card card">
            <div className="card-header">
              <h2>📤 Carica Nuovo Documento</h2>
              <span className="card-subtitle">(Opzionale - puoi anche chattare con documenti esistenti)</span>
            </div>
            <div className="upload-section">
              <div className="file-input-container">
                <label className="file-input-label">
                  <input
                    type="file"
                    accept=".pdf"
                    onChange={handleFileChange}
                    disabled={uploading}
                    className="file-input"
                  />
                  <span className="file-input-custom">
                    {file ? file.name : 'Scegli file PDF...'}
                  </span>
                </label>
                <div className="upload-buttons">
                  <button
                    onClick={handleUpload}
                    disabled={uploading || !file}
                    className="btn btn-primary"
                  >
                    {uploading ? (
                      <>
                        <span className="spinner"></span>
                        Caricamento...
                      </>
                    ) : 'Carica PDF'}
                  </button>
                  {chatStarted && (
                    <button
                      onClick={startNewChat}
                      className="btn btn-secondary"
                    >
                      Nuovo Documento
                    </button>
                  )}
                </div>
              </div>

              {jobId && (
                <div className="status-section">
                  <div className="status-header">
                    <h3>Stato Elaborazione</h3>
                    <div className={`status-indicator status-${status || ''}`}>
                      {status === 'queued' && '⏳ In coda'}
                      {status === 'processing' && '⚙️ Elaborazione...'}
                      {status === 'completed' && '✅ Completato'}
                      {status === 'error' && '❌ Errore'}
                    </div>
                  </div>
                  <div className="status-details">
                    <p><strong>ID Processo:</strong> <code>{jobId}</code></p>
                    {status === 'completed' && (
                      <div className="status-completed-info">
                        <span>✅ Pronto per le domande</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </section>

          {/* Chat Section */}
          <section className="chat-card card">
            <div className="card-header">
              <h2>💬 Chat con il Bando</h2>
              <div className="chat-stats">
                {messages.length > 0 && (
                  <span className="message-count">{messages.length} messaggi</span>
                )}
              </div>
            </div>

            <div className="chat-container">
              {status === 'processing' && (
                <div className="processing-notice">
                  <div className="notice-content">
                    <span className="notice-icon">⚙️</span>
                    <div>
                      <strong>Documento in elaborazione...</strong>
                      <p>Puoi comunque chattare con i documenti già processati nel sistema.</p>
                    </div>
                  </div>
                </div>
              )}
              
              {status === null && messages.length === 0 && (
                <div className="welcome-notice">
                  <div className="notice-content">
                    <span className="notice-icon">💡</span>
                    <div>
                      <strong>Benvenuto in GraphRAG Bandi Chat!</strong>
                      <p>Carica un documento PDF per iniziare una nuova conversazione, oppure fai direttamente una domanda sui documenti già presenti nel sistema.</p>
                    </div>
                  </div>
                </div>
              )}
              
              <div className="chat-messages">
                {messages.map((msg, idx) => (
                  <div key={idx} className={`message ${msg.role}`}>
                    <div className="message-header">
                      <div className="message-avatar">
                        {msg.role === 'user' ? '👤' : '🤖'}
                      </div>
                      <span className="message-role">
                        {msg.role === 'user' ? 'Tu' : 'Assistente'}
                      </span>
                    </div>
                    <div className="message-content">
                      {msg.content}
                    </div>
                    <div className="message-time">
                      {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </div>
                  </div>
                ))}
                {loadingAnswer && (
                  <div className="message assistant">
                    <div className="message-header">
                      <div className="message-avatar">🤖</div>
                      <span className="message-role">Assistente</span>
                    </div>
                    <div className="message-content">
                      <div className="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              <div className="chat-input-container">
                <div className="input-wrapper">
                  <textarea
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyDown={handleKeyPress}
                    placeholder="Scrivi la tua domanda sul bando... (Premi Invio per inviare, Shift+Invio per andare a capo)"
                    className="chat-input"
                    rows={2}
                    disabled={loadingAnswer}
                  />
                  <button
                    onClick={handleSendMessage}
                    disabled={loadingAnswer || !inputMessage.trim()}
                    className="send-button"
                    aria-label="Invia messaggio"
                  >
                    {loadingAnswer ? (
                      <span className="sending-icon">⏳</span>
                    ) : (
                      <span className="send-icon">📤</span>
                    )}
                  </button>
                </div>
                <div className="input-hint">
                  <span>GraphRAG analizzerà il documento e risponderà alle tue domande</span>
                </div>
              </div>
            </div>
          </section>
        </main>

        <footer className="footer">
          <p>GraphRAG Bandi Chat v1.0 • Analisi documenti con AI avanzata</p>
        </footer>
      </div>
    </div>
  )
}

export default App