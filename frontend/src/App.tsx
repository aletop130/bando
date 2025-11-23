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
          // Aggiungi un messaggio di benvenuto quando il documento è pronto
          setMessages([{
            role: 'assistant',
            content: `✅ Documento processato con successo! Ho analizzato ${newStatus.chunks_count || 'N/A'} sezioni. Puoi iniziare a farmi domande sul bando.`
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
    }, 2000) // Poll ogni 2 secondi
  }

  const handleSendMessage = async (): Promise<void> => {
    if (!inputMessage.trim() || loadingAnswer) return

    if (status !== 'completed') {
      alert('Il documento deve essere processato prima di iniziare la chat')
      return
    }

    const userMessage: ChatMessage = {
      role: 'user',
      content: inputMessage.trim()
    }

    // Aggiungi il messaggio dell'utente
    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setLoadingAnswer(true)

    try {
      // Prepara tutti i messaggi per il contesto conversazionale
      const allMessages: ChatMessage[] = [...messages, userMessage]

      const request: ChatRequest = {
        messages: allMessages,
        collection_name: 'Bandi'
      }

      const response = await axios.post<ChatResponse>(`${API_BASE}/chat`, request)

      // Aggiungi la risposta dell'assistente
      setMessages(prev => [...prev, response.data.message])
    } catch (error) {
      console.error('Errore chat:', error)
      const errorMessage = axios.isAxiosError(error)
        ? error.response?.data?.detail || error.message
        : 'Errore sconosciuto'
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '❌ Errore durante l\'elaborazione della domanda: ' + errorMessage
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
        <h1>📄 GraphRAG Bandi Chat</h1>
        <p className="subtitle">Carica un bando e chattaci sopra</p>

        {/* Upload Section */}
        <div className="card">
          <h2>1. Carica Documento</h2>
          <div className="upload-section">
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileChange}
              disabled={uploading || chatStarted}
              className="file-input"
            />
            <button
              onClick={handleUpload}
              disabled={uploading || !file || chatStarted}
              className="btn btn-primary"
            >
              {uploading ? 'Caricamento...' : 'Carica PDF'}
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

          {jobId && (
            <div className="status-section">
              <p><strong>Job ID:</strong> {jobId}</p>
              <p><strong>Stato:</strong> 
                <span className={`status-badge status-${status || ''}`}>
                  {status === 'queued' && '⏳ In coda'}
                  {status === 'processing' && '⚙️ Elaborazione...'}
                  {status === 'completed' && '✅ Completato'}
                  {status === 'error' && '❌ Errore'}
                </span>
              </p>
            </div>
          )}
        </div>

        {/* Chat Section */}
        {status === 'completed' && (
          <div className="card chat-card">
            <h2>2. Chat con il Bando</h2>
            
            <div className="chat-messages">
              {messages.map((msg, idx) => (
                <div key={idx} className={`message ${msg.role}`}>
                  <div className="message-avatar">
                    {msg.role === 'user' ? '👤' : '🤖'}
                  </div>
                  <div className="message-content">
                    {msg.content}
                  </div>
                </div>
              ))}
              {loadingAnswer && (
                <div className="message assistant">
                  <div className="message-avatar">🤖</div>
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

            <div className="chat-input-section">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder="Fai una domanda sul bando... (Premi Invio per inviare, Shift+Invio per andare a capo)"
                className="chat-input"
                rows={2}
                disabled={loadingAnswer}
              />
              <button
                onClick={handleSendMessage}
                disabled={loadingAnswer || !inputMessage.trim()}
                className="btn btn-secondary send-btn"
              >
                {loadingAnswer ? '⏳' : '📤'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
